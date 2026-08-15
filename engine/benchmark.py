"""Unified training, quality evaluation, and latency benchmarking."""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from pytorch_msssim import ssim
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.paired_dataset import get_dataloaders
from losses import CombinedRestorationLoss
from metrics import LPIPSMetric
from models import MODEL_NAMES, build_model


MODEL_NOTES = {
    "bicubic": "Non-learning interpolation floor.",
    "nafnet": "Compact NAFNet with a late 2x PixelShuffle head and bicubic residual.",
    "swinir": "Compact shifted-window SwinIR adaptation for joint denoising and 2x SR.",
    "restormer": "Compact two-level Restormer adaptation with a late 2x head.",
    "esrgan_rrdb": "ESRGAN RRDB generator trained in fidelity/PSNR mode; no discriminator or GAN loss.",
    "mprnet": "Compact three-stage progressive supervised-attention adaptation at 256x256.",
    "daf_restormer": (
        "Degradation-aware frequency Restormer with prompt modulation, spatial noise mapping, "
        "progressive clean-LR supervision, and uncertainty estimation."
    ),
}


@dataclass
class BenchmarkSettings:
    data_root: str
    output_dir: str
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    seed: int = 42
    num_workers: int = 2
    train_limit: int | None = None
    val_limit: int | None = None
    ssim_loss_weight: float = 0.1
    frequency_loss_weight: float = 0.0
    gradient_loss_weight: float = 0.0
    auxiliary_loss_weight: float = 0.2
    uncertainty_loss_weight: float = 0.01
    perceptual_loss_weight: float = 0.0
    synthetic_degradation_probability: float = 0.0
    amp: bool = True
    compute_lpips: bool = True


class BicubicBaseline(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_record(device: torch.device) -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "git_revision": git_revision(),
    }


def _batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    mse = (prediction - target).square().flatten(1).mean(1)
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


@torch.inference_mode()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool,
    lpips_metric: LPIPSMetric | None = None,
    prediction_dir: Path | None = None,
    max_saved_predictions: int = 8,
) -> dict[str, float | int | None]:
    model.eval()
    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []
    saved = 0
    wall_start = time.perf_counter()

    if prediction_dir is not None:
        prediction_dir.mkdir(parents=True, exist_ok=True)

    for noisy, target, names in dataloader:
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(noisy)
        prediction_float = prediction.float()
        psnr_values.extend(_batch_psnr(prediction_float, target).cpu().tolist())
        batch_ssim = ssim(
            prediction_float.clamp(0.0, 1.0),
            target.float().clamp(0.0, 1.0),
            data_range=1.0,
            size_average=False,
        )
        ssim_values.extend(batch_ssim.cpu().tolist())
        if lpips_metric is not None:
            lpips_values.extend(lpips_metric(prediction_float, target.float()).cpu().tolist())
        if prediction_dir is not None and saved < max_saved_predictions:
            clipped = prediction_float.clamp(0.0, 1.0).cpu().numpy()
            for index, name in enumerate(names):
                if saved >= max_saved_predictions:
                    break
                np.save(prediction_dir / f"{name}.npy", clipped[index, 0].astype(np.float32))
                saved += 1

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - wall_start
    return {
        "samples": len(psnr_values),
        "psnr_mean_db": float(np.mean(psnr_values)),
        "psnr_median_db": float(np.median(psnr_values)),
        "psnr_worst_decile_db": float(np.percentile(psnr_values, 10)),
        "ssim_mean": float(np.mean(ssim_values)),
        "lpips_mean": float(np.mean(lpips_values)) if lpips_values else None,
        "validation_wall_seconds": elapsed,
        "validation_images_per_second": len(psnr_values) / elapsed,
    }


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    device: torch.device,
    use_amp: bool,
    warmup: int = 20,
    iterations: int = 100,
) -> dict[str, float]:
    model.eval()
    sample = torch.zeros(1, 1, 128, 128, device=device)
    if device.type != "cuda":
        warmup = min(warmup, 3)
        iterations = min(iterations, 10)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for _ in range(warmup):
            model(sample)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        timings = []
        for _ in range(iterations):
            start.record()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                model(sample)
            end.record()
            torch.cuda.synchronize(device)
            timings.append(start.elapsed_time(end))
    else:
        timings = []
        for _ in range(iterations):
            begin = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - begin) * 1000.0)

    return {
        "latency_median_ms_batch1": float(np.median(timings)),
        "latency_p90_ms_batch1": float(np.percentile(timings, 90)),
        "latency_iterations": iterations,
    }


def train_one_model(
    name: str,
    settings: BenchmarkSettings,
    device: torch.device,
    lpips_metric: LPIPSMetric | None,
) -> dict[str, object]:
    seed_everything(settings.seed)
    data_root = Path(settings.data_root)
    train_loader, val_loader = get_dataloaders(
        str(data_root / "NoisyLR"),
        str(data_root / "GT"),
        batch_size=settings.batch_size,
        seed=settings.seed,
        num_workers=settings.num_workers,
        train_limit=settings.train_limit,
        val_limit=settings.val_limit,
        synthetic_degradation_probability=settings.synthetic_degradation_probability,
    )
    model_dir = Path(settings.output_dir) / name
    checkpoint_dir = model_dir / "checkpoints"
    prediction_dir = model_dir / "predictions"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(name).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    criterion = CombinedRestorationLoss(
        ssim_weight=settings.ssim_loss_weight,
        frequency_weight=settings.frequency_loss_weight,
        gradient_weight=settings.gradient_loss_weight,
        auxiliary_weight=settings.auxiliary_loss_weight,
        uncertainty_weight=settings.uncertainty_loss_weight,
        perceptual_weight=settings.perceptual_loss_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(settings.epochs, 1))
    use_amp = settings.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history: list[dict[str, float | int]] = []
    best_psnr = -math.inf
    best_path = checkpoint_dir / "best.pt"
    train_start = time.perf_counter()
    peak_vram = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, settings.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        progress = tqdm(train_loader, desc=f"{name} epoch {epoch}/{settings.epochs}", leave=False)
        for noisy, target, _ in progress:
            noisy = noisy.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if hasattr(model, "forward_with_aux"):
                    outputs = model.forward_with_aux(noisy)
                    prediction = outputs["prediction"]
                else:
                    outputs = None
                    prediction = model(noisy)
            # FFT and fixed Sobel kernels are evaluated in float32 outside AMP.
            # This also avoids CUDA dtype mismatches for functional convolutions.
            loss = criterion(
                prediction.float(),
                target.float(),
                clean_lr=outputs.get("clean_lr").float() if outputs is not None else None,
                uncertainty=outputs.get("uncertainty").float() if outputs is not None else None,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch = noisy.shape[0]
            running_loss += float(loss.item()) * batch
            seen += batch
            progress.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        validation = evaluate_model(model, val_loader, device, use_amp)
        record = {
            "epoch": epoch,
            "train_loss": running_loss / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_psnr_mean_db": float(validation["psnr_mean_db"]),
            "val_ssim_mean": float(validation["ssim_mean"]),
        }
        history.append(record)
        print(json.dumps({"model": name, **record}), flush=True)
        if record["val_psnr_mean_db"] > best_psnr:
            best_psnr = record["val_psnr_mean_db"]
            torch.save(
                {
                    "model_name": name,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "settings": asdict(settings),
                },
                best_path,
            )

    training_seconds = time.perf_counter() - train_start
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = evaluate_model(
        model,
        val_loader,
        device,
        use_amp,
        lpips_metric=lpips_metric,
        prediction_dir=prediction_dir,
    )
    latency = benchmark_latency(model, device, use_amp)
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)
    result: dict[str, object] = {
        "status": "completed",
        "model": name,
        "notes": MODEL_NOTES[name],
        "parameters": parameters,
        "checkpoint": str(best_path),
        "checkpoint_bytes": best_path.stat().st_size,
        "best_epoch": checkpoint["epoch"],
        "training_seconds": training_seconds,
        "peak_vram_mib": peak_vram,
        "history": history,
        "metrics": final_metrics,
        **latency,
    }
    (model_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluate_bicubic(
    settings: BenchmarkSettings,
    device: torch.device,
    lpips_metric: LPIPSMetric | None,
) -> dict[str, object]:
    seed_everything(settings.seed)
    data_root = Path(settings.data_root)
    _, val_loader = get_dataloaders(
        str(data_root / "NoisyLR"),
        str(data_root / "GT"),
        batch_size=settings.batch_size,
        seed=settings.seed,
        num_workers=settings.num_workers,
        train_limit=settings.train_limit,
        val_limit=settings.val_limit,
    )
    model = BicubicBaseline().to(device)
    model_dir = Path(settings.output_dir) / "bicubic"
    result: dict[str, object] = {
        "status": "completed",
        "model": "bicubic",
        "notes": MODEL_NOTES["bicubic"],
        "parameters": 0,
        "checkpoint": None,
        "checkpoint_bytes": 0,
        "best_epoch": None,
        "training_seconds": 0.0,
        "peak_vram_mib": 0.0,
        "history": [],
        "metrics": evaluate_model(
            model,
            val_loader,
            device,
            use_amp=False,
            lpips_metric=lpips_metric,
            prediction_dir=model_dir / "predictions",
        ),
        **benchmark_latency(model, device, use_amp=False),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_benchmarks(model_names: Iterable[str], settings: BenchmarkSettings) -> dict[str, object]:
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_lpips = settings.compute_lpips
    lpips_metric = LPIPSMetric(device) if use_lpips else None
    summary: dict[str, object] = {
        "settings": asdict(settings),
        "environment": environment_record(device),
        "results": {},
    }
    summary_path = output_dir / "benchmark_summary.json"

    for name in model_names:
        print(f"=== Benchmarking {name} ===", flush=True)
        try:
            if name == "bicubic":
                result = evaluate_bicubic(settings, device, lpips_metric)
            else:
                if name not in MODEL_NAMES:
                    raise KeyError(f"Unknown model: {name}")
                result = train_one_model(name, settings, device, lpips_metric)
        except Exception as error:
            result = {
                "status": "failed",
                "model": name,
                "error_type": type(error).__name__,
                "error": str(error),
            }
            print(json.dumps(result), flush=True)
        summary["results"][name] = result
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary
