#!/usr/bin/env python3
"""Select scalar residual calibration for a DAF geometric self-ensemble."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_msssim import ssim

from datasets.paired_dataset import get_dataloaders
from engine.benchmark import _batch_psnr, benchmark_latency
from metrics import LPIPSMetric
from models import build_model
from models.self_ensemble import GeometricSelfEnsemble, ResidualCalibrator


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transforms", type=int, choices=(1, 4, 8), default=8)
    parser.add_argument("--minimum-ssim", type=float, default=0.7709464966785162)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    _, loader = get_dataloaders(
        str(args.data_root / "NoisyLR"),
        str(args.data_root / "GT"),
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=2,
    )
    base_model = load_model(args.checkpoint, device)
    ensemble = GeometricSelfEnsemble(base_model, transforms=args.transforms).to(device).eval()
    gains = np.linspace(0.90, 1.10, 9).tolist()
    biases = np.linspace(-0.002, 0.002, 5).tolist()
    candidates = [(float(gain), float(bias)) for gain in gains for bias in biases]
    accumulators = {candidate: {"psnr": [], "ssim": []} for candidate in candidates}

    for noisy, target, _ in loader:
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True).float()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = ensemble(noisy).float()
        bicubic = F.interpolate(noisy.float(), scale_factor=2, mode="bicubic", align_corners=False)
        residual = prediction - bicubic
        for candidate in candidates:
            gain, bias = candidate
            calibrated = bicubic + gain * residual + bias
            accumulators[candidate]["psnr"].extend(_batch_psnr(calibrated, target).cpu().tolist())
            accumulators[candidate]["ssim"].extend(
                ssim(calibrated.clamp(0, 1), target.clamp(0, 1), data_range=1.0, size_average=False)
                .cpu()
                .tolist()
            )

    surface = [
        {
            "residual_gain": gain,
            "bias": bias,
            "psnr_mean_db": float(np.mean(accumulators[(gain, bias)]["psnr"])),
            "ssim_mean": float(np.mean(accumulators[(gain, bias)]["ssim"])),
        }
        for gain, bias in candidates
    ]
    eligible = [row for row in surface if row["ssim_mean"] >= args.minimum_ssim]
    selected = max(eligible or surface, key=lambda row: row["psnr_mean_db"])
    selected["met_ssim_constraint"] = bool(eligible)

    calibrated_model = ResidualCalibrator(
        ensemble,
        residual_gain=selected["residual_gain"],
        bias=selected["bias"],
    ).to(device).eval()
    lpips = LPIPSMetric(device)
    lpips_values = []
    for noisy, target, _ in loader:
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            calibrated = calibrated_model(noisy)
        lpips_values.extend(lpips(calibrated.float(), target.float()).cpu().tolist())
    selected["lpips_mean"] = float(np.mean(lpips_values))

    result = {
        "checkpoint": str(args.checkpoint),
        "transforms": args.transforms,
        "selection_rule": f"highest PSNR with SSIM >= {args.minimum_ssim}",
        "selected": selected,
        "latency": benchmark_latency(calibrated_model, device, use_amp, warmup=5, iterations=30),
        "surface": surface,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
