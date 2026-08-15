#!/usr/bin/env python3
"""Fine-tune a DAF checkpoint for structural and perceptual fidelity."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

from datasets.paired_dataset import get_dataloaders
from engine.benchmark import evaluate_model, seed_everything
from losses import CombinedRestorationLoss
from metrics import LPIPSMetric
from models import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--ssim-weight", type=float, default=0.2)
    parser.add_argument("--perceptual-weight", type=float, default=0.02)
    parser.add_argument("--perceptual-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    train_loader, val_loader = get_dataloaders(
        str(args.data_root / "NoisyLR"),
        str(args.data_root / "GT"),
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=2,
    )
    criterion = CombinedRestorationLoss(
        ssim_weight=args.ssim_weight,
        perceptual_weight=args.perceptual_weight,
        perceptual_max_size=args.perceptual_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lpips_metric = LPIPSMetric(device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for noisy, target, _ in train_loader:
            noisy = noisy.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(noisy)
            loss = criterion(prediction.float(), target.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * noisy.shape[0]
            seen += noisy.shape[0]
        scheduler.step()
        metrics = evaluate_model(model, val_loader, device, use_amp, lpips_metric=lpips_metric)
        record = {
            "epoch": epoch,
            "train_loss": loss_sum / seen,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "metrics": metrics,
        }
        history.append(record)
        torch.save(
            {
                "model_name": checkpoint["model_name"],
                "model_state_dict": model.state_dict(),
                "epoch": checkpoint.get("epoch", 0) + epoch,
                "validation": metrics,
                "fine_tune": {
                    "source": str(args.checkpoint),
                    "ssim_weight": args.ssim_weight,
                    "perceptual_weight": args.perceptual_weight,
                    "perceptual_size": args.perceptual_size,
                    "learning_rate": args.learning_rate,
                    "epochs": args.epochs,
                },
            },
            args.output_dir / f"epoch-{epoch}.pt",
        )
        print(json.dumps(record), flush=True)

    # Select without pretending that unpublished KLA metric weights are known:
    # prefer candidates that dominate the 10-epoch Restormer on all metrics;
    # otherwise retain the highest balanced normalized improvement.
    reference = {"psnr": 28.393100327253343, "ssim": 0.7709464966785162, "lpips": 0.27190778822405265}
    dominating = [
        row
        for row in history
        if row["metrics"]["psnr_mean_db"] > reference["psnr"]
        and row["metrics"]["ssim_mean"] > reference["ssim"]
        and row["metrics"]["lpips_mean"] < reference["lpips"]
    ]

    def balanced_score(row: dict) -> float:
        metrics = row["metrics"]
        return (
            (metrics["psnr_mean_db"] - reference["psnr"]) / 0.1
            + (metrics["ssim_mean"] - reference["ssim"]) / 0.002
            + (reference["lpips"] - metrics["lpips_mean"]) / 0.005
        )

    selected = max(dominating or history, key=balanced_score)
    selected_path = args.output_dir / f"epoch-{selected['epoch']}.pt"
    best_path = args.output_dir / "best.pt"
    shutil.copy2(selected_path, best_path)
    result = {
        "status": "completed",
        "source_checkpoint": str(args.checkpoint),
        "reference": reference,
        "selection": "all-metric dominance, else balanced normalized improvement",
        "dominates_reference": bool(dominating),
        "selected_epoch": selected["epoch"],
        "selected_metrics": selected["metrics"],
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "peak_vram_mib": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None,
        "checkpoint": str(best_path),
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
