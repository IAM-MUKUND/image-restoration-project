#!/usr/bin/env python3
"""Compare direct and geometric self-ensemble inference on the held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.paired_dataset import get_dataloaders
from engine.benchmark import benchmark_latency, evaluate_model
from metrics import LPIPSMetric
from models import build_model
from models.self_ensemble import GeometricSelfEnsemble


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
    base = load_model(args.checkpoint, device)
    lpips_metric = LPIPSMetric(device)
    variants = {}
    for transforms in (1, 4, 8):
        model = GeometricSelfEnsemble(base, transforms=transforms).to(device).eval()
        quality = evaluate_model(model, loader, device, use_amp, lpips_metric=lpips_metric)
        variants[f"x{transforms}"] = {
            **quality,
            **benchmark_latency(model, device, use_amp, warmup=5, iterations=30),
        }

    direct = variants["x1"]
    for name, metrics in variants.items():
        metrics["delta_vs_x1"] = {
            "psnr_mean_db": metrics["psnr_mean_db"] - direct["psnr_mean_db"],
            "ssim_mean": metrics["ssim_mean"] - direct["ssim_mean"],
            "lpips_mean": metrics["lpips_mean"] - direct["lpips_mean"],
        }

    result = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
