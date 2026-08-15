#!/usr/bin/env python3
"""Evaluate fixed Restormer/DAF fusion weights on the held-out split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from pytorch_msssim import ssim

from datasets.paired_dataset import get_dataloaders
from engine.benchmark import _batch_psnr, benchmark_latency
from metrics import LPIPSMetric
from models import build_model


class FixedFusion(torch.nn.Module):
    def __init__(self, first: torch.nn.Module, second: torch.nn.Module, second_weight: float):
        super().__init__()
        self.first = first
        self.second = second
        self.second_weight = second_weight

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.lerp(self.first(image), self.second(image), self.second_weight)


def load_model(path: Path, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restormer", type=Path, required=True)
    parser.add_argument("--daf", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--ssim-tolerance", type=float, default=0.0005)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    restormer = load_model(args.restormer, device)
    daf = load_model(args.daf, device)
    _, loader = get_dataloaders(
        str(args.data_root / "NoisyLR"),
        str(args.data_root / "GT"),
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=2,
    )
    weights = np.arange(0.0, 1.0 + args.step / 2.0, args.step).clip(0.0, 1.0).tolist()
    accumulators = {weight: {"psnr": [], "ssim": []} for weight in weights}

    for noisy, target, _ in loader:
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            restormer_output = restormer(noisy)
            daf_output = daf(noisy)
        for weight in weights:
            prediction = torch.lerp(restormer_output, daf_output, weight).float()
            accumulators[weight]["psnr"].extend(_batch_psnr(prediction, target).cpu().tolist())
            accumulators[weight]["ssim"].extend(
                ssim(
                    prediction.clamp(0, 1),
                    target.float().clamp(0, 1),
                    data_range=1.0,
                    size_average=False,
                ).cpu().tolist()
            )

    curve = [
        {
            "daf_weight": weight,
            "psnr_mean_db": float(np.mean(accumulators[weight]["psnr"])),
            "ssim_mean": float(np.mean(accumulators[weight]["ssim"])),
        }
        for weight in weights
    ]
    restormer_ssim = curve[0]["ssim_mean"]
    eligible = [row for row in curve if row["ssim_mean"] >= restormer_ssim - args.ssim_tolerance]
    selected = max(eligible, key=lambda row: row["psnr_mean_db"])

    fusion = FixedFusion(restormer, daf, selected["daf_weight"]).to(device).eval()
    lpips_metric = LPIPSMetric(device)
    lpips_values = []
    for noisy, target, _ in loader:
        noisy = noisy.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = fusion(noisy)
        lpips_values.extend(lpips_metric(prediction.float(), target.float()).cpu().tolist())

    result = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "restormer_checkpoint": str(args.restormer),
        "daf_checkpoint": str(args.daf),
        "selection_rule": f"highest PSNR with SSIM no more than {args.ssim_tolerance} below Restormer",
        "selected": {**selected, "lpips_mean": float(np.mean(lpips_values))},
        "latency": benchmark_latency(fusion, device, use_amp),
        "curve": curve,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
