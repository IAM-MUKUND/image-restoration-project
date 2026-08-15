#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from datasets.paired_dataset import get_dataloaders
from engine.benchmark import evaluate_model
from metrics import LPIPSMetric
from models import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on paired .npy data")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-predictions", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["model_name"])
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _, loader = get_dataloaders(
        str(args.data_root / "NoisyLR"),
        str(args.data_root / "GT"),
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=2,
    )
    metrics = evaluate_model(
        model,
        loader,
        device,
        use_amp=device.type == "cuda",
        lpips_metric=LPIPSMetric(device),
        prediction_dir=args.output_predictions,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
