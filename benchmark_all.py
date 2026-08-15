#!/usr/bin/env python3
"""Benchmark bicubic and all requested restoration model families."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.benchmark import BenchmarkSettings, run_benchmarks


DEFAULT_DATA = (
    Path(__file__).resolve().parent
    / "hackathon"
    / "semicon-image-restoration-hackathon-2026"
    / "dataset"
    / "extracted"
    / "train"
)


def parse_optional_limit(value: int) -> int | None:
    return None if value <= 0 else value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument(
        "--models",
        default="bicubic,nafnet,swinir,restormer,esrgan_rrdb,mprnet,daf_restormer",
        help="Comma-separated model names",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--ssim-weight", type=float, default=0.1)
    parser.add_argument("--frequency-weight", type=float, default=0.0)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument("--uncertainty-weight", type=float, default=0.01)
    parser.add_argument("--synthetic-probability", type=float, default=0.0)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-lpips", action="store_true")
    args = parser.parse_args()

    settings = BenchmarkSettings(
        data_root=str(args.data_root.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_workers=args.num_workers,
        train_limit=parse_optional_limit(args.train_limit),
        val_limit=parse_optional_limit(args.val_limit),
        ssim_loss_weight=args.ssim_weight,
        frequency_loss_weight=args.frequency_weight,
        gradient_loss_weight=args.gradient_weight,
        auxiliary_loss_weight=args.auxiliary_weight,
        uncertainty_loss_weight=args.uncertainty_weight,
        synthetic_degradation_probability=args.synthetic_probability,
        amp=not args.no_amp,
        compute_lpips=not args.skip_lpips,
    )
    models = [name.strip() for name in args.models.split(",") if name.strip()]
    summary = run_benchmarks(models, settings)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
