#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from engine.benchmark import BenchmarkSettings, run_benchmarks


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one adapted restoration model")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/training"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frequency-weight", type=float, default=0.0)
    parser.add_argument("--gradient-weight", type=float, default=0.0)
    parser.add_argument("--auxiliary-weight", type=float, default=0.2)
    parser.add_argument("--uncertainty-weight", type=float, default=0.01)
    parser.add_argument("--synthetic-probability", type=float, default=0.0)
    args = parser.parse_args()
    settings = BenchmarkSettings(
        data_root=str(args.data_root.resolve()),
        output_dir=str(args.output_dir.resolve()),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        frequency_loss_weight=args.frequency_weight,
        gradient_loss_weight=args.gradient_weight,
        auxiliary_loss_weight=args.auxiliary_weight,
        uncertainty_loss_weight=args.uncertainty_weight,
        synthetic_degradation_probability=args.synthetic_probability,
    )
    run_benchmarks([args.model], settings)


if __name__ == "__main__":
    main()
