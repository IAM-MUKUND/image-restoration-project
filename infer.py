#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from engine.inference import restore_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore all KLA .npy inputs in a directory")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--self-ensemble",
        type=int,
        choices=(1, 4, 8),
        default=1,
        help="Geometric prediction views to average (1 disables self-ensemble)",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            restore_directory(
                args.input,
                args.output,
                args.weights,
                args.batch_size,
                self_ensemble_transforms=args.self_ensemble,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
