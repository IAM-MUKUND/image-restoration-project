#!/usr/bin/env python3
"""Submission entrypoint script for KLA Problem Statement - AI-Based Restoration of Degraded Images.

Usage:
    python run.py <input-dir> <output-dir>
    python run.py --input <input-dir> --output <output-dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from engine.inference import restore_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KLA Problem Statement: AI-Based Restoration of Degraded Images (SEMICON Hackathon 2026)"
    )
    # Support both positional arguments (as mandated by hackathon submission guidelines)
    # and optional --input/--output flags.
    parser.add_argument(
        "pos_input",
        nargs="?",
        type=Path,
        help="Input directory containing degraded .npy files",
    )
    parser.add_argument(
        "pos_output",
        nargs="?",
        type=Path,
        help="Output directory to save restored .npy files",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        dest="flag_input",
        help="Input directory containing degraded .npy files",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        dest="flag_output",
        help="Output directory to save restored .npy files",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=Path,
        default=None,
        help="Path to model checkpoint weights (.pt)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference (default: 16)",
    )
    parser.add_argument(
        "--self-ensemble",
        type=int,
        choices=(1, 4, 8),
        default=8,
        help="Geometric prediction views to average (default: 8 for top accuracy, 1 for low latency)",
    )

    args = parser.parse_args()

    input_path = args.pos_input or args.flag_input
    output_path = args.pos_output or args.flag_output

    if not input_path or not output_path:
        parser.error("Both input directory and output directory must be provided.\nExample: python run.py <input-dir> <output-dir>")

    args.resolved_input = input_path
    args.resolved_output = output_path
    return args


def main() -> None:
    args = parse_args()
    summary = restore_directory(
        input_dir=args.resolved_input,
        output_dir=args.resolved_output,
        weights=args.weights,
        batch_size=args.batch_size,
        self_ensemble_transforms=args.self_ensemble,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
