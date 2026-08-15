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
    args = parser.parse_args()
    print(json.dumps(restore_directory(args.input, args.output, args.weights, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
