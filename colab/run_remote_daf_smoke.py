"""Run a bounded GPU/AMP smoke test for DAF-Restormer and package it."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = PROJECT / "artifacts" / "daf-gpu-smoke"
ARCHIVE = Path("/content/kla-daf-gpu-smoke.tar.gz")


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "benchmark_all.py",
            "--models",
            "daf_restormer",
            "--data-root",
            "/content/kla-data/extracted/train",
            "--output-dir",
            str(OUTPUT),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--num-workers",
            "2",
            "--train-limit",
            "256",
            "--val-limit",
            "64",
            "--frequency-weight",
            "0.03",
            "--gradient-weight",
            "0.02",
            "--synthetic-probability",
            "0.35",
            "--skip-lpips",
        ],
        cwd=PROJECT,
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-gpu-smoke")
    print(f"DAF_GPU_SMOKE_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
