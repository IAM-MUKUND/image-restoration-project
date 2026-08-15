"""Run the fixed-budget full-data benchmark and package weights/results."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = PROJECT / "artifacts" / "colab-full"
ARCHIVE = Path("/content/kla-benchmark-full.tar.gz")


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "benchmark_all.py",
            "--data-root",
            "/content/kla-data/extracted/train",
            "--output-dir",
            str(OUTPUT),
            "--epochs",
            "3",
            "--batch-size",
            "8",
            "--num-workers",
            "2",
        ],
        cwd=PROJECT,
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="colab-full")
    print(f"FULL_BENCHMARK_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
