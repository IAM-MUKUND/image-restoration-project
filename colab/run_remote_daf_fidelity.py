"""Isolate DAF architecture gains using the original Restormer fidelity loss."""

from __future__ import annotations

import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = PROJECT / "artifacts" / "daf-fidelity-10ep"
ARCHIVE = Path("/content/kla-daf-fidelity-10ep.tar.gz")


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
            "10",
            "--batch-size",
            "8",
            "--num-workers",
            "2",
            "--frequency-weight",
            "0",
            "--gradient-weight",
            "0",
            "--auxiliary-weight",
            "0",
            "--uncertainty-weight",
            "0",
            "--synthetic-probability",
            "0",
        ],
        cwd=PROJECT,
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-fidelity-10ep")
    print(f"DAF_FIDELITY_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
