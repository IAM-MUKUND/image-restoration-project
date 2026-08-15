"""Run and package the DAF residual calibration search."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = Path("/content/daf-calibration")
ARCHIVE = Path("/content/kla-daf-calibration.tar.gz")


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/evaluate_calibration.py",
            "--checkpoint",
            str(PROJECT / "artifacts/daf-fidelity-10ep/daf_restormer/checkpoints/best.pt"),
            "--data-root",
            "/content/kla-data/extracted/train",
            "--output",
            str(OUTPUT / "calibration.json"),
            "--transforms",
            "8",
        ],
        cwd=PROJECT,
        env=environment,
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-calibration")
    print(f"DAF_CALIBRATION_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
