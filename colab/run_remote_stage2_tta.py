"""Benchmark self-ensemble for both full-resolution perceptual epochs."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = Path("/content/daf-stage2-tta")
ARCHIVE = Path("/content/kla-daf-stage2-tta.tar.gz")


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for epoch in (1, 2):
        subprocess.check_call(
            [
                sys.executable,
                "scripts/evaluate_self_ensemble.py",
                "--checkpoint",
                str(PROJECT / f"artifacts/daf-perceptual-stage2/epoch-{epoch}.pt"),
                "--data-root",
                "/content/kla-data/extracted/train",
                "--output",
                str(OUTPUT / f"epoch-{epoch}.json"),
            ],
            cwd=PROJECT,
            env=environment,
        )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-stage2-tta")
    print(f"DAF_STAGE2_TTA_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
