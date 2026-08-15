"""Fine-tune DAF for structural/perceptual quality and package the result."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = PROJECT / "artifacts/daf-perceptual-finetune"
ARCHIVE = Path("/content/kla-daf-perceptual.tar.gz")


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/finetune_daf_perceptual.py",
            "--checkpoint",
            str(PROJECT / "artifacts/daf-fidelity-10ep/daf_restormer/checkpoints/best.pt"),
            "--data-root",
            "/content/kla-data/extracted/train",
            "--output-dir",
            str(OUTPUT),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--learning-rate",
            "1e-5",
            "--ssim-weight",
            "0.2",
            "--perceptual-weight",
            "0.02",
        ],
        cwd=PROJECT,
        env=environment,
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-perceptual-finetune")
    print(f"DAF_PERCEPTUAL_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
