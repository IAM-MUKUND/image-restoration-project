"""Continue DAF fine-tuning with full-resolution structural/perceptual loss."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
OUTPUT = PROJECT / "artifacts/daf-perceptual-stage2"
ARCHIVE = Path("/content/kla-daf-perceptual-stage2.tar.gz")


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)
    subprocess.check_call(
        [
            sys.executable,
            "scripts/finetune_daf_perceptual.py",
            "--checkpoint",
            str(PROJECT / "artifacts/daf-perceptual-finetune/best.pt"),
            "--data-root",
            "/content/kla-data/extracted/train",
            "--output-dir",
            str(OUTPUT),
            "--epochs",
            "2",
            "--batch-size",
            "8",
            "--learning-rate",
            "3e-6",
            "--ssim-weight",
            "0.25",
            "--perceptual-weight",
            "0.05",
            "--perceptual-size",
            "256",
        ],
        cwd=PROJECT,
        env=environment,
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-perceptual-stage2")
    print(f"DAF_PERCEPTUAL_STAGE2_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
