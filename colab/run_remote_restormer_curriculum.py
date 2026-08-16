"""Run 2-stage patch-to-full curriculum fine-tuning on 1.6M Compact Restormer."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path

PROJECT = Path("/content/image-restoration-project")
DATA_ROOT = Path("/content/kla-data/extracted/train")

STAGE1_OUTPUT = PROJECT / "artifacts" / "restormer-curriculum-stage1"
STAGE2_OUTPUT = PROJECT / "artifacts" / "restormer-curriculum-stage2"
ARCHIVE = Path("/content/kla-restormer-curriculum.tar.gz")

# Base pre-trained Restormer checkpoint if available
BASE_CHECKPOINT = PROJECT / "artifacts" / "remote-runs" / "colab-full" / "restormer" / "checkpoints" / "best.pt"


def main() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)

    print("=== STARTING STAGE 1: Patch Curriculum (64x64 patches, 70/30 synthetic mix) ===")
    stage1_cmd = [
        sys.executable,
        "train.py",
        "--model",
        "restormer",
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(STAGE1_OUTPUT),
        "--epochs",
        "10",
        "--batch-size",
        "16",
        "--learning-rate",
        "1e-4",
        "--synthetic-probability",
        "0.30",
        "--crop-size",
        "64",
        "--eta-min",
        "1e-7",
    ]
    if BASE_CHECKPOINT.exists():
        stage1_cmd.extend(["--resume-checkpoint", str(BASE_CHECKPOINT)])

    subprocess.check_call(stage1_cmd, cwd=PROJECT, env=environment)

    stage1_checkpoint = STAGE1_OUTPUT / "restormer" / "checkpoints" / "best.pt"
    if not stage1_checkpoint.exists():
        raise FileNotFoundError(f"Stage 1 checkpoint not found at {stage1_checkpoint}")

    print("\n=== STARTING STAGE 2: Full-Image Perceptual Fine-Tuning (Full 128x128, mild LPIPS) ===")
    stage2_cmd = [
        sys.executable,
        "train.py",
        "--model",
        "restormer",
        "--data-root",
        str(DATA_ROOT),
        "--output-dir",
        str(STAGE2_OUTPUT),
        "--epochs",
        "5",
        "--batch-size",
        "8",
        "--learning-rate",
        "3e-5",
        "--synthetic-probability",
        "0.30",
        "--perceptual-weight",
        "0.01",
        "--eta-min",
        "1e-7",
        "--resume-checkpoint",
        str(stage1_checkpoint),
    ]

    subprocess.check_call(stage2_cmd, cwd=PROJECT, env=environment)

    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(STAGE2_OUTPUT, arcname="restormer-curriculum-stage2")

    print(f"\nRESTORMER_CURRICULUM_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
