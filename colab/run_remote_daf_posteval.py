"""Run DAF fidelity checkpoint self-ensemble and fusion evaluations."""

from __future__ import annotations

import os
import subprocess
import sys
import tarfile
from pathlib import Path


PROJECT = Path("/content/image-restoration-project")
DATA = Path("/content/kla-data/extracted/train")
DAF = PROJECT / "artifacts/daf-fidelity-10ep/daf_restormer/checkpoints/best.pt"
RESTORMER = Path("/content/restormer-epoch8.pt")
OUTPUT = Path("/content/daf-posteval")
ARCHIVE = Path("/content/kla-daf-posteval.tar.gz")


def run(*arguments: str) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT)
    subprocess.check_call([sys.executable, *arguments], cwd=PROJECT, env=environment)


def main() -> None:
    if not DAF.exists() or not RESTORMER.exists():
        raise FileNotFoundError(f"Missing checkpoint: daf={DAF.exists()} restormer={RESTORMER.exists()}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run(
        "scripts/evaluate_self_ensemble.py",
        "--checkpoint", str(DAF),
        "--data-root", str(DATA),
        "--output", str(OUTPUT / "self_ensemble.json"),
    )
    run(
        "scripts/evaluate_fusion.py",
        "--restormer", str(RESTORMER),
        "--daf", str(DAF),
        "--data-root", str(DATA),
        "--output", str(OUTPUT / "fusion.json"),
    )
    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-posteval")
    print(f"DAF_POSTEVAL_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size}")


if __name__ == "__main__":
    main()
