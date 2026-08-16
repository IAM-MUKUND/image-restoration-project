"""Prepare a persistent Colab session from an uploaded source archive."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


SOURCE_ARCHIVE = Path("/content/image-restoration-project.tar.gz")
PROJECT_ROOT = Path("/content/image-restoration-project")
DATA_ROOT = Path("/content/kla-data")
TRAIN_ZIP_ID = "1SNPXs_E9GHQuHiiElXOsmnzOxT4PFubx"


def main() -> None:
    if not SOURCE_ARCHIVE.exists():
        raise FileNotFoundError(f"Upload the source archive first: {SOURCE_ARCHIVE}")
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(SOURCE_ARCHIVE, "r:gz") as archive:
        archive.extractall(PROJECT_ROOT, filter="data")

    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "pytorch-msssim",
            "lpips",
            "gdown",
            "einops",
            "PyYAML",
        ]
    )

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train_zip = DATA_ROOT / "train.zip"
    if not train_zip.exists():
        import gdown

        result = gdown.download(id=TRAIN_ZIP_ID, output=str(train_zip), quiet=False)
        if not result:
            raise RuntimeError("gdown did not return a downloaded train archive")

    extracted = DATA_ROOT / "extracted"
    gt_dir = extracted / "train" / "GT"
    noisy_dir = extracted / "train" / "NoisyLR"
    if len(list(gt_dir.glob("*.npy"))) != 3200 or len(list(noisy_dir.glob("*.npy"))) != 3200:
        if extracted.exists():
            shutil.rmtree(extracted)
        with zipfile.ZipFile(train_zip) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"CRC failure in {bad_member}")
            for member in archive.infolist():
                if member.is_dir() or member.filename.startswith("__MACOSX/") or member.filename.endswith(".DS_Store"):
                    continue
                archive.extract(member, extracted)

    print(
        f"REMOTE_READY project={PROJECT_ROOT} gt={len(list(gt_dir.glob('*.npy')))} "
        f"noisy={len(list(noisy_dir.glob('*.npy')))}"
    )


if __name__ == "__main__":
    main()
