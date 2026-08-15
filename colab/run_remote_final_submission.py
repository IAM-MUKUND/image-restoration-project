"""Generate and validate the 400-image DAF accuracy-mode submission."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT = Path("/content/image-restoration-project")
INPUT_ARCHIVE = Path("/content/kla-test-inputs.tar.gz")
INPUT_ROOT = Path("/content/kla-test")
INPUT_DIR = INPUT_ROOT / "NoisyLR"
CHECKPOINT = PROJECT / "artifacts/daf-perceptual-stage2/epoch-1.pt"
OUTPUT = Path("/content/daf-final-submission")
PREDICTIONS = OUTPUT / "predictions"
ARCHIVE = Path("/content/kla-daf-final-submission.tar.gz")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if INPUT_ROOT.exists():
        shutil.rmtree(INPUT_ROOT)
    INPUT_ROOT.mkdir(parents=True)
    with tarfile.open(INPUT_ARCHIVE, "r:gz") as archive:
        archive.extractall(INPUT_ROOT, filter="data")
    if len(list(INPUT_DIR.glob("*.npy"))) != 400:
        raise RuntimeError("Official test input archive did not contain exactly 400 arrays")

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    PREDICTIONS.mkdir(parents=True)

    import sys

    sys.path.insert(0, str(PROJECT))
    from engine.inference import restore_directory

    inference = restore_directory(
        INPUT_DIR,
        PREDICTIONS,
        CHECKPOINT,
        batch_size=8,
        self_ensemble_transforms=8,
    )

    input_names = {path.name for path in INPUT_DIR.glob("*.npy")}
    output_files = sorted(PREDICTIONS.glob("*.npy"))
    output_names = {path.name for path in output_files}
    failures = []
    minima, maxima, means = [], [], []
    for path in output_files:
        array = np.load(path, allow_pickle=False)
        if array.shape != (256, 256):
            failures.append(f"{path.name}: shape={array.shape}")
        if array.dtype != np.float32:
            failures.append(f"{path.name}: dtype={array.dtype}")
        if not np.isfinite(array).all():
            failures.append(f"{path.name}: non-finite values")
        minima.append(float(array.min()))
        maxima.append(float(array.max()))
        means.append(float(array.mean()))
    validation = {
        "status": "passed" if not failures and input_names == output_names else "failed",
        "input_count": len(input_names),
        "output_count": len(output_names),
        "filenames_match": input_names == output_names,
        "expected_shape": [256, 256],
        "expected_dtype": "float32",
        "finite": not any("non-finite" in failure for failure in failures),
        "global_min": min(minima),
        "global_max": max(maxima),
        "mean_of_image_means": float(np.mean(means)),
        "within_submission_range": min(minima) >= 0.0 and max(maxima) <= 1.0,
        "failures": failures,
        "checkpoint_sha256": sha256(CHECKPOINT),
        "input_archive_sha256": sha256(INPUT_ARCHIVE),
    }
    if validation["status"] != "passed" or not validation["within_submission_range"]:
        raise RuntimeError(json.dumps(validation, indent=2))
    (OUTPUT / "INFERENCE-VALIDATION.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    selection = {
        "mode": "accuracy",
        "model": "daf_restormer",
        "self_ensemble_transforms": 8,
        "validation_samples": 320,
        "selected": {"psnr_mean_db": 28.471262151002882, "ssim_mean": 0.7711622648406774, "lpips_mean": 0.22848921299446373},
        "restormer_reference": {"psnr_mean_db": 28.393100327253343, "ssim_mean": 0.7709464966785162, "lpips_mean": 0.27190778822405265},
        "t4_latency_median_ms_batch1": 264.72344970703125,
    }
    (OUTPUT / "SELECTION.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")

    checkpoint_dir = OUTPUT / "checkpoints"
    checkpoint_dir.mkdir()
    shutil.copy2(CHECKPOINT, checkpoint_dir / "daf-restormer-perceptual-epoch1.pt")

    preview_names = sorted(input_names)[:4]
    figure, axes = plt.subplots(2, len(preview_names), figsize=(12, 6))
    for column, name in enumerate(preview_names):
        noisy = np.load(INPUT_DIR / name, allow_pickle=False)
        restored = np.load(PREDICTIONS / name, allow_pickle=False)
        axes[0, column].imshow(noisy, cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(f"Noisy LR\n{name}")
        axes[1, column].imshow(restored, cmap="gray", vmin=0, vmax=1)
        axes[1, column].set_title("DAF restored")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT / "preview.png", dpi=160)
    plt.close(figure)

    with tarfile.open(ARCHIVE, "w:gz") as archive:
        archive.add(OUTPUT, arcname="daf-final-submission")
    print(
        f"DAF_FINAL_SUBMISSION_COMPLETE archive={ARCHIVE} bytes={ARCHIVE.stat().st_size} "
        f"sha256={sha256(ARCHIVE)} validation={validation['status']} images={len(output_files)}"
    )


if __name__ == "__main__":
    main()
