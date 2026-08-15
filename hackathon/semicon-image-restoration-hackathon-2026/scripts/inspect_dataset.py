#!/usr/bin/env python3
"""Inspect the downloaded KLA image-restoration dataset and emit a manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SOURCE_URL = (
    "https://drive.google.com/drive/folders/"
    "1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_archive(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        files = [info for info in archive.infolist() if not info.is_dir()]
        useful = [
            info
            for info in files
            if not info.filename.startswith("__MACOSX/")
            and not info.filename.endswith(".DS_Store")
        ]
        return {
            "filename": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
            "zip_crc_check": "passed" if bad_member is None else f"failed: {bad_member}",
            "all_file_entries": len(files),
            "useful_file_entries": len(useful),
            "uncompressed_useful_bytes": sum(info.file_size for info in useful),
        }


def inspect_arrays(paths: list[Path]) -> dict[str, object]:
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    total_values = 0
    out_of_unit_range = 0
    global_min = float("inf")
    global_max = float("-inf")
    value_sum = 0.0
    value_square_sum = 0.0

    for path in paths:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        shape_counts["x".join(map(str, array.shape))] += 1
        dtype_counts[str(array.dtype)] += 1
        total_values += array.size
        global_min = min(global_min, float(np.min(array)))
        global_max = max(global_max, float(np.max(array)))
        out_of_unit_range += int(np.count_nonzero((array < 0) | (array > 1)))
        value_sum += float(np.sum(array, dtype=np.float64))
        value_square_sum += float(np.sum(np.square(array, dtype=np.float64)))

    mean = value_sum / total_values if total_values else None
    variance = value_square_sum / total_values - mean * mean if total_values else None
    return {
        "file_count": len(paths),
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "global_min": global_min if paths else None,
        "global_max": global_max if paths else None,
        "mean": mean,
        "std": variance**0.5 if variance is not None and variance >= 0 else None,
        "values_outside_0_1": out_of_unit_range,
        "values_outside_0_1_fraction": (
            out_of_unit_range / total_values if total_values else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    extracted = root / "extracted"
    train_gt = sorted((extracted / "train" / "GT").glob("*.npy"))
    train_noisy = sorted((extracted / "train" / "NoisyLR").glob("*.npy"))
    test_noisy = sorted((extracted / "NoisyLR").glob("*.npy"))
    gt_names = {path.name for path in train_gt}
    noisy_names = {path.name for path in train_noisy}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_dataset_source": SOURCE_URL,
        "dataset_root": str(root),
        "archives": [
            inspect_archive(path) for path in sorted(root.glob("*.zip"))
        ],
        "extracted": {
            "train_ground_truth": inspect_arrays(train_gt),
            "train_degraded": inspect_arrays(train_noisy),
            "test_degraded": inspect_arrays(test_noisy),
            "train_pairing": {
                "matched_filenames": len(gt_names & noisy_names),
                "missing_ground_truth_for": sorted(noisy_names - gt_names),
                "missing_degraded_for": sorted(gt_names - noisy_names),
            },
            "test_ground_truth_in_download": False,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
