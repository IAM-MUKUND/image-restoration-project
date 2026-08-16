import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

pred_dir = Path("artifacts/remote-runs/restormer-curriculum-stage2/predictions")
input_dir = Path("main_dataset/Test_NoisyLR/NoisyLR")
output_dir = Path("artifacts/remote-runs/restormer-curriculum-stage2")

input_files = sorted(list(input_dir.glob("*.npy")))
output_files = sorted(list(pred_dir.glob("*.npy")))

failures = []
minima, maxima, means = [], [], []

for p in output_files:
    arr = np.load(p, allow_pickle=False)
    if arr.shape != (256, 256):
        failures.append(f"{p.name}: shape={arr.shape}")
    if arr.dtype != np.float32:
        failures.append(f"{p.name}: dtype={arr.dtype}")
    if not np.isfinite(arr).all():
        failures.append(f"{p.name}: non-finite values")
    minima.append(float(arr.min()))
    maxima.append(float(arr.max()))
    means.append(float(arr.mean()))

validation = {
    "status": "passed" if not failures and len(input_files) == len(output_files) else "failed",
    "input_count": len(input_files),
    "output_count": len(output_files),
    "expected_shape": [256, 256],
    "expected_dtype": "float32",
    "finite": not any("non-finite" in f for f in failures),
    "global_min": min(minima),
    "global_max": max(maxima),
    "mean_of_image_means": float(np.mean(means)),
    "within_submission_range": min(minima) >= 0.0 and max(maxima) <= 1.0,
    "failures": failures
}

val_path = output_dir / "INFERENCE-VALIDATION.json"
val_path.write_text(json.dumps(validation, indent=2))
print("INFERENCE-VALIDATION:\n", json.dumps(validation, indent=2))

# Render preview grid
preview_names = [p.name for p in input_files[:4]]
fig, axes = plt.subplots(2, len(preview_names), figsize=(12, 6))

for col, name in enumerate(preview_names):
    noisy = np.load(input_dir / name, allow_pickle=False)
    restored = np.load(pred_dir / name, allow_pickle=False)
    axes[0, col].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(f"Noisy LR (128x128)\n{name}", fontsize=10)
    axes[0, col].axis("off")
    axes[1, col].imshow(restored, cmap="gray", vmin=0, vmax=1)
    axes[1, col].set_title("Restormer 1.6M Restored", fontsize=10, color="#1f77b4")
    axes[1, col].axis("off")

fig.tight_layout()
fig.savefig(output_dir / "preview.png", dpi=160)
plt.close(fig)
print(f"Saved preview grid to {output_dir / 'preview.png'}")
