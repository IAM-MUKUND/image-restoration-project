import json
from pathlib import Path
import numpy as np

pred_dir = Path("artifacts/remote-runs/daf-curriculum-stage2/predictions")
input_dir = Path("main_dataset/Test_NoisyLR/NoisyLR")

in_files = sorted(list(input_dir.glob("*.npy")))
out_files = sorted(list(pred_dir.glob("*.npy")))

failures = []
for f in out_files:
    arr = np.load(f)
    if arr.shape != (256, 256):
        failures.append(f"{f.name}: shape {arr.shape}")
    if arr.dtype != np.float32:
        failures.append(f"{f.name}: dtype {arr.dtype}")
    if not np.isfinite(arr).all():
        failures.append(f"{f.name}: non-finite")

val = {
    "status": "passed" if not failures and len(in_files) == len(out_files) else "failed",
    "input_count": len(in_files),
    "output_count": len(out_files),
    "expected_shape": [256, 256],
    "expected_dtype": "float32",
    "failures": failures
}

val_path = Path("artifacts/remote-runs/daf-curriculum-stage2/INFERENCE-VALIDATION.json")
val_path.write_text(json.dumps(val, indent=2))
print(json.dumps(val, indent=2))
