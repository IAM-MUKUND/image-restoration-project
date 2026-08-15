from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from models import build_model


@torch.inference_mode()
def restore_directory(
    input_dir: Path,
    output_dir: Path,
    weights: Path,
    batch_size: int = 16,
) -> dict[str, object]:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model_name = checkpoint["model_name"]
    model = build_model(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    use_amp = device.type == "cuda"
    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise ValueError(f"No .npy inputs found in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    for offset in range(0, len(files), batch_size):
        batch_files = files[offset : offset + batch_size]
        arrays = [np.load(path, allow_pickle=False).astype(np.float32, copy=False) for path in batch_files]
        tensor = torch.from_numpy(np.stack(arrays)[:, None]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(tensor)
        prediction = prediction.float().clamp(0.0, 1.0).cpu().numpy()[:, 0]
        for path, array in zip(batch_files, prediction):
            np.save(output_dir / path.name, array.astype(np.float32))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    summary = {
        "model": model_name,
        "weights": str(weights.resolve()),
        "input_dir": str(input_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "images": len(files),
        "seconds": elapsed,
        "images_per_second": len(files) / elapsed,
        "device": str(device),
    }
    (output_dir / "inference-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
