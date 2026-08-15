#!/usr/bin/env python3
"""Render a same-sample benchmark comparison without changing source arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


DISPLAY_NAMES = {
    "bicubic": "Bicubic",
    "nafnet": "NAFNet",
    "swinir": "SwinIR",
    "restormer": "Restormer",
    "esrgan_rrdb": "ESRGAN-RRDB",
    "mprnet": "MPRNet",
}


def to_image(array: np.ndarray, size: int = 256) -> Image.Image:
    array = np.asarray(array, dtype=np.float32)
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.rint(array * 255.0).astype(np.uint8), mode="L")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.BICUBIC)
    return image.convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", help="Sample stem; defaults to the first shared saved prediction")
    args = parser.parse_args()

    model_dirs = [
        path
        for path in args.benchmark_dir.iterdir()
        if path.is_dir() and (path / "predictions").is_dir() and path.name in DISPLAY_NAMES
    ]
    model_dirs.sort(key=lambda path: list(DISPLAY_NAMES).index(path.name))
    if not model_dirs:
        raise FileNotFoundError(f"No prediction directories found under {args.benchmark_dir}")

    shared = None
    for model_dir in model_dirs:
        names = {path.stem for path in (model_dir / "predictions").glob("*.npy")}
        shared = names if shared is None else shared & names
    if not shared:
        raise RuntimeError("The model prediction directories have no shared sample")
    stem = args.sample or sorted(shared)[0]
    if stem not in shared:
        raise KeyError(f"Sample {stem!r} is not available from every model")

    panels: list[tuple[str, Image.Image]] = [
        ("Noisy LR (display upscaled)", to_image(np.load(args.data_root / "NoisyLR" / f"{stem}.npy"))),
        ("Ground truth", to_image(np.load(args.data_root / "GT" / f"{stem}.npy"))),
    ]
    panels.extend(
        (
            DISPLAY_NAMES[model_dir.name],
            to_image(np.load(model_dir / "predictions" / f"{stem}.npy")),
        )
        for model_dir in model_dirs
    )

    tile = 256
    label_height = 34
    gap = 10
    columns = 4
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile + (columns + 1) * gap, rows * (tile + label_height) + (rows + 1) * gap),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=15)
    for index, (label, image) in enumerate(panels):
        row, column = divmod(index, columns)
        x = gap + column * (tile + gap)
        y = gap + row * (tile + label_height + gap)
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 4, y + 8), label, fill="black", font=font)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"sample={stem} output={args.output}")


if __name__ == "__main__":
    main()
