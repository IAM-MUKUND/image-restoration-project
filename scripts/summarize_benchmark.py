#!/usr/bin/env python3
"""Turn benchmark_summary.json into a concise Markdown comparison report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", default="Fixed-budget benchmark")
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding="utf-8"))
    settings = data["settings"]
    environment = data["environment"]
    rows = []
    for name, result in data["results"].items():
        if result.get("status") != "completed":
            rows.append((name, result, None))
            continue
        rows.append((name, result, result["metrics"]))

    completed = [(name, result, metrics) for name, result, metrics in rows if metrics]
    quality_rank = sorted(completed, key=lambda row: row[2]["psnr_mean_db"], reverse=True)
    speed_rank = sorted(completed, key=lambda row: row[1]["latency_median_ms_batch1"])
    lines = [
        f"# {args.label}",
        "",
        "> These numbers compare task-adapted, compute-matched variants under a fixed training budget. "
        "They are not fully converged final competition models.",
        "",
        "## Protocol",
        "",
        f"- Hardware: `{environment.get('device_name')}`",
        f"- PyTorch/CUDA: `{environment.get('torch')}` / `{environment.get('cuda_runtime')}`",
        f"- Training epochs: {settings.get('epochs')}",
        f"- Training/validation limits: {settings.get('train_limit') or 'all'} / {settings.get('val_limit') or 'all'}",
        f"- Batch size: {settings.get('batch_size')}",
        f"- Seed: {settings.get('seed')}",
        f"- Loss: Charbonnier + {settings.get('ssim_loss_weight')} x SSIM",
        "",
        "## Results",
        "",
        "| Model | Status | Params | PSNR dB | SSIM | LPIPS |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for name, result, metrics in rows:
        if metrics is None:
            lines.append(f"| {name} | failed: {result.get('error')} | - | - | - | - |")
            continue
        lines.append(
            f"| {name} | completed | {result['parameters']:,} | "
            f"{fmt(metrics['psnr_mean_db'])} | {fmt(metrics['ssim_mean'], 4)} | "
            f"{fmt(metrics['lpips_mean'], 4)} |"
        )

    if completed:
        lines.extend(
            [
                "",
                "## Compute and deployability",
                "",
                "| Model | Train seconds | Checkpoint MiB | Median / p90 latency ms | Validation img/s | Peak VRAM MiB |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for name, result, metrics in completed:
            lines.append(
                f"| {name} | {fmt(result['training_seconds'], 1)} | "
                f"{fmt(result['checkpoint_bytes'] / (1024 ** 2), 2)} | "
                f"{fmt(result['latency_median_ms_batch1'])} / {fmt(result['latency_p90_ms_batch1'])} | "
                f"{fmt(metrics['validation_images_per_second'], 1)} | {fmt(result['peak_vram_mib'], 1)} |"
            )
        perceptual_rank = sorted(completed, key=lambda row: row[2]["lpips_mean"])
        lines.extend(
            [
                "",
                "## Direct takeaways",
                "",
                f"- Highest PSNR: **{quality_rank[0][0]}** at {quality_rank[0][2]['psnr_mean_db']:.3f} dB.",
                f"- Lowest LPIPS: **{perceptual_rank[0][0]}** at {perceptual_rank[0][2]['lpips_mean']:.4f}.",
                f"- Fastest trainable model: **{next(row for row in speed_rank if row[0] != 'bicubic')[0]}**.",
                f"- Bicubic remains the non-learning reference, not a trainable competitor.",
                "- Choose the final model only after a longer convergence run and visual residual review.",
            ]
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
