# KLA PS01 full-data T4 fixed-budget benchmark

> These numbers compare task-adapted, compute-matched variants under a fixed training budget. They are not fully converged final competition models.

## Protocol

- Hardware: `Tesla T4`
- PyTorch/CUDA: `2.11.0+cu128` / `12.8`
- Training epochs: 3
- Training/validation limits: all / all
- Batch size: 8
- Seed: 42
- Loss: Charbonnier + 0.1 x SSIM

## Results

| Model | Status | Params | PSNR dB | SSIM | LPIPS |
|---|---|---:|---:|---:|---:|
| bicubic | completed | 0 | 23.279 | 0.5443 | 0.4312 |
| nafnet | completed | 253,057 | 25.932 | 0.6467 | 0.3952 |
| swinir | completed | 405,505 | 26.377 | 0.6588 | 0.3641 |
| restormer | completed | 1,608,757 | 26.614 | 0.6778 | 0.3334 |
| esrgan_rrdb | completed | 1,136,097 | 26.514 | 0.6721 | 0.3314 |
| mprnet | completed | 265,451 | 25.955 | 0.6478 | 0.3692 |

## Compute and deployability

| Model | Train seconds | Checkpoint MiB | Median / p90 latency ms | Validation img/s | Peak VRAM MiB |
|---|---:|---:|---:|---:|---:|
| bicubic | 0.0 | 0.00 | 0.128 / 0.134 | 276.2 | 0.0 |
| nafnet | 130.8 | 1.00 | 10.040 / 12.911 | 174.6 | 1319.0 |
| swinir | 446.8 | 1.60 | 21.564 / 21.777 | 44.3 | 4664.6 |
| restormer | 210.8 | 6.18 | 17.293 / 20.910 | 123.0 | 2323.8 |
| esrgan_rrdb | 274.1 | 4.39 | 13.984 / 15.070 | 79.6 | 1952.9 |
| mprnet | 269.2 | 1.06 | 11.056 / 11.092 | 85.4 | 1968.9 |

## Direct takeaways

- Highest PSNR: **restormer** at 26.614 dB.
- Lowest LPIPS: **esrgan_rrdb** at 0.3314.
- Fastest trainable model: **nafnet**.
- Bicubic remains the non-learning reference, not a trainable competitor.
- Choose the final model only after a longer convergence run and visual residual review.
