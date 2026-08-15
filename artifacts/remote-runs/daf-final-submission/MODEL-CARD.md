# DAF-Restormer-P accuracy model

## Intended use

Joint restoration and 2x super-resolution of the KLA/i4C hackathon's noisy
128x128 grayscale NumPy arrays to clean 256x256 float32 arrays.

## Architecture

DAF-Restormer adds a degradation encoder, global prompt conditioning, a local
noise map, prompt-modulated Restormer blocks, phase-preserving frequency
refinement, progressive clean-LR supervision, and uncertainty estimation to a
two-level restoration backbone. The selected checkpoint has 4,172,518
parameters.

## Selected training recipe

1. Train for 10 epochs on the fixed seed-42 2,880/320 split with Charbonnier +
   0.1 SSIM loss, AdamW, cosine learning-rate decay, and no synthetic
   degradation.
2. Fine-tune for two epochs at `1e-5` with 0.2 SSIM and 0.02 frozen
   LPIPS-AlexNet loss evaluated at 128x128.
3. Continue at `3e-6` with 0.25 SSIM and 0.05 full-resolution LPIPS loss;
   select the first epoch of this stage.
4. Average inverse-mapped predictions over all eight rotations/reflections at
   inference.

## Held-out result

| Model | PSNR dB | SSIM | LPIPS | T4 median ms/image |
|---|---:|---:|---:|---:|
| 10-epoch Restormer | 28.393100 | 0.770946 | 0.271908 | 13.660 |
| DAF-Restormer-P x8 | 28.471262 | 0.771162 | 0.228489 | 264.723 |

The comparison uses 320 held-out official pairs. Higher PSNR/SSIM and lower
LPIPS are better. Metric weights for the private KLA evaluation are not public,
so no invented aggregate score is reported.

## Artifacts and integrity

- Checkpoint: `checkpoints/daf-restormer-perceptual-epoch1.pt`
- Checkpoint SHA-256:
  `0ee70a5d4698444e5a47c6f85382f55c27d6dd6fa0af8e2853b25954e90c13a3`
- Accuracy selection: `SELECTION.json`
- 400-image contract validation: `INFERENCE-VALIDATION.json`
- Full held-out x1/x4/x8 metrics: `../daf-stage2-tta/epoch-1.json`

The local full submission archive is
`artifacts/colab-downloads/kla-daf-final-submission.tar.gz` with SHA-256
`04b6c7ea06d7a7727ca9a0a8f430a3c2fbdea19277903a4756ba382fe5475ffd`.
It is intentionally excluded from Git because it contains 400 generated NumPy
arrays; the model, manifests, metrics, and preview are tracked.

## Limitations

- Accuracy mode is about 19x slower than the compact Restormer control.
- The released arrays visibly contain generic grayscale natural scenes rather
  than clear semiconductor inspection imagery, so acquisition-domain shift
  remains the main deployment risk.
- The 320-image holdout guided recipe selection. A private evaluation set is
  still necessary to confirm generalization.
