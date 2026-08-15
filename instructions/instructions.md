# Semiconductor Image Restoration Project Plan

## Phase 0 — Research-Grade Foundation

This is the part most students skip. Our repository should look like an actual research repository.

```
kla-restoration/
├── configs/
│   ├── nafnet.yaml
│   ├── swinir.yaml
│   └── restormer.yaml
├── datasets/
│   ├── paired_dataset.py
│   ├── augmentations.py
│   └── transforms.py
├── models/
│   ├── nafnet/
│   ├── swinir/
│   └── restormer/
├── losses/
│   ├── l1.py
│   ├── ssim.py
│   ├── frequency.py
│   └── combined.py
├── metrics/
│   ├── psnr.py
│   ├── ssim.py
│   └── lpips.py
├── engine/
│   ├── trainer.py
│   ├── evaluator.py
│   └── inference.py
├── checkpoints/
├── logs/
├── outputs/
├── train.py
├── evaluate.py
├── infer.py
├── requirements.txt
└── README.md
```

### Why this matters

When KLA says "reproducibility," this is exactly what they're looking for.

---

## Phase 1 — Dataset Pipeline

This is actually the most important phase. Our dataset isn't PNGs; it's paired `.npy` arrays.

```
GT/
├── 000000.npy
└── 000001.npy

NoisyLR/
├── 000000.npy
└── 000001.npy
```

### Dataset class

We'll build a custom PyTorch Dataset.

**Responsibilities:**

- Load paired images
- Preserve `float32` precision
- Resize handling
- Augmentations
- Tensor conversion

### Train / Validation split

Instead of a random split every run:

- **Total Images:** 3,200
- **Train:** 2,880
- **Validation:** 320
- **Seed:** Fixed seed for reproducibility.

---

## Phase 1.5 — Smart Augmentations

This is where KLA's webinar becomes useful. They explicitly allow generating synthetic degradations.

We'll **not** modify GT. We'll augment the input pair consistently.

| Augmentation                   | Why                    |
| :----------------------------- | :--------------------- |
| **Horizontal Flip**      | Orientation robustness |
| **Vertical Flip**        | Symmetry               |
| **90° Rotation**        | Pattern diversity      |
| **Random Crop**          | Better local learning  |
| **Extra Gaussian Noise** | Robustness             |
| **Extra Speckle**        | Matches degradation    |
| **Downsample-Upsample**  | OOD robustness         |

### Examples

![Data Augmentation for Image Classification | MetricGate](https://images.openai.com/static-rsc-4/hPMRHHgGz5IyV1N3Bj7s4C8xEV605J6QE3AA_cv2EU9NAZvDD9Rwjbf1TbORrnFcCZBWHTNjVhK3QycB4hYODCBNoEQTwqKMO_peTV_qSxLotlRa3UMvd0wGnQJFj8wOGkloWNHls-6BlowUg-5JWMu59mC9wjU80jDtnPabg9A?purpose=inline)

![What Is Image Data Augmentation? — Picsellia](https://images.openai.com/static-rsc-4/vvC-lQu1LpYOZCcJbfmN7lGdiOM0cdPadb4nTIE5rLgQDmtWZLytXHuV5qhRsXU4HWelwjEmSPK1aoGYNwcTK-SKBEO7ghnXCL0ekNGLDOm-bv8T5dmHiulgKGl8ZkfFJDgRz3RMnF0KK-Ln_jjep3d9EnNdlU3-xQ2oRemPLmU?purpose=inline)

![Image Augmentation for Machine Learning: Techniques, Examples & Code | Datature Blog](https://images.openai.com/static-rsc-4/ywy3QT3X1A1j-_39C8IcdfVa5sZ1E5QLjSU44cLvDPyKI7RlJPOyDbsU_JRpIQLOtPah8aSmqvXQA1UYX_aT9J_rzfiiG31WiR7x5IGDyAN3aFGIE6iGhSb0sz6zASa4pJC73ZsPUukeBiwvytZOgdQIOONPJ-23dmTJFpu-6F4?purpose=inline)

> Data augmentation is likely more valuable than changing architectures early.

---

## Phase 2 — NAFNet Baseline

We're intentionally starting with NAFNet. Not because it's the strongest, but because it validates the entire pipeline.

**Loss:**

$$
\mathcal{L} = \mathcal{L}_1
$$

**Optimizer:**

- AdamW
- Cosine LR Scheduler

### Sanity check

Before full training, overfit on **2 images**.

**Expected:**

- Loss should approach nearly zero.
- Output should visually match GT.

> **Warning:** If this fails, **do not continue.** This catches 90% of bugs.

---

## Phase 3 — Evaluation Framework

Now we become a research lab. Every experiment automatically produces:

| Metric                   |
| :----------------------- |
| **PSNR**           |
| **SSIM**           |
| **LPIPS**          |
| **Inference Time** |

And saves output in:

```
outputs/
└── experiment_01/
    ├── before_after/
    └── metrics.csv
```

### Example comparison board

![CoLoRA: Contribution-based Low-Rank Adaptation with Pre-training Model for Real Image Restoration](https://images.openai.com/static-rsc-4/Y0c8EVTqjRr9rvnWqmp_obWOPUGz0pjZdJBAbHdxVpazE4LhqsjVwvoLkGAm9p9ChjsHHEn3eLKApFgwYD71guhBDU7jZReiMqYGLvQDOY-uA2wsS9h754fIDTSUMo5tUQGEMDyAasyK0J643lK199SncGsnO8vMIxjTl1uLl5o?purpose=inline)

Visual logging like this becomes incredibly valuable later.

---

## Phase 4 — SwinIR Integration

Now we introduce the first serious model.

Why SwinIR first? Because our dataset naturally performs $128 \rightarrow 256$, and SwinIR's architecture was designed around this.

**Implementation:**

- Clone official repo / integrate module
- Adapt Dataloader
- Load pretrained weights where applicable
- Replace output head if needed

**Training:**

- Automatic Mixed Precision (AMP)
- Checkpointing
- Gradient clipping

---

## Phase 5 — Loss Engineering

This is where leaderboard gains often happen. Instead of changing architectures immediately, we improve supervision.

- **Stage 1:** $\mathcal{L} = \mathcal{L}_1$
- **Stage 2:** $\mathcal{L} = \mathcal{L}_1 + 0.2 \cdot \mathcal{L}_{\text{SSIM}}$
- **Stage 3:** $\mathcal{L} = \mathcal{L}_1 + \mathcal{L}_{\text{SSIM}} + \mathcal{L}_{\text{Frequency}}$

### Why Frequency Loss?

Downsampling destroys high-frequency components.

![Spatial and Frequency Domain — Image Processing | by Anshul Sachdev | VITHelper | Medium](https://images.openai.com/static-rsc-4/58FGt0xuJxv2B0e_37HDDXf0vV6CHm4MWtmN_VV3zIlhAI5SLkY_v5yzcfP_XKTeg28ZnwN8dHkOmBnHMiUCtn7bijjvLAsevzgycsJIjen-XbjmSYJzQb6-CK1TmNaNaznYMZhu7mehZzxl_O7H3EAjX-gPyPAmA5NIXJgqnFI?purpose=inline)

Frequency loss explicitly encourages recovering those details. We'll test each combination.

---

## Phase 6 — Restormer Experiments

Only after SwinIR works. We'll answer one research question:

> *Does global attention improve mixed degradation restoration?*

- Same training pipeline
- Same metrics

This keeps comparisons fair.

---

## Phase 7 — Throughput Optimization

This phase exists purely because KLA scores inference speed. We'll optimize the whole pipeline:

### Data Loading

- `pin_memory=True`
- `persistent_workers=True`
- Multiple workers

### GPU

- Mixed Precision
- `torch.compile()`
- Batched inference
- Asynchronous transfers

### Disk I/O

Instead of sequential `load` $\rightarrow$ `predict` $\rightarrow$ `save`, we'll pipeline operations.

---

## Phase 8 — Final Inference Script

This must satisfy KLA's evaluation.

**Input Command:**

```bash
python infer.py \
  --input test/ \
  --output predictions/
```

---

## Phase 9 — Documentation Like a Research Team

Instead of writing documentation at the end, we'll collect artifacts throughout. Every experiment gets:

- Config
- Metrics
- Qualitative comparison
- Runtime
- Observations

By the final presentation, we will already have our complete ablation study.
