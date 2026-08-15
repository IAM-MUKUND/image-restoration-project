# KLA SEMICON India Hackathon 2026
## Image Restoration Model Selection Guide

### Problem Context

The KLA challenge requires learning a mapping from **NoisyLR** $\rightarrow$ **Ground Truth (GT)**.

The degradation combines:
- Speckle Noise
- Gaussian Noise
- Downsampling ($\approx 1.5\times - 4\times$)

Evaluation depends on:
- PSNR
- SSIM
- LPIPS
- End-to-end inference speed
- Reproducibility

Therefore, the ideal model should balance restoration quality and throughput.

---

## Model Comparison

| Model | Type | Best For | Difficulty | Hackathon Fit |
| :--- | :--- | :--- | :--- | :--- |
| **NAFNet** | CNN | Denoising | Easy | ⭐⭐⭐⭐⭐ |
| **Restormer** | Transformer | General Restoration | Medium | ⭐⭐⭐⭐ |
| **SwinIR** | Swin Transformer | Super Resolution | Medium | ⭐⭐⭐⭐⭐ |
| **HINet** | CNN | Image Restoration | Medium | ⭐⭐⭐⭐ |
| **MPRNet** | Multi-stage CNN | Progressive Restoration | Hard | ⭐⭐⭐⭐ |

---

## Detailed Model Breakdown

### 1. NAFNet

- **Paper:** *Simple Baselines for Image Restoration* ([arXiv:2204.04676](https://arxiv.org/abs/2204.04676))
- **Official Code:** [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet)

#### Core Idea
NAFNet removed many components previously considered essential:
- No attention
- No GELU
- No LayerNorm

Instead, it relies on extremely simple residual blocks. Surprisingly, this simplicity produces state-of-the-art performance on many restoration benchmarks.

#### Strengths
- Extremely fast.
- Low GPU memory.
- Easy to reproduce.
- Excellent baseline.
- Great for denoising.

#### Weaknesses
- Not specifically designed for super-resolution.
- Slightly weaker than transformer methods on recovering very fine textures.

#### Why it fits KLA
Since throughput matters, NAFNet offers an excellent quality-speed tradeoff.

---

### 2. Restormer

- **Paper:** *Restormer: Efficient Transformer for High-Resolution Image Restoration* ([arXiv:2111.09881](https://arxiv.org/abs/2111.09881))
- **Official Code:** [swz30/Restormer](https://github.com/swz30/Restormer)

#### Core Idea
Restormer redesigned self-attention for high-resolution images. Instead of expensive global attention, it introduces **Multi-DConv Head Transposed Attention (MDTA)**. This drastically reduces computational cost.

#### Strengths
- Excellent restoration quality.
- Handles multiple degradations simultaneously.
- Strong global context.
- Preserves long-range structures.

#### Weaknesses
- Slower than CNNs.
- Larger memory footprint.
- More complex training.

#### Why it fits KLA
The challenge combines several degradations, and Restormer was designed for exactly this scenario.

---

### 3. SwinIR

- **Paper:** *SwinIR: Image Restoration Using Swin Transformer* ([arXiv:2108.10257](https://arxiv.org/abs/2108.10257))
- **Official Code:** [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)

#### Core Idea
SwinIR combines:
- Hierarchical features
- Shifted-window attention
- Efficient transformer blocks

Unlike standard Vision Transformers, attention is computed within local windows that shift between layers.

#### Strengths
- Excellent $2\times$ and $4\times$ super-resolution.
- Strong denoising.
- Better texture reconstruction.
- Efficient compared to vanilla ViT.

#### Weaknesses
- More parameters than NAFNet.
- Training is slower.
- Window size becomes an important hyperparameter.

#### Why it fits KLA
Your dataset contains $128 \times 128 \rightarrow 256 \times 256$ paired images. This makes SwinIR one of the strongest candidates because super-resolution is built into its design.

---

### 4. HINet

- **Paper:** *Half Instance Normalization Network* ([arXiv:2105.06086](https://arxiv.org/abs/2105.06086))
- **Official Code:** [megvii-model/HINet](https://github.com/megvii-model/HINet)

#### Core Idea
Instead of normalizing every feature channel, HINet applies Instance Normalization to only half of them. This preserves more image details.

#### Strengths
- Better detail preservation.
- Strong denoising.
- Lightweight compared to transformers.
- Stable training.

#### Weaknesses
- Less effective for large-scale texture modeling.
- Doesn't explicitly optimize super-resolution.

#### Why it fits KLA
A strong CNN alternative when transformer inference becomes too expensive.

---

### 5. MPRNet

- **Paper:** *Multi-Stage Progressive Image Restoration* ([arXiv:2102.02808](https://arxiv.org/abs/2102.02808))
- **Official Code:** [swz30/MPRNet](https://github.com/swz30/MPRNet)

#### Core Idea
MPRNet restores images progressively. Instead of fixing everything at once, it performs multiple refinement stages. Each stage improves the previous output.

#### Strengths
- Outstanding restoration quality.
- Strong edge recovery.
- Excellent on mixed degradations.

#### Weaknesses
- Heavy architecture.
- High memory usage.
- Slower inference.
- More difficult implementation.

#### Why it fits KLA
Excellent candidate if maximizing PSNR and SSIM becomes more important than throughput.

---

## Strategy Options

### Option A — Safe Competition Strategy

| Stage | Model |
| :--- | :--- |
| **Baseline** | Bicubic |
| **Experiment 1** | NAFNet |
| **Experiment 2** | SwinIR |
| **Final** | SwinIR |

**Advantages:**
- Fast experimentation.
- Excellent quality.
- Good inference speed.

### Option B — Research Strategy

| Stage | Model |
| :--- | :--- |
| **Baseline** | NAFNet |
| **Advanced** | Restormer |
| **Final** | Restormer |

**Advantages:**
- Better handling of mixed degradations.
- Stronger research story.

---

## Loss Functions Worth Exploring

The KLA webinar specifically encourages experimenting with losses.

- **L1 Loss:** $L = |y - \hat{y}|$
  - Sharp results
  - Stable training
- **SSIM Loss:**
  - Encourages structural similarity rather than pixel similarity.
  - Useful because evaluation already includes SSIM.
- **Perceptual Loss:**
  - Uses pretrained CNN features.
  - Good for recovering realistic textures.
- **Frequency Loss:**
  - Compares images in the Fourier domain.
  - Useful because downsampling destroys high-frequency information.

### Recommended Combinations

- **Initial Loss:**
  $$L = L_1 + 0.2 \times L_{\text{SSIM}}$$
- **Later Experiments:**
  $$L = L_1 + L_{\text{SSIM}} + L_{\text{Freq}}$$

---

## Throughput Optimization

Since KLA measures end-to-end inference time, optimization matters.

**Recommended:**
- Batch inference
- `torch.cuda.amp`
- `pin_memory=True`
- `num_workers > 0`
- `torch.compile()` (PyTorch 2.x)
- Efficient disk I/O
- Avoid unnecessary CPU-GPU transfers.

---

## Suggested Experiment Plan

| Experiment | Model | Goal |
| :--- | :--- | :--- |
| **E1** | Bicubic | Baseline |
| **E2** | NAFNet | Fast benchmark |
| **E3** | SwinIR | Super-resolution benchmark |
| **E4** | Restormer | Mixed degradation benchmark |
| **E5** | Best model + improved losses | Final submission |

> **Note:** Document every experiment. KLA explicitly values reproducibility.

---

## Repository Structure

```
kla-hackathon/
├── data/
│   ├── train/
│   ├── noisyLR/
│   └── test/
├── configs/
├── models/
├── datasets/
├── losses/
├── train.py
├── inference.py
├── evaluate.py
├── requirements.txt
└── README.md
```

This structure matches common research repositories and makes the project easy for judges to reproduce.
