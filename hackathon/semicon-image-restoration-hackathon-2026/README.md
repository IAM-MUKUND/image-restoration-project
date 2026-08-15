# SEMICON India Hackathon 2026 - KLA PS01 working pack

Last verified from the official i4C page and downloads: **15 August 2026 (IST)**.

## Urgent status

- **Problem:** AI-Based Restoration of Degraded Images for Semiconductor Inspection
- **Track / industry partner:** Track 1 - KLA
- **Registration and initial submission deadline shown by i4C:** **16 August 2026**
- **Deadline time is not stated.** Register and upload early rather than assuming end-of-day.
- **Registration portal:** <https://hackathon2026.i4c.in/>
- **Main official page:** <https://i4c.in/hackathon-2026/>

## The task

Train one restoration model that maps a degraded grayscale image to a clean,
higher-resolution estimate of its paired ground truth. The model must handle a
combination of:

1. **Speckle noise** - multiplicative-looking grain that can push values outside
   the ground-truth intensity range.
2. **Additive Gaussian noise** - confirmed by the KLA technical deck. The landing
   page's description also uses blur-like wording, so do not silently substitute
   Gaussian blur for additive Gaussian noise.
3. **Spatial-resolution reduction / 2x super-resolution** - the downloaded data
   maps 128x128 inputs to 256x256 targets.

The order of degradations is not fixed. A sample may contain all of them. The
official material also says that test data will contain both similar and
dissimilar/out-of-distribution sources, and that inference speed matters.

### Important signal-range rule

- Ground truth is normalized to **[0, 1]**.
- Degraded input values are allowed outside **[0, 1]**; KLA explicitly calls this
  a feature, not a bug.
- Do **not** clip the degraded arrays before the model. If you normalize them,
  document the transform and keep it consistent at inference.

## Dataset downloaded and verified

Official source:
<https://drive.google.com/drive/folders/1VKiFW-kDk9-q5XRPu3nrl08OM94EwzV6?usp=drive_link>

The original ZIP files are preserved in `dataset/`, and clean extracted copies
are under `dataset/extracted/`. macOS metadata from `__MACOSX` was excluded from
the extracted copy. The complete machine-readable audit is in
[`dataset-manifest.json`](dataset-manifest.json).

```text
dataset/
|- train.zip
|- Test_NoisyLR.zip
`- extracted/
   |- train/
   |  |- GT/          3,200 arrays, 256x256, float32
   |  `- NoisyLR/     3,200 arrays, 128x128, float32
   `- NoisyLR/          400 test inputs, 128x128, float32
```

### Observed facts from every downloaded array

| Split | Files | Shape | Dtype | Global range | Values outside [0,1] |
|---|---:|---|---|---|---:|
| Train ground truth | 3,200 | 256x256 | float32 | 0.0000 to 1.0000 | 0% |
| Train degraded | 3,200 | 128x128 | float32 | -0.2786 to 2.1580 | 3.3933% |
| Public test degraded | 400 | 128x128 | float32 | -0.2249 to 2.1580 | 3.7403% |

All 3,200 training filenames pair exactly between `GT` and `NoisyLR`. The public
test download contains **no test ground truth**.

The sampled files and KLA's own slides show generic grayscale natural scenes,
not obvious semiconductor micrographs. Treat semantic generalization as a real
risk: optimize restoration quality without depending on object categories or
assuming that future evaluation images look like the training subjects.

Preview: [`dataset-preview.png`](dataset-preview.png)

### Archive integrity

Both archives passed a full ZIP CRC test and every extracted `.npy` file was
read successfully during the dataset scan.

| Archive | Size | SHA-256 |
|---|---:|---|
| `train.zip` | 918,994,209 bytes | `b93dc4486a1181338630a55a88596e722cfdf75a0c1bbe2ed8404f01980c0abb` |
| `Test_NoisyLR.zip` | 23,419,125 bytes | `f2904f75d6938c23f7ad5f7d41194744a5cdceb3c1d1ea066b59a9dbf9b45f83` |

The whole local pack, including original archives, extracted arrays, documents,
manifest, and preview, currently occupies about **1.91 GiB**.

### Minimal loader

```python
from pathlib import Path
import numpy as np

root = Path("dataset/extracted/train")
x = np.load(root / "NoisyLR" / "000000.npy", allow_pickle=False)  # (128, 128)
y = np.load(root / "GT" / "000000.npy", allow_pickle=False)       # (256, 256)

assert x.dtype == y.dtype == np.float32
assert x.shape == (128, 128) and y.shape == (256, 256)
# x may be below 0 or above 1; y is in [0, 1].
```

## Evaluation signals stated publicly

The result slide is expected to report:

- **SSIM**
- **PSNR** (written as pSNR on the page)
- **LPIPS**
- before/after/ground-truth comparisons
- model size, training time, and inference time per image

KLA's deck says evaluation is not based only on leaderboard statistics; it also
considers the work put into **data, AI model, loss design, and compute/training
hygiene**. The page says the final evaluation script will be run as-is on an
**NVIDIA H100 GPU** and timed.

No public source reviewed here gives metric weights, a target score, an inference
timeout, an exact output-file format for the `.npy` test inputs, or whether LPIPS
will be computed by replicating grayscale into three channels. Do not invent
these details; check the portal announcements/webinars or ask KLA/i4C.

## Initial proposal / PDF content

The problem-specific page asks for these 9 content slides:

1. Team details
2. Problem statement addressed and why it matters
3. Idea description and model choice
4. Proposed solution: architecture, training, loss, augmentation, pipeline
5. Innovation and uniqueness
6. Results: SSIM, PSNR, LPIPS, and visual comparisons
7. Technology and feasibility: stack, hardware, time, size, inference speed
8. GitHub link and optional/recommended video link
9. References

Save the final submission as PDF, remove the instruction slide, and use the
problem-specific filename format **`TeamName_KLA_PS01.pdf`**. The exact official
template is saved in `official-resources/`.

## Mandatory public GitHub repository

The problem-specific page requires:

1. `README.md` with clone-to-inference instructions that work without contacting
   the team.
2. A standalone evaluation `.py` script accepting a test-image directory and an
   output directory, with no manual edits.
3. Reproducible training script or notebook.
4. Final weights (`.pt`, `.onnx`, `.h5`, etc.); use Git LFS or a stable external
   download if large.
5. Restored outputs for the released test inputs.
6. A complete `requirements.txt` / environment record.

The evaluation script is the highest-risk deliverable: test it from a fresh
environment, use relative/configurable paths, automatically load weights, and
process every input deterministically.

## Timeline

| Date | Milestone |
|---|---|
| 24 Jul 2026 | Registration opened |
| 16 Aug 2026 | Registration and initial submission deadline |
| 17-26 Aug 2026 | Round 1 evaluation |
| 27 Aug 2026 | Top 30 announcement |
| 28 Aug 2026 | Round 2 begins |
| 4 Sep 2026 | Round 2 submission deadline |
| 5 Sep 2026 | Semifinal evaluation |
| 6 Sep 2026 | Top 10 announcement |
| 7-12 Sep 2026 | Finalist mentoring |
| 17 Sep 2026 | Grand Finale presentation at Yashobhoomi, New Delhi |
| 18 Sep 2026 | Winner announcement / awards on the main page |

SEMICON India itself is listed as 17-19 September 2026.

## Eligibility

- Undergraduate, graduate, postgraduate, PhD students, or research scholars
- Team of 2-4 members
- Any stream/domain
- The brochure further says students must be from a recognized Indian institution
  and that no prior semiconductor experience is required

## Official inconsistencies to verify before upload

| Topic | Conflicting official statements | Safe action |
|---|---|---|
| Prize pool | Main webpage: ₹5,00,000. Brochure cover: ₹3,00,000. Brochure prize breakdown totals ₹4,00,000 (₹1,25,000 winner + ₹75,000 runner-up for each track). | Do not quote a prize amount in the proposal; ask i4C which figure is current. |
| Slide count | Problem-specific page: maximum 8-9 slides. Downloaded template instruction slide: maximum 6-7 including title, even though the template contains 9 content slides after instructions. | Use the official template, keep all required problem-specific sections concise, and get written confirmation if the portal enforces a limit. |
| Demo video | General webpage checklist: demo video up to 5 minutes. Problem-specific slide 8: optional but recommended. | Prepare a <=5 minute link if at all possible. |
| Finale dates | Main roadmap: presentations 17 Sep and awards 18 Sep. Brochure describes the finale/event as 17-19 Sep. | Keep 17-19 Sep free until finalists receive instructions. |

Official contact printed in the brochure: `support@i4c.in`, `sourabh@i4c.in`,
and `+91 98504 58254`.

## Saved official resources

| Local file | Official source |
|---|---|
| `official-resources/KLA-detailed-problem-statement.pptx` | <https://i4c.in/wp-content/uploads/2026/08/7b675083-e081-47d3-8c55-fde76a77b673.pptx> |
| `official-resources/Idea-Submission-Template_Hackathon-2026.pptx` | <https://i4c.in/wp-content/uploads/2026/07/Idea-Submission-Template_Hackathon-2026-1.pptx> |
| `official-resources/Semicon-India-Hackathon-2026-Brochure.pdf` | <https://i4c.in/wp-content/uploads/2026/07/Semicon-India-Hacakthon-Brochure.pdf> |
| `official-resources/Registration-Process.pdf` | <https://i4c.in/wp-content/uploads/2026/01/How-to-register-for-IESA-Hackathon-2026.pdf> |

KLA webinar links listed by i4C:

- Problem-statement explanation: <https://www.youtube.com/watch?v=RMSDaviTOIw>
- KLA knowledge/Q&A session: <https://www.youtube.com/watch?v=Q__rlK1Q3uw>

## Immediate action checklist

1. Register the final 2-4 member team now.
2. Open the saved idea template and produce the initial PDF before the deadline.
3. Establish a leakage-resistant train/validation split and record the seed.
4. Implement a simple, fast 2x baseline first; measure bicubic, PSNR, SSIM,
   LPIPS, and per-image latency before adding complexity.
5. Keep input values outside [0,1] through preprocessing; constrain or clip only
   the restored output where the evaluation contract requires it.
6. Make the inference CLI and environment reproducible early, not after training.
7. Ask i4C to clarify the four conflicting rules above and the test output format.
