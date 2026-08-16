import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

project_dir = Path("/home/mukundvinayak/semiconductor-image-restoration")
input_dir = project_dir / "main_dataset/Test_NoisyLR/NoisyLR"

curriculum_pred_dir = project_dir / "artifacts/remote-runs/restormer-curriculum-stage2/predictions"
daf_pred_dir = project_dir / "artifacts/remote-runs/daf-final-submission/predictions"
output_path = project_dir / "artifacts/remote-runs/3way_inference_comparison.png"

# Pick 3 representative test files
sample_names = ["000015.npy", "000045.npy", "000120.npy"]

fig, axes = plt.subplots(len(sample_names), 3, figsize=(13, 11))

for row, name in enumerate(sample_names):
    noisy = np.load(input_dir / name, allow_pickle=False)
    curriculum_pred = np.load(curriculum_pred_dir / name, allow_pickle=False)
    daf_pred = np.load(daf_pred_dir / name, allow_pickle=False)
    
    # Col 1: Noisy LR Input
    axes[row, 0].imshow(noisy, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Noisy LR (128x128)\n{name}", fontsize=11, fontweight="bold")
    axes[row, 0].axis("off")
    
    # Col 2: 1.6M Restormer Curriculum (1-View, 13.46ms)
    axes[row, 1].imshow(curriculum_pred, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title("1.6M Restormer Curriculum (1-View)\n~13.46 ms/img | 1.61M Params", fontsize=11, fontweight="bold", color="#1f77b4")
    axes[row, 1].axis("off")
    
    # Col 3: DAF-Restormer (8-View TTA)
    axes[row, 2].imshow(daf_pred, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title("DAF-Restormer-P (8-View TTA)\n~264.72 ms/img | 4.17M Params", fontsize=11, fontweight="bold", color="#2ca02c")
    axes[row, 2].axis("off")

plt.suptitle("Inference & Visual Quality Comparison: 1.6M Restormer vs DAF-Restormer (8-View TTA)", fontsize=14, y=0.99, fontweight="bold")
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Successfully generated 3-way inference comparison preview at {output_path}")
