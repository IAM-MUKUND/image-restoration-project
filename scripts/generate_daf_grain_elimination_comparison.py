import os
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

project_dir = Path("/home/mukundvinayak/semiconductor-image-restoration")
sys.path.insert(0, str(project_dir))

from models import build_model

data_root = project_dir / "main_dataset/train/train"
noisy_dir = data_root / "NoisyLR"
gt_dir = data_root / "GT"

orig_checkpoint_path = project_dir / "artifacts/remote-runs/daf-gpu-smoke/daf_restormer/checkpoints/best.pt"
new_checkpoint_path = project_dir / "artifacts/remote-runs/daf-curriculum-stage2/daf_restormer/checkpoints/best.pt"

output_path = project_dir / "artifacts/remote-runs/daf_grain_elimination_comparison.png"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Original DAF model
orig_model = build_model("daf_restormer").to(device)
if orig_checkpoint_path.exists():
    ckpt = torch.load(orig_checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    orig_model.load_state_dict(state_dict, strict=False)
orig_model.eval()

# Load New Grain-Eliminated DAF Curriculum model
new_model = build_model("daf_restormer").to(device)
if new_checkpoint_path.exists():
    ckpt = torch.load(new_checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    new_model.load_state_dict(state_dict, strict=False)
new_model.eval()

# Select 4 representative samples
sample_names = ["000000.npy", "000001.npy", "000002.npy", "000003.npy"]

fig, axes = plt.subplots(len(sample_names), 4, figsize=(18, 13))

for row, name in enumerate(sample_names):
    noisy_arr = np.load(noisy_dir / name, allow_pickle=False).astype(np.float32)
    gt_arr = np.load(gt_dir / name, allow_pickle=False).astype(np.float32)
    noisy_tensor = torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0).to(device)
    
    with torch.no_grad():
        orig_out = orig_model(noisy_tensor).clamp(0.0, 1.0).squeeze().cpu().numpy()
        new_out = new_model(noisy_tensor).clamp(0.0, 1.0).squeeze().cpu().numpy()
    
    mse_orig = float(np.mean((orig_out - gt_arr) ** 2))
    psnr_orig = -10.0 * np.log10(max(mse_orig, 1e-12))

    mse_new = float(np.mean((new_out - gt_arr) ** 2))
    psnr_new = -10.0 * np.log10(max(mse_new, 1e-12))

    # Col 1: Noisy LR Input
    axes[row, 0].imshow(noisy_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Noisy LR (128x128)\n{name}", fontsize=11, fontweight="bold")
    axes[row, 0].axis("off")

    # Col 2: Original DAF 1-View (With residual grain)
    axes[row, 1].imshow(orig_out, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title(f"Original DAF 1-View\nPSNR: {psnr_orig:.2f} dB (Grain Present)", fontsize=10, fontweight="bold", color="#d62728")
    axes[row, 1].axis("off")

    # Col 3: New DAF Curriculum (Fourier + Smooth Loss)
    axes[row, 2].imshow(new_out, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title(f"New DAF Curriculum (1-View)\nPSNR: {psnr_new:.2f} dB (Grain Suppressed)", fontsize=10, fontweight="bold", color="#1f77b4")
    axes[row, 2].axis("off")

    # Col 4: Ground Truth GT Target
    axes[row, 3].imshow(gt_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 3].set_title(f"Ground Truth GT (256x256)\nClean Target", fontsize=11, fontweight="bold", color="#2ca02c")
    axes[row, 3].axis("off")

plt.suptitle("Grain Elimination Comparison: Original DAF vs New DAF Curriculum (Fourier + Smooth Variance Loss) vs GT", fontsize=13, y=0.99, fontweight="bold")
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Comparison image successfully saved to {output_path}")
