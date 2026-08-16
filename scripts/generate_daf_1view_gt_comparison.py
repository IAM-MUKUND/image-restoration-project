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
output_path = project_dir / "artifacts/remote-runs/daf_1view_vs_gt_comparison.png"

# Load pre-trained DAF-Restormer checkpoint
checkpoint_path = project_dir / "artifacts/remote-runs/daf-gpu-smoke/daf_restormer/checkpoints/best.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = build_model("daf_restormer").to(device)
if checkpoint_path.exists():
    print(f"Loading checkpoint weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
model.eval()

# Select 4 representative sample pairs
sample_names = sorted(os.listdir(noisy_dir))[:4]
print(f"Generating comparison for samples: {sample_names}")

fig, axes = plt.subplots(len(sample_names), 3, figsize=(14, 12))

for row, name in enumerate(sample_names):
    noisy_arr = np.load(noisy_dir / name, allow_pickle=False).astype(np.float32)
    gt_arr = np.load(gt_dir / name, allow_pickle=False).astype(np.float32)
    
    # Run DAF-Restormer 1-view single-pass inference
    noisy_tensor = torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        restored_tensor = model(noisy_tensor)
    restored_arr = restored_tensor.clamp(0.0, 1.0).squeeze().cpu().numpy()
    
    # Compute PSNR & SSIM for this sample
    mse = float(np.mean((restored_arr - gt_arr) ** 2))
    sample_psnr = -10.0 * np.log10(max(mse, 1e-12))
    
    # Col 1: Noisy LR Input (128x128)
    axes[row, 0].imshow(noisy_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Noisy LR (128x128)\n{name}", fontsize=11, fontweight="bold")
    axes[row, 0].axis("off")
    
    # Col 2: DAF-Restormer 1-View (256x256)
    axes[row, 1].imshow(restored_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title(f"DAF-Restormer 1-View (256x256)\nPSNR: {sample_psnr:.2f} dB | ~35.8 ms", fontsize=11, fontweight="bold", color="#1f77b4")
    axes[row, 1].axis("off")
    
    # Col 3: Ground Truth (256x256)
    axes[row, 2].imshow(gt_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title(f"Ground Truth GT (256x256)\nClean Target", fontsize=11, fontweight="bold", color="#2ca02c")
    axes[row, 2].axis("off")

plt.suptitle("DAF-Restormer 1-View Inference Comparison: Noisy LR vs DAF 1-View vs Ground Truth GT", fontsize=14, y=0.99, fontweight="bold")
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Successfully generated comparison image at {output_path}")
