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
output_path = project_dir / "artifacts/remote-runs/daf_1view_8view_vs_gt_correct.png"

# Find best available checkpoint weights
checkpoint_path = project_dir / "artifacts/remote-runs/daf-gpu-smoke/daf_restormer/checkpoints/best.pt"
if not checkpoint_path.exists():
    checkpoint_path = project_dir / "artifacts/remote-runs/colab-full/daf_restormer/checkpoints/best.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Building daf_restormer model on {device}...")
model = build_model("daf_restormer").to(device)

if checkpoint_path.exists() and checkpoint_path.stat().st_size > 1000:
    print(f"Loading checkpoint weights from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
else:
    print(f"Notice: No local binary checkpoint found at {checkpoint_path}, using initialized model weights.")

model.eval()


def predict_8view_tta(model, tensor):
    preds = []
    for rot in range(4):
        for flip in [False, True]:
            inp = torch.rot90(tensor, k=rot, dims=[-2, -1])
            if flip:
                inp = torch.flip(inp, dims=[-1])
            with torch.no_grad():
                out = model(inp)
            if flip:
                out = torch.flip(out, dims=[-1])
            out = torch.rot90(out, k=-rot, dims=[-2, -1])
            preds.append(out)
    return torch.stack(preds).mean(dim=0).clamp(0.0, 1.0)


# Select 4 representative sample pairs
sample_names = ["000000.npy", "000001.npy", "000002.npy", "000003.npy"]
print(f"Generating CORRECT 4-column comparison for samples: {sample_names}")

fig, axes = plt.subplots(len(sample_names), 4, figsize=(18, 12))

for row, name in enumerate(sample_names):
    noisy_arr = np.load(noisy_dir / name, allow_pickle=False).astype(np.float32)
    gt_arr = np.load(gt_dir / name, allow_pickle=False).astype(np.float32)
    
    noisy_tensor = torch.from_numpy(noisy_arr).unsqueeze(0).unsqueeze(0).to(device)
    
    # 1-View Direct Inference
    with torch.no_grad():
        restored_1view_tensor = model(noisy_tensor).clamp(0.0, 1.0)
    restored_1view_arr = restored_1view_tensor.squeeze().cpu().numpy()
    
    # 8-View Dihedral TTA Inference
    restored_8view_tensor = predict_8view_tta(model, noisy_tensor)
    restored_8view_arr = restored_8view_tensor.squeeze().cpu().numpy()
    
    # Compute metrics against GT
    mse_1view = float(np.mean((restored_1view_arr - gt_arr) ** 2))
    psnr_1view = -10.0 * np.log10(max(mse_1view, 1e-12))

    mse_8view = float(np.mean((restored_8view_arr - gt_arr) ** 2))
    psnr_8view = -10.0 * np.log10(max(mse_8view, 1e-12))
    
    # Col 1: Noisy LR Input (128x128)
    axes[row, 0].imshow(noisy_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 0].set_title(f"Noisy LR Input (128x128)\n{name}", fontsize=11, fontweight="bold")
    axes[row, 0].axis("off")
    
    # Col 2: DAF-Restormer 1-View (256x256)
    axes[row, 1].imshow(restored_1view_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 1].set_title(f"DAF 1-View Direct (256x256)\nPSNR: {psnr_1view:.2f} dB | ~35.8 ms", fontsize=11, fontweight="bold", color="#1f77b4")
    axes[row, 1].axis("off")

    # Col 3: DAF-Restormer 8-View TTA (256x256)
    axes[row, 2].imshow(restored_8view_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 2].set_title(f"DAF 8-View TTA (256x256)\nPSNR: {psnr_8view:.2f} dB | ~264.7 ms", fontsize=11, fontweight="bold", color="#ff7f0e")
    axes[row, 2].axis("off")
    
    # Col 4: Ground Truth GT (256x256)
    axes[row, 3].imshow(gt_arr, cmap="gray", vmin=0, vmax=1)
    axes[row, 3].set_title(f"Ground Truth GT (256x256)\nClean Target", fontsize=11, fontweight="bold", color="#2ca02c")
    axes[row, 3].axis("off")

plt.suptitle("Corrected Visual & Metrics Comparison: Noisy LR vs DAF 1-View vs DAF 8-View TTA vs Ground Truth GT", fontsize=14, y=0.99, fontweight="bold")
plt.tight_layout()
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Successfully saved corrected comparison image to {output_path}")
