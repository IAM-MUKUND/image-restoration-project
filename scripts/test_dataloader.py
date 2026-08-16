import sys, os
sys.path.insert(0, os.getcwd())

import torch
from datasets.paired_dataset import get_dataloaders

lr_dir = "main_dataset/train/train/NoisyLR"
gt_dir = "main_dataset/train/train/GT"

print("Testing dataloader with crop_size=64 AND synthetic_degradation_probability=0.5...")
train_loader, val_loader = get_dataloaders(
    lr_dir,
    gt_dir,
    batch_size=4,
    crop_size=64,
    synthetic_degradation_probability=0.5,
    train_limit=32,
    val_limit=16,
)

for noisy, target, names in train_loader:
    print(f"Train batch - Noisy shape: {noisy.shape}, Target shape: {target.shape}")
    assert noisy.shape == torch.Size([4, 1, 64, 64]), f"Expected [4, 1, 64, 64], got {noisy.shape}"
    assert target.shape == torch.Size([4, 1, 128, 128]), f"Expected [4, 1, 128, 128], got {target.shape}"

print("Synthetic degradation + crop_size test PASSED!")
