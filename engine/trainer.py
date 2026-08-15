import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from metrics.psnr import calculate_psnr

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device
) -> float:
    """Trains model for one epoch using Automatic Mixed Precision (AMP)."""
    model.train()
    total_loss = 0.0

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for lr_imgs, gt_imgs in pbar:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        gt_imgs = gt_imgs.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred_imgs = model(lr_imgs)
            loss = criterion(pred_imgs, gt_imgs)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * lr_imgs.size(0)
        pbar.set_postfix({'loss': f"{loss.item():.6f}"})

    return total_loss / len(dataloader.dataset)

@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple[float, float]:
    """Evaluates model performance on validation dataset, returning loss and average PSNR."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0

    for lr_imgs, gt_imgs in dataloader:
        lr_imgs = lr_imgs.to(device, non_blocking=True)
        gt_imgs = gt_imgs.to(device, non_blocking=True)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            pred_imgs = model(lr_imgs)
            loss = criterion(pred_imgs, gt_imgs)

        total_loss += loss.item() * lr_imgs.size(0)
        
        # Calculate PSNR per batch item
        for p, g in zip(pred_imgs, gt_imgs):
            total_psnr += calculate_psnr(p, g)

    avg_loss = total_loss / len(dataloader.dataset)
    avg_psnr = total_psnr / len(dataloader.dataset)

    return avg_loss, avg_psnr
