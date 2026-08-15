import os
import glob
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class PairedNpyDataset(Dataset):
    """Dataset for loading paired low-resolution noisy images (NoisyLR) and high-resolution ground truth images (GT)."""
    def __init__(self, lr_dir: str, gt_dir: str, augment: bool = True):
        super().__init__()
        self.lr_dir = lr_dir
        self.gt_dir = gt_dir
        self.augment = augment

        self.lr_filenames = sorted(glob.glob(os.path.join(lr_dir, "*.npy")))
        self.gt_filenames = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))

        if len(self.lr_filenames) == 0 or len(self.gt_filenames) == 0:
            # Fallback for empty or non-existent paths (useful during initial exploration)
            self.file_pairs = []
        else:
            assert len(self.lr_filenames) == len(self.gt_filenames), (
                f"Mismatch in dataset size: {len(self.lr_filenames)} LR vs {len(self.gt_filenames)} GT"
            )
            self.file_pairs = list(zip(self.lr_filenames, self.gt_filenames))

    def __len__(self) -> int:
        return len(self.file_pairs)

    def _apply_augmentations(self, lr: np.ndarray, gt: np.ndarray):
        """Applies consistent spatial augmentations to both LR and GT arrays."""
        # Horizontal Flip
        if random.random() > 0.5:
            lr = np.fliplr(lr)
            gt = np.fliplr(gt)

        # Vertical Flip
        if random.random() > 0.5:
            lr = np.flipud(lr)
            gt = np.flipud(gt)

        # 90-degree Rotation
        rot_k = random.randint(0, 3)
        if rot_k > 0:
            lr = np.rot90(lr, k=rot_k)
            gt = np.rot90(gt, k=rot_k)

        return lr.copy(), gt.copy()

    def __getitem__(self, idx: int):
        lr_path, gt_path = self.file_pairs[idx]

        # Load raw float32 arrays
        lr_arr = np.load(lr_path).astype(np.float32)
        gt_arr = np.load(gt_path).astype(np.float32)

        # Clip values to valid normalized range [0.0, 1.0]
        lr_arr = np.clip(lr_arr, 0.0, 1.0)
        gt_arr = np.clip(gt_arr, 0.0, 1.0)

        # Apply augmentations if training
        if self.augment:
            lr_arr, gt_arr = self._apply_augmentations(lr_arr, gt_arr)

        # Add channel dimension: (H, W) -> (1, H, W)
        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0)

        return lr_tensor, gt_tensor

def get_dataloaders(
    lr_dir: str,
    gt_dir: str,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 2
):
    """Creates train and validation dataloaders with a fixed reproducible random split."""
    full_dataset = PairedNpyDataset(lr_dir=lr_dir, gt_dir=gt_dir, augment=True)
    num_total = len(full_dataset)

    if num_total == 0:
        raise ValueError(f"No .npy files found in {lr_dir} or {gt_dir}")

    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(full_dataset, [num_train, num_val], generator=generator)

    # Disable data augmentation on validation set
    val_dataset.dataset.augment = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
