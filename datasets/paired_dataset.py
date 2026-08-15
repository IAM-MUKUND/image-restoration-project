import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .degradations import MixedDegradationAugmentor

class PairedNpyDataset(Dataset):
    """Dataset for loading paired low-resolution noisy images (NoisyLR) and high-resolution ground truth images (GT)."""
    def __init__(
        self,
        lr_dir: str,
        gt_dir: str,
        augment: bool = True,
        filenames: Optional[Sequence[str]] = None,
        synthetic_degradation_probability: float = 0.0,
    ):
        super().__init__()
        self.lr_dir = Path(lr_dir)
        self.gt_dir = Path(gt_dir)
        self.augment = augment
        self.synthetic_degradation = MixedDegradationAugmentor(
            synthetic_degradation_probability if augment else 0.0
        )

        lr_names = {path.name for path in self.lr_dir.glob("*.npy")}
        gt_names = {path.name for path in self.gt_dir.glob("*.npy")}
        if lr_names != gt_names:
            missing_gt = sorted(lr_names - gt_names)[:10]
            missing_lr = sorted(gt_names - lr_names)[:10]
            raise ValueError(
                "Training pairs do not match by filename. "
                f"Missing GT examples: {missing_gt}; missing LR examples: {missing_lr}"
            )

        selected = sorted(lr_names) if filenames is None else list(filenames)
        unknown = set(selected) - lr_names
        if unknown:
            raise ValueError(f"Unknown dataset filenames: {sorted(unknown)[:10]}")
        self.file_pairs = [
            (self.lr_dir / name, self.gt_dir / name) for name in selected
        ]

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
        lr_arr = np.load(lr_path, allow_pickle=False).astype(np.float32, copy=False)
        gt_arr = np.load(gt_path, allow_pickle=False).astype(np.float32, copy=False)

        # KLA explicitly states that NoisyLR may exceed [0, 1]. Preserve that signal.
        if lr_arr.shape != (128, 128) or gt_arr.shape != (256, 256):
            raise ValueError(
                f"Unexpected pair shapes for {lr_path.name}: {lr_arr.shape} -> {gt_arr.shape}"
            )

        # Apply augmentations if training
        if self.augment:
            lr_arr, gt_arr = self._apply_augmentations(lr_arr, gt_arr)

        # Add channel dimension: (H, W) -> (1, H, W)
        lr_tensor = torch.from_numpy(lr_arr).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_arr).unsqueeze(0)
        lr_tensor = self.synthetic_degradation(gt_tensor, lr_tensor)

        return lr_tensor, gt_tensor, lr_path.stem

def get_dataloaders(
    lr_dir: str,
    gt_dir: str,
    batch_size: int = 16,
    val_ratio: float = 0.1,
    seed: int = 42,
    num_workers: int = 2,
    train_limit: Optional[int] = None,
    val_limit: Optional[int] = None,
    synthetic_degradation_probability: float = 0.0,
):
    """Creates train and validation dataloaders with a fixed reproducible random split."""
    all_names = sorted(path.name for path in Path(lr_dir).glob("*.npy"))
    gt_names = {path.name for path in Path(gt_dir).glob("*.npy")}
    if set(all_names) != gt_names:
        raise ValueError("NoisyLR and GT directories do not contain identical filenames")
    num_total = len(all_names)

    if num_total == 0:
        raise ValueError(f"No .npy files found in {lr_dir} or {gt_dir}")

    num_val = int(num_total * val_ratio)
    num_train = num_total - num_val

    rng = random.Random(seed)
    shuffled_names = list(all_names)
    rng.shuffle(shuffled_names)
    val_names = sorted(shuffled_names[:num_val])
    train_names = sorted(shuffled_names[num_val:])
    if train_limit is not None:
        train_names = train_names[:train_limit]
    if val_limit is not None:
        val_names = val_names[:val_limit]

    # Separate dataset instances prevent validation settings from mutating training.
    train_dataset = PairedNpyDataset(
        lr_dir,
        gt_dir,
        augment=True,
        filenames=train_names,
        synthetic_degradation_probability=synthetic_degradation_probability,
    )
    val_dataset = PairedNpyDataset(lr_dir, gt_dir, augment=False, filenames=val_names)

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        generator=generator,
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
