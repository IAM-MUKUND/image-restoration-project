from pathlib import Path

import numpy as np

from datasets.paired_dataset import PairedNpyDataset, get_dataloaders


def write_pair(root: Path, name: str, noisy_value: float = 1.5) -> None:
    noisy_dir = root / "NoisyLR"
    gt_dir = root / "GT"
    noisy_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    np.save(noisy_dir / name, np.full((128, 128), noisy_value, dtype=np.float32))
    np.save(gt_dir / name, np.full((256, 256), 0.5, dtype=np.float32))


def test_loader_preserves_out_of_range_noisy_signal(tmp_path: Path):
    write_pair(tmp_path, "000000.npy")
    dataset = PairedNpyDataset(tmp_path / "NoisyLR", tmp_path / "GT", augment=False)
    noisy, target, name = dataset[0]
    assert float(noisy.max()) == 1.5
    assert target.shape == (1, 256, 256)
    assert name == "000000"


def test_train_and_validation_have_independent_augmentation_settings(tmp_path: Path):
    for index in range(10):
        write_pair(tmp_path, f"{index:06d}.npy")
    train, validation = get_dataloaders(
        tmp_path / "NoisyLR",
        tmp_path / "GT",
        batch_size=2,
        val_ratio=0.2,
        num_workers=0,
    )
    assert train.dataset.augment is True
    assert validation.dataset.augment is False
    assert len(train.dataset) == 8
    assert len(validation.dataset) == 2
