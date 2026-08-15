import random

import torch

from datasets.degradations import MixedDegradationAugmentor


def test_mixed_degradation_augmentation_shape_and_finiteness():
    random.seed(9)
    torch.manual_seed(9)
    clean = torch.linspace(0, 1, 256 * 256).reshape(1, 256, 256)
    official = torch.zeros(1, 128, 128)
    degraded = MixedDegradationAugmentor(probability=1.0)(clean, official)
    assert degraded.shape == official.shape
    assert degraded.dtype == torch.float32
    assert torch.isfinite(degraded).all()
    assert not torch.equal(degraded, official)
