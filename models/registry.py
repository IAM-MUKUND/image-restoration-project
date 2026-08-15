"""Model registry for fair, task-adapted architecture benchmarks."""

from __future__ import annotations

from torch import nn

from .esrgan import ESRGANGeneratorSR
from .mprnet import MPRNetSR
from .nafnet import NAFNetSR
from .restormer import RestormerSR
from .swinir import SwinIRSR


MODEL_NAMES = ("nafnet", "swinir", "restormer", "esrgan_rrdb", "mprnet")


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "nafnet":
        return NAFNetSR(width=24, enc_blocks=(1, 1), middle_blocks=2, dec_blocks=(1, 1))
    if name == "swinir":
        return SwinIRSR(dim=48, groups=3, depth=4, heads=4, window=8)
    if name == "restormer":
        return RestormerSR(dim=32, blocks_per_level=(2, 2, 4))
    if name == "esrgan_rrdb":
        return ESRGANGeneratorSR(channels=32, growth=16, rrdb_blocks=6)
    if name == "mprnet":
        return MPRNetSR(channels=32, blocks_per_stage=4)
    raise KeyError(f"Unknown model {name!r}. Available: {', '.join(MODEL_NAMES)}")
