"""Degradation-aware spatial/frequency Restormer for mixed-noise 2x SR."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from models.restormer.restormer import Downsample, GDFN, LayerNorm2d, MDTA, Upsample


def icnr_init(conv: nn.Conv2d, upscale_factor: int = 2) -> None:
    """ICNR initialisation for a Conv2d that feeds into PixelShuffle.

    Initialises the weights so every sub-pixel position starts with the same
    value (equivalent to a nearest-neighbour upsample), eliminating the
    cross-channel variance imbalance that causes checkerboard / grain artefacts.

    Reference: Aitken et al. 2017 "Checkerboard artifact free sub-pixel
    convolution" (https://arxiv.org/abs/1707.02937).
    """
    out_channels, in_channels, kH, kW = conv.weight.shape
    base_channels = out_channels // (upscale_factor ** 2)
    if out_channels % (upscale_factor ** 2) != 0:
        raise ValueError(
            f"Conv2d out_channels ({out_channels}) must be divisible by "
            f"upscale_factor² ({upscale_factor ** 2}) for ICNR init."
        )
    subkernel = torch.empty(base_channels, in_channels, kH, kW)
    nn.init.kaiming_normal_(subkernel, mode="fan_out", nonlinearity="relu")
    kernel = subkernel.repeat_interleave(upscale_factor ** 2, dim=0)
    conv.weight.data.copy_(kernel)


class DegradationEncoder(nn.Module):
    """Infer global degradation context and a local noise-strength map."""

    def __init__(self, prompt_dim: int = 64, width: int = 24):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 2, width * 3, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width * 3, width * 4, 3, stride=2, padding=1),
            nn.GELU(),
        )
        self.prompt = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(width * 4, prompt_dim),
            nn.GELU(),
            nn.Linear(prompt_dim, prompt_dim),
        )
        self.noise_map = nn.Sequential(
            nn.Conv2d(2, width, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local_mean = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2)
        high_pass = image - local_mean
        evidence = torch.cat((image, high_pass), dim=1)
        return self.prompt(self.features(evidence)), self.noise_map(evidence)


class PromptModulation(nn.Module):
    """Feature-wise affine modulation initialized as an identity transform."""

    def __init__(self, channels: int, prompt_dim: int):
        super().__init__()
        self.to_affine = nn.Sequential(nn.GELU(), nn.Linear(prompt_dim, channels * 2))
        nn.init.zeros_(self.to_affine[-1].weight)
        nn.init.zeros_(self.to_affine[-1].bias)

    def forward(self, features: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        scale, shift = self.to_affine(prompt).chunk(2, dim=1)
        scale = 0.25 * torch.tanh(scale).unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return features * (1.0 + scale) + shift


class PromptedTransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, prompt_dim: int):
        super().__init__()
        self.modulate_attention = PromptModulation(dim, prompt_dim)
        self.norm1 = LayerNorm2d(dim)
        self.attention = MDTA(dim, heads)
        self.modulate_ffn = PromptModulation(dim, prompt_dim)
        self.norm2 = LayerNorm2d(dim)
        self.feed_forward = GDFN(dim)

    def forward(self, features: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        conditioned = self.modulate_attention(features, prompt)
        features = features + self.attention(self.norm1(conditioned))
        conditioned = self.modulate_ffn(features, prompt)
        return features + self.feed_forward(self.norm2(conditioned))


class PromptedStage(nn.Module):
    def __init__(self, dim: int, count: int, heads: int, prompt_dim: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            PromptedTransformerBlock(dim, heads, prompt_dim) for _ in range(count)
        )

    def forward(self, features: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            features = block(features, prompt)
        return features


class FrequencyRefinement(nn.Module):
    """Prompt-conditioned spectral magnitude gate that preserves phase."""

    def __init__(self, channels: int, prompt_dim: int):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.depthwise = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.frequency_gate = nn.Conv2d(channels, channels, 1)
        self.prompt_gate = nn.Linear(prompt_dim, channels)
        self.project = nn.Conv2d(channels, channels, 1, bias=False)
        nn.init.zeros_(self.frequency_gate.weight)
        nn.init.zeros_(self.frequency_gate.bias)
        nn.init.zeros_(self.prompt_gate.weight)
        nn.init.zeros_(self.prompt_gate.bias)

    def forward(self, features: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        normalized = self.norm(features).float()
        spectrum = torch.fft.rfft2(normalized, norm="ortho")
        magnitude = torch.log1p(torch.abs(spectrum))
        local_gate = self.frequency_gate(F.gelu(self.depthwise(magnitude)))
        global_gate = self.prompt_gate(prompt.float()).unsqueeze(-1).unsqueeze(-1)
        modulation = 0.5 * torch.tanh(local_gate + global_gate)
        refined = torch.fft.irfft2(spectrum * (1.0 + modulation), s=(height, width), norm="ortho")
        return features + self.project(refined.to(features.dtype))


class DAFRestormerSR(nn.Module):
    """Degradation-Aware Frequency Restormer with progressive supervision."""

    def __init__(
        self,
        dim: int = 40,
        prompt_dim: int = 64,
        blocks_per_level: tuple[int, int, int] = (3, 4, 6),
    ):
        super().__init__()
        self.degradation = DegradationEncoder(prompt_dim=prompt_dim)
        self.embed = nn.Conv2d(2, dim, 3, padding=1, bias=False)
        self.noise_inject1 = nn.Conv2d(1, dim, 3, padding=1, bias=False)
        self.enc1 = PromptedStage(dim, blocks_per_level[0], 1, prompt_dim)

        self.down1 = Downsample(dim)
        self.noise_inject2 = nn.Conv2d(1, dim * 2, 3, padding=1, bias=False)
        self.enc2 = PromptedStage(dim * 2, blocks_per_level[1], 2, prompt_dim)

        self.down2 = Downsample(dim * 2)
        self.noise_inject3 = nn.Conv2d(1, dim * 4, 3, padding=1, bias=False)
        self.latent = PromptedStage(dim * 4, blocks_per_level[2], 4, prompt_dim)
        self.frequency = FrequencyRefinement(dim * 4, prompt_dim)

        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = PromptedStage(dim * 2, blocks_per_level[1], 2, prompt_dim)
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = PromptedStage(dim, blocks_per_level[0], 1, prompt_dim)

        self.clean_lr_head = nn.Conv2d(dim, 1, 3, padding=1)
        _sr_conv = nn.Conv2d(dim, dim * 4, 3, padding=1)
        icnr_init(_sr_conv, upscale_factor=2)
        self.sr = nn.Sequential(_sr_conv, nn.PixelShuffle(2))
        self.output = nn.Conv2d(dim, 1, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # Decouple uncertainty to branch independently off d1 at LR scale
        self.uncertainty = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim // 2, 1, 3, padding=1),
            nn.Softplus(),
        )

    def _restore(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        prompt, noise_map = self.degradation(image)
        embedded = self.embed(torch.cat((image, noise_map), dim=1))
        e1 = self.enc1(embedded + self.noise_inject1(noise_map), prompt)

        noise2 = F.avg_pool2d(noise_map, 2)
        e2 = self.enc2(self.down1(e1) + self.noise_inject2(noise2), prompt)

        noise3 = F.avg_pool2d(noise2, 2)
        latent = self.latent(self.down2(e2) + self.noise_inject3(noise3), prompt)
        latent = self.frequency(latent, prompt)

        d2 = self.reduce2(torch.cat((self.up2(latent), e2), dim=1))
        d2 = self.dec2(d2, prompt)
        d1 = self.reduce1(torch.cat((self.up1(d2), e1), dim=1))
        d1 = self.dec1(d1, prompt)

        clean_lr = image + self.clean_lr_head(d1)
        hr_features = self.sr(d1)
        base = F.interpolate(clean_lr, scale_factor=2, mode="bicubic", align_corners=False, antialias=True)
        prediction = base + self.output(hr_features)
        return {
            "prediction": prediction,
            "clean_lr": clean_lr,
            "uncertainty": self.uncertainty(d1),
            "noise_map": noise_map,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self._restore(image)["prediction"]

    def forward_with_aux(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._restore(image)
