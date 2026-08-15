"""Compact Restormer adaptation for joint grayscale denoising and 2x SR."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        return (x - mean) * torch.rsqrt(variance + self.eps) * self.weight + self.bias


class MDTA(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_dw = nn.Conv2d(dim * 3, dim * 3, 3, padding=1, groups=dim * 3, bias=False)
        self.project = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv_dw(self.qkv(x)).chunk(3, dim=1)
        channels = c // self.heads
        q = q.reshape(b, self.heads, channels, h * w)
        k = k.reshape(b, self.heads, channels, h * w)
        v = v.reshape(b, self.heads, channels, h * w)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attention = (q @ k.transpose(-2, -1)) * self.temperature
        attention = attention.softmax(dim=-1)
        out = (attention @ v).reshape(b, c, h, w)
        return self.project(out)


class GDFN(nn.Module):
    def __init__(self, dim: int, expansion: float = 2.66):
        super().__init__()
        hidden = int(dim * expansion)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=False)
        self.depthwise = nn.Conv2d(hidden * 2, hidden * 2, 3, padding=1, groups=hidden * 2, bias=False)
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.depthwise(self.project_in(x)).chunk(2, dim=1)
        return self.project_out(F.gelu(left) * right)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attention = MDTA(dim, heads)
        self.norm2 = LayerNorm2d(dim)
        self.feed_forward = GDFN(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.feed_forward(self.norm2(x))


def blocks(dim: int, count: int, heads: int) -> nn.Sequential:
    return nn.Sequential(*(TransformerBlock(dim, heads) for _ in range(count)))


class Downsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, padding=1, bias=False), nn.PixelShuffle(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class RestormerSR(nn.Module):
    def __init__(self, dim: int = 32, blocks_per_level: tuple[int, int, int] = (2, 2, 4)):
        super().__init__()
        self.embed = nn.Conv2d(1, dim, 3, padding=1, bias=False)
        self.enc1 = blocks(dim, blocks_per_level[0], 1)
        self.down1 = Downsample(dim)
        self.enc2 = blocks(dim * 2, blocks_per_level[1], 2)
        self.down2 = Downsample(dim * 2)
        self.latent = blocks(dim * 4, blocks_per_level[2], 4)
        self.up2 = Upsample(dim * 4)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=False)
        self.dec2 = blocks(dim * 2, blocks_per_level[1], 2)
        self.up1 = Upsample(dim * 2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.dec1 = blocks(dim, blocks_per_level[0], 1)
        self.sr = nn.Sequential(nn.Conv2d(dim, dim * 4, 3, padding=1), nn.PixelShuffle(2))
        self.output = nn.Conv2d(dim, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        e1 = self.enc1(self.embed(x))
        e2 = self.enc2(self.down1(e1))
        z = self.latent(self.down2(e2))
        d2 = self.dec2(self.reduce2(torch.cat((self.up2(z), e2), dim=1)))
        d1 = self.dec1(self.reduce1(torch.cat((self.up1(d2), e1), dim=1)))
        return self.output(self.sr(d1)) + base
