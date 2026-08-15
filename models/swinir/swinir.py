"""Compact SwinIR adaptation with shifted-window attention and a 2x head."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


def window_partition(x: torch.Tensor, window: int) -> torch.Tensor:
    b, h, w, c = x.shape
    x = x.view(b, h // window, window, w // window, window, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window * window, c)


def window_reverse(windows: torch.Tensor, window: int, h: int, w: int, b: int) -> torch.Tensor:
    x = windows.view(b, h // window, w // window, window, window, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window: int, heads: int):
        super().__init__()
        self.dim = dim
        self.window = window
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        size = (2 * window - 1) ** 2
        self.relative_bias = nn.Parameter(torch.zeros(size, heads))
        coords = torch.stack(torch.meshgrid(torch.arange(window), torch.arange(window), indexing="ij"))
        flat = coords.flatten(1)
        relative = flat[:, :, None] - flat[:, None, :]
        relative = relative.permute(1, 2, 0).contiguous()
        relative[:, :, 0] += window - 1
        relative[:, :, 1] += window - 1
        relative[:, :, 0] *= 2 * window - 1
        self.register_buffer("relative_index", relative.sum(-1), persistent=False)
        nn.init.trunc_normal_(self.relative_bias, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.heads, channels // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attention = (q * self.scale) @ k.transpose(-2, -1)
        bias = self.relative_bias[self.relative_index.reshape(-1)]
        bias = bias.reshape(tokens, tokens, self.heads).permute(2, 0, 1)
        attention = attention + bias.unsqueeze(0)
        if mask is not None:
            n_windows = mask.shape[0]
            attention = attention.view(batch_windows // n_windows, n_windows, self.heads, tokens, tokens)
            attention = attention + mask.unsqueeze(0).unsqueeze(2)
            attention = attention.view(-1, self.heads, tokens, tokens)
        attention = attention.softmax(dim=-1)
        out = (attention @ v).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int = 8, shift: int = 0, mlp_ratio: float = 2.0):
        super().__init__()
        self.window = window
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attention = WindowAttention(dim, window, heads)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def _mask(self, h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        if self.shift == 0:
            return None
        img_mask = torch.zeros((1, h, w, 1), device=device, dtype=dtype)
        h_slices = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
        w_slices = (slice(0, -self.window), slice(-self.window, -self.shift), slice(-self.shift, None))
        count = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = count
                count += 1
        mask_windows = window_partition(img_mask, self.window).squeeze(-1)
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, original_h, original_w = x.shape
        pad_h = (self.window - original_h % self.window) % self.window
        pad_w = (self.window - original_w % self.window) % self.window
        x_nhwc = x.permute(0, 2, 3, 1)
        if pad_h or pad_w:
            x_nhwc = F.pad(x_nhwc, (0, 0, 0, pad_w, 0, pad_h))
        _, h, w, _ = x_nhwc.shape
        shortcut = x_nhwc
        y = self.norm1(x_nhwc)
        if self.shift:
            y = torch.roll(y, shifts=(-self.shift, -self.shift), dims=(1, 2))
        windows = window_partition(y, self.window)
        windows = self.attention(windows, self._mask(h, w, x.device, x.dtype))
        y = window_reverse(windows, self.window, h, w, b)
        if self.shift:
            y = torch.roll(y, shifts=(self.shift, self.shift), dims=(1, 2))
        y = shortcut + y
        y = y + self.mlp(self.norm2(y))
        y = y[:, :original_h, :original_w, :]
        return y.permute(0, 3, 1, 2).contiguous()


class ResidualSwinGroup(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, window: int):
        super().__init__()
        self.blocks = nn.Sequential(*(
            SwinBlock(dim, heads, window, 0 if index % 2 == 0 else window // 2)
            for index in range(depth)
        ))
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(self.blocks(x))


class SwinIRSR(nn.Module):
    def __init__(self, dim: int = 48, groups: int = 3, depth: int = 4, heads: int = 4, window: int = 8):
        super().__init__()
        self.first = nn.Conv2d(1, dim, 3, padding=1)
        self.body = nn.Sequential(*(ResidualSwinGroup(dim, depth, heads, window) for _ in range(groups)))
        self.after_body = nn.Conv2d(dim, dim, 3, padding=1)
        self.upsample = nn.Sequential(nn.Conv2d(dim, dim * 4, 3, padding=1), nn.PixelShuffle(2))
        self.last = nn.Conv2d(dim, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.interpolate(x, scale_factor=2, mode="bicubic", align_corners=False)
        shallow = self.first(x)
        features = shallow + self.after_body(self.body(shallow))
        return self.last(self.upsample(features)) + base
