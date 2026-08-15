import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    """Channel-first 2D Layer Normalization."""
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x

class SimpleGate(nn.Module):
    """Splits input tensor along channel dimension into two equal halves and computes element-wise multiplication."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """Non-Linear Activation Free Block (NAFBlock)."""
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw_channel = c * dw_expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel)
        self.sg1 = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1)
        )
        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1)

        ffn_channel = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_channel, 1)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = res + x * self.beta

        res = y
        y = self.norm2(y)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        return res + y * self.gamma

class NAFNetSR(nn.Module):
    """NAFNet configured for Joint 2x Super-Resolution & Denoising (128x128 -> 256x256)."""
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        enc_blocks: list = [2, 2],
        middle_blocks: int = 4,
        dec_blocks: list = [2, 2],
        upscale: int = 2
    ):
        super().__init__()
        self.upscale = upscale
        self.intro = nn.Conv2d(in_channels, width, 3, padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        curr_width = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(n)]))
            self.downs.append(nn.Conv2d(curr_width, curr_width * 2, 2, stride=2))
            curr_width *= 2

        # Middle
        self.middle = nn.Sequential(*[NAFBlock(curr_width) for _ in range(middle_blocks)])

        # Decoder
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(curr_width, curr_width * 2, 1),
                nn.PixelShuffle(2)
            ))
            curr_width = curr_width // 2
            self.decoders.append(nn.Sequential(*[NAFBlock(curr_width) for _ in range(n)]))

        # 2x Super Resolution Head
        self.sr_upsample = nn.Sequential(
            nn.Conv2d(curr_width, curr_width * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale),
            NAFBlock(curr_width)
        )
        self.ending = nn.Conv2d(curr_width, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.intro(x)
        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feats = encoder(feats)
            skips.append(feats)
            feats = down(feats)

        feats = self.middle(feats)

        for up, decoder in zip(self.ups, self.decoders):
            feats = up(feats)
            feats = feats + skips.pop()
            feats = decoder(feats)

        feats = self.sr_upsample(feats)
        out = self.ending(feats)
        return out
