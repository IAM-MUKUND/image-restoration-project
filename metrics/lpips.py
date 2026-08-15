from __future__ import annotations

import torch


class LPIPSMetric:
    def __init__(self, device: torch.device):
        import lpips as lpips_package

        self.model = lpips_package.LPIPS(net="alex").to(device).eval()

    @torch.inference_mode()
    def __call__(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prediction = prediction.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
        target = target.clamp(0.0, 1.0).repeat(1, 3, 1, 1) * 2.0 - 1.0
        return self.model(prediction, target, normalize=False).flatten()
