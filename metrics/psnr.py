import torch
import math

def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) between prediction and ground truth target tensors."""
    pred = torch.clamp(pred, 0.0, max_val)
    target = torch.clamp(target, 0.0, max_val)

    mse = torch.mean((pred - target) ** 2).item()
    if mse == 0:
        return float('inf')

    return 20.0 * math.log10(max_val) - 10.0 * math.log10(mse)
