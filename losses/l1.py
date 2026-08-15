import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (Smooth L1 Loss variant for super-resolution and image restoration)."""
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + (self.eps * self.eps))
        return torch.mean(loss)
