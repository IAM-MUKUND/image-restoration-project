import torch

from losses import CombinedRestorationLoss, FrequencyLoss, GradientLoss


def test_frequency_and_gradient_losses_are_zeroish_for_identical_images():
    image = torch.rand(2, 1, 32, 32)
    assert FrequencyLoss()(image, image).item() == 0.0
    assert GradientLoss()(image, image).item() < 0.0011


def test_combined_loss_supports_progressive_and_uncertainty_outputs():
    prediction = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand_like(prediction)
    clean_lr = torch.rand(2, 1, 32, 32, requires_grad=True)
    uncertainty = torch.rand_like(prediction, requires_grad=True)
    criterion = CombinedRestorationLoss(
        ssim_weight=0.1,
        frequency_weight=0.03,
        gradient_weight=0.02,
        auxiliary_weight=0.2,
        uncertainty_weight=0.01,
    )
    loss = criterion(prediction, target, clean_lr=clean_lr, uncertainty=uncertainty)
    assert torch.isfinite(loss)
    loss.backward()
    assert prediction.grad is not None
    assert clean_lr.grad is not None
    assert uncertainty.grad is not None
