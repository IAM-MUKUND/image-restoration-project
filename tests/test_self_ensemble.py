import torch
from torch import nn

from models.self_ensemble import GeometricSelfEnsemble, ResidualCalibrator


class NearestTwoX(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.interpolate(image, scale_factor=2, mode="nearest")


def test_geometric_self_ensemble_preserves_equivariant_prediction() -> None:
    image = torch.randn(2, 1, 8, 8)
    expected = NearestTwoX()(image)
    for transforms in (1, 4, 8):
        actual = GeometricSelfEnsemble(NearestTwoX(), transforms)(image)
        assert actual.shape == (2, 1, 16, 16)
        torch.testing.assert_close(actual, expected)


def test_geometric_self_ensemble_rejects_unsupported_count() -> None:
    try:
        GeometricSelfEnsemble(NearestTwoX(), transforms=2)
    except ValueError as error:
        assert "1, 4, or 8" in str(error)
    else:
        raise AssertionError("unsupported transform count was accepted")


def test_residual_calibrator_identity_and_bias() -> None:
    image = torch.randn(1, 1, 8, 8)
    prediction = NearestTwoX()(image)
    bicubic = torch.nn.functional.interpolate(
        image, scale_factor=2, mode="bicubic", align_corners=False
    )
    identity = ResidualCalibrator(NearestTwoX())(image)
    shifted = ResidualCalibrator(NearestTwoX(), residual_gain=0.7, bias=0.01)(image)
    torch.testing.assert_close(identity, prediction)
    torch.testing.assert_close(shifted, bicubic + 0.7 * (prediction - bicubic) + 0.01)
