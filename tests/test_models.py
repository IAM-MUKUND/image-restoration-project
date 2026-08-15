import pytest
import torch

from models import MODEL_NAMES, build_model


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_restores_expected_shape(name: str):
    model = build_model(name).eval()
    sample = torch.randn(1, 1, 32, 32)
    with torch.inference_mode():
        output = model(sample)
    assert output.shape == (1, 1, 64, 64)
    assert torch.isfinite(output).all()
