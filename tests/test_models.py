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


def test_daf_restormer_auxiliary_outputs_and_gradients():
    model = build_model("daf_restormer")
    sample = torch.randn(1, 1, 32, 32, requires_grad=True)
    outputs = model.forward_with_aux(sample)
    assert outputs["prediction"].shape == (1, 1, 64, 64)
    assert outputs["clean_lr"].shape == sample.shape
    assert outputs["uncertainty"].shape == (1, 1, 64, 64)
    assert outputs["noise_map"].shape == sample.shape
    assert torch.all(outputs["uncertainty"] >= 0)
    assert torch.all((outputs["noise_map"] >= 0) & (outputs["noise_map"] <= 1))
    outputs["prediction"].mean().backward()
    assert model.frequency.project.weight.grad is not None
    assert model.degradation.features[0].weight.grad is not None
