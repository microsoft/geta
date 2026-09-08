import pytest
import torch
import torchvision
from conftest import compare_outputs
from torch import nn

from only_train_once.quantization.quant_layers import (
    QuantizationMode,
    QuantizationType,
)
from only_train_once.quantization.quant_model import (
    get_bitwidth_dict,
    get_quant_param_dict,
    model_to_quantize_model,
)


@pytest.fixture
def base_model():
    return torchvision.models.resnet50(pretrained=True)


@pytest.fixture
def sample_input():
    return torch.randn(1, 3, 224, 224)


def test_qresnet50_vs_resnet50_equivalence():
    """Test numerical equivalence between quantized and FP32 ResNet50."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    d_quant_init = 1e-5
    t_quant_init = 1.0
    q_m_init = 10.0
    rtol = 1e-2
    atol = 1e-3

    # Create both models using same seed
    resnet50 = torchvision.models.resnet50(pretrained=True)
    qresnet50 = model_to_quantize_model(
        resnet50,  # Use same model instance to ensure weights match
        d_quant_init,
        t_quant_init,
        q_m_init,
    )

    # Fixed input size (224, 224) which is standard for ResNet
    dummy_input = torch.rand(1, 3, 224, 224)
    assert compare_outputs(qresnet50, resnet50, dummy_input, rtol=rtol, atol=atol)


def test_model_quantization_basic(base_model, sample_input):
    """Test basic model quantization runs."""
    quant_model = model_to_quantize_model(base_model)
    assert isinstance(quant_model, nn.Module)

    with torch.no_grad():
        output = quant_model(sample_input)
        assert output.shape == (1, 1000)


def test_quantization_types(base_model):
    """Test different quantization types can be created."""
    model1 = model_to_quantize_model(base_model, quant_type="symmetric+linear")
    assert isinstance(model1, nn.Module)

    model2 = model_to_quantize_model(
        base_model, quant_type=QuantizationType.SYMMETRIC_NONLINEAR
    )
    assert isinstance(model2, nn.Module)


def test_quantization_modes(base_model):
    """Test different quantization modes can be created."""
    model1 = model_to_quantize_model(base_model, quant_mode="weight_only")
    assert isinstance(model1, nn.Module)

    model2 = model_to_quantize_model(
        base_model, quant_mode=QuantizationMode.WEIGHT_AND_ACTIVATION
    )
    assert isinstance(model2, nn.Module)


def test_invalid_parameters(base_model):
    """Test error handling for invalid parameters."""
    with pytest.raises(ValueError):
        model_to_quantize_model(base_model, quant_type="invalid_type")

    with pytest.raises(ValueError):
        model_to_quantize_model(base_model, quant_mode="invalid_mode")


def test_quant_param_extraction(base_model):
    """Test extraction of quantization parameters."""
    quant_model = model_to_quantize_model(base_model)
    param_dict = get_quant_param_dict(quant_model)
    assert isinstance(param_dict, dict)
    assert len(param_dict) > 0


def test_bitwidth_calculation(base_model):
    """Test bitwidth calculation functionality."""
    quant_model = model_to_quantize_model(base_model)
    param_dict = get_quant_param_dict(quant_model)
    bit_dict = get_bitwidth_dict(param_dict)
    assert isinstance(bit_dict, dict)
    assert len(bit_dict) > 0
