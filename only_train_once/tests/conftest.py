"""Shared fixtures for tests."""

import torch


def compare_outputs(quant_model, fp32_model, input_tensor, rtol=1e-5, atol=1e-8):
    """Test numerical equivalence between quantized and FP32 models."""
    quant_model.eval()
    fp32_model.eval()

    with torch.no_grad():
        quant_output = quant_model(input_tensor)
        fp32_output = fp32_model(input_tensor)

    is_close = torch.allclose(quant_output, fp32_output, rtol=rtol, atol=atol)
    if not is_close:
        max_diff = torch.max(torch.abs(quant_output - fp32_output))
        print(f"Is_close: {is_close}, Max difference is {max_diff}")
        assert max_diff <= rtol
    print(f"Is_close: {is_close}")
    return is_close
