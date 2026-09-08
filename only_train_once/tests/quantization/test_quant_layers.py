import torch
from conftest import compare_outputs
from torch import nn

from only_train_once.quantization.quant_layers import (
    DGEQuantizer,
    QuantizeConv2d,
    QuantizeLinear,
)


def test_qlinear_vs_linear_equivalence():
    """Test numerical equivalence between quantized and FP32 Linear layer."""
    in_feats = 256
    out_feats = 128
    bias = True
    rtol = 1e-4
    atol = 1e-8

    q_linear = QuantizeLinear(
        in_features=in_feats,
        out_features=out_feats,
        bias=bias,
        d_quant_init=1e-4,
        t_quant_init=1.0,
        q_m_init=10.0,
    )
    print(q_linear.weight_bit)
    linear = nn.Linear(in_features=in_feats, out_features=out_feats, bias=bias)

    linear.weight.data.copy_(torch.rand(out_feats, in_feats))
    q_linear.weight.data.copy_(linear.weight.data)
    if bias:
        q_linear.bias.data.copy_(linear.bias.data)

    dummy_input = torch.rand(1, in_feats)
    assert compare_outputs(q_linear, linear, dummy_input, rtol=rtol, atol=atol)


def test_qconv2d_vs_conv2d_equivalence():
    """Test numerical equivalence between quantized and FP32 Conv2d layer."""
    in_channels = 3
    out_channels = 64
    kernel_size = 3
    stride = 1
    padding = 1
    bias = True
    rtol = 1e-4
    atol = 1e-8

    q_conv = QuantizeConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        d_quant_init=1e-4,
        t_quant_init=1.0,
        q_m_init=10.0,
    )
    print(q_conv.weight_bit)
    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
    )

    conv.weight.data.copy_(
        torch.rand(out_channels, in_channels, kernel_size, kernel_size)
    )
    q_conv.weight.data.copy_(conv.weight.data)
    if bias:
        q_conv.bias.data.copy_(conv.bias.data)

    dummy_input = torch.rand(1, in_channels, 32, 32)
    assert compare_outputs(q_conv, conv, dummy_input, rtol=rtol, atol=atol)


def test_dge_quantizer():
    """Test DGE quantizer maintains trainable quantization in forward pass
    but uses DGE gradient estimation in backward."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup trainable parameters
    input = torch.tensor([0.3], device=device, requires_grad=True)
    d_quant = torch.tensor([0.2], device=device, requires_grad=True)
    q_m = torch.tensor([1.0], device=device, requires_grad=True)
    clip_val = torch.tensor([-1.0, 1.0], device=device)
    q_s = torch.tensor([0.0], device=device)
    num_bits = torch.tensor([4.0], device=device)  # k=5 case

    # Forward pass should use normal quantization
    output = DGEQuantizer.apply(input, d_quant, q_m, clip_val, q_s, num_bits)

    # Verify forward pass is using regular quantization
    expected = d_quant * torch.round(input / d_quant)
    assert torch.allclose(output, expected)

    # Run backward pass
    output.backward()

    # Verify input gradient is computed using DGE formula
    # f'(x) = (1/k) · |x - δ/2|^(1/k - 1) with k=5
    x_centered = input - d_quant / 2
    k = 5.0
    expected_grad = (1 / k) * torch.pow(torch.abs(x_centered), 1 / k - 1)
    assert torch.allclose(input.grad, expected_grad)

    # Verify d_quant and q_m still receive gradients
    assert d_quant.grad is not None
    assert q_m.grad is not None


def test_dge_vs_ste():
    """Compare gradient behavior between DGE and STE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup input within quantization range
    x = torch.tensor([0.3], device=device, requires_grad=True)
    d_quant = torch.tensor([0.2], device=device)
    q_m = torch.tensor([1.0], device=device)
    clip_val = torch.tensor([-1.0, 1.0], device=device)
    q_s = torch.tensor([0.0], device=device)
    num_bits = torch.tensor([4.0], device=device)

    # DGE gradient
    output_dge = DGEQuantizer.apply(x, d_quant, q_m, clip_val, q_s, num_bits)
    output_dge.backward()
    dge_grad = x.grad.clone()
    x.grad.zero_()

    # STE gradient
    output_ste = d_quant * torch.round(x / d_quant)  # Basic quantization
    output_ste.backward()
    ste_grad = x.grad

    # Verify:
    # 1. DGE gradient should be different from STE gradient
    # 2. DGE gradient should follow f'(x) = (1/k) · |x - δ/2|^(1/k - 1)
    assert not torch.allclose(dge_grad, ste_grad)

    # Calculate expected DGE gradient
    k = 5.0  # for 4 bits
    x_centered = x - d_quant / 2
    expected_dge_grad = (1 / k) * torch.pow(torch.abs(x_centered), 1 / k - 1)
    assert torch.allclose(dge_grad, expected_dge_grad)


# def test_k_scaling_with_bits():
#    """Test k scales properly with bit width relative to k=5 at 4 bits."""
#    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#    input = torch.tensor([0.3], device=device)
#    d_quant = torch.tensor([0.2], device=device)
#    q_m = torch.tensor([1.0], device=device)
#    clip_val = torch.tensor([-1.0, 1.0], device=device)
#    q_s = torch.tensor([0.0], device=device)

#    bit_widths = [2.0, 4.0, 8.0, 16.0]
#    expected_k = [10.0, 5.0, 2.5, 1.25]  # k = 5 * (4/num_bits)

#    for bits, exp_k in zip(bit_widths, expected_k):
#        ctx = type('MockCtx', (), {'save_for_backward': lambda *args: None})()
#        DGEQuantizer.forward(ctx, input, d_quant, q_m, clip_val, q_s,
#                           torch.tensor([bits], device=device))
#        k = ctx.saved_tensors[3]  # k should be 4th saved tensor
#        assert torch.allclose(k, torch.tensor([exp_k], device=device))
