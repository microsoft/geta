import logging
import math
from enum import Enum
from typing import Union

import torch
from torch import nn


class NanInGradientError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


# submodule level logger
logger = logging.getLogger(__name__)


class QuantizationType(Enum):
    # TODO: make uniform naming convention for the enum values
    SYMMETRIC_LINEAR = "symmetric+linear"
    SYMMETRIC_NONLINEAR = "symmetric+nonlinear"
    DGE = "dge"


class QuantizationMode(Enum):
    WEIGHT_ONLY = "weight_only"
    WEIGHT_AND_ACTIVATION = "weight_and_activation"


# Symmetric quantizer with nonlinear mapping (yes t)
class SymQuantizerNonLinear(torch.autograd.Function):
    """Symmetric quantizer with nonlinear mapping.

    Implements quantization using nonlinear mapping with parameter t.
    Uses gradient computation rules from: https://arxiv.org/pdf/2007.10463
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        d_quant: torch.Tensor,
        q_m: torch.Tensor,
        t_quant: torch.Tensor,
        clip_val: torch.Tensor,
        q_s: torch.Tensor,
    ) -> torch.Tensor:
        input_abs = torch.abs(input)
        device = input.device

        # Ensure all tensors are on the same device
        d_quant = d_quant.to(device)
        q_m = q_m.to(device)
        t_quant = t_quant.to(device)
        clip_val = clip_val.to(device)
        q_s = q_s.to(device)
        ctx.save_for_backward(input, d_quant, q_m, t_quant, clip_val, q_s)

        # q_m <= q_s can happen
        range_pow = torch.exp(t_quant * torch.log(torch.abs(q_m - q_s) + 1e-6))
        input_pow = torch.exp(t_quant * torch.log(input_abs - q_s))  # input_abs >= q_s

        output = d_quant * torch.round(input_pow.div(d_quant))
        output[input_abs <= q_s] = 0
        output[input_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))
        output = torch.sign(input) * output
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor]:
        input, d_quant, q_m, t_quant, clip_val, q_s = ctx.saved_tensors
        device = input.device
        input_abs = torch.abs(input)

        grad_x = grad_output.clone()
        grad_x[input.ge(clip_val[1])] = 0
        grad_x[input.le(clip_val[0])] = 0
        # Useful quantities
        range_pow = torch.exp(
            t_quant * torch.log(torch.abs(q_m - q_s) + 1e-6)
        )  # q_m <= q_s can happen
        range_pow_low = torch.exp(
            (t_quant - 1) * torch.log(torch.abs(q_m - q_s) + 1e-6)
        )  # q_m <= q_s can happen
        input_pow = torch.exp(t_quant * torch.log(input_abs - q_s))  # input_abs >= q_s

        grad_d_xq = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
        grad_d_xq[input_abs >= q_m] = torch.round(
            range_pow.div(d_quant)
        ) - range_pow.div(d_quant)
        grad_d_xq[input_abs <= q_s] = 0
        grad_d_xq = torch.sign(input) * grad_d_xq
        grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

        grad_qm_xq = torch.sign(input) * ((t_quant * range_pow_low).expand_as(input))
        grad_qm_xq[input_abs <= q_m] = 0
        grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

        grad_t_xq = input_pow * (torch.log(input_abs - q_s))
        grad_t_xq[input_abs >= q_m] = range_pow * torch.log(torch.abs(q_m - q_s) + 1e-6)
        grad_t_xq[input_abs <= q_s] = 0
        grad_t_xq = torch.sign(input) * grad_t_xq
        grad_t = torch.tensor([torch.sum(grad_output * grad_t_xq)], device=device)

        # NaN detection
        if torch.allclose(
            torch.tensor([torch.sum(grad_output * grad_t_xq)], device=device),
            torch.tensor([float("nan")], device=device),
            equal_nan=True,
        ):
            error_message = (
                f"Error: NaN appears in gradient!\n"
                f"d: {d_quant.item():.5f}, t: {t_quant.item():.5f}, q_m: {q_m.item():.5f}, q_s: {q_s.item():.5f}\n"
                f"input_abs-max: {torch.max(input_abs).item():.5f}, grad_output-max: {torch.max(grad_output).item():.5f}, grad_t_xq: {torch.max(grad_t_xq)}\n"
                f"input_pow: {torch.min(input_pow)}, range_pow: {range_pow}, input_abs-min: {torch.min(input_abs)}\n"
                f"grad_x: min={torch.min(grad_x):.5f}, max={torch.max(grad_x):.5f}, mean={torch.mean(grad_x):.5f}, std={torch.std(grad_x):.5f}\n"
                f"grad_d: {grad_d.item():.5f}\n"
                f"grad_qm: {grad_qm.item():.5f}\n"
                f"grad_t: {grad_t.item():.5f}"
            )
            raise NanInGradientError(error_message)

        return grad_x, grad_d, grad_qm, grad_t, None, None


class SymQuantizerLinear(torch.autograd.Function):
    """Symmetric quantizer with linear mapping.

    Implements quantization using linear mapping without nonlinearity parameter t.
    Uses gradient computation rules from: https://arxiv.org/pdf/2007.10463
    """

    # TODO: Add safeguard mechanism to handle the bad cases.
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        d_quant: torch.Tensor,
        q_m: torch.Tensor,
        clip_val: torch.Tensor,
        q_s: torch.Tensor,
    ) -> torch.Tensor:
        device = input.device
        input_abs = torch.abs(input)
        # Ensure all tensors are on the same device
        d_quant = d_quant.to(device)
        q_m = q_m.to(device)
        clip_val = clip_val.to(device)
        q_s = q_s.to(device)
        ctx.save_for_backward(input, d_quant, q_m, clip_val, q_s)

        range_pow = torch.abs(q_m - q_s)
        input_pow = input_abs - q_s  # input_abs >= q_s

        output = d_quant * torch.round(input_pow.div(d_quant))
        output[input_abs <= q_s] = 0
        output[input_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))
        output = torch.sign(input) * output
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor]:
        input, d_quant, q_m, clip_val, q_s = ctx.saved_tensors
        device = input.device
        input_abs = torch.abs(input)

        grad_x = grad_output.clone()
        grad_x[input.ge(clip_val[1])] = 0
        grad_x[input.le(clip_val[0])] = 0

        # Useful quantities
        range_pow = torch.abs(q_m - q_s)  # q_m <= q_s can happen
        input_pow = input_abs - q_s  # input_abs >= q_s

        grad_d_xq = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
        grad_d_xq[input_abs >= q_m] = torch.round(
            range_pow.div(d_quant)
        ) - range_pow.div(d_quant)
        grad_d_xq[input_abs <= q_s] = 0
        grad_d_xq = torch.sign(input) * grad_d_xq
        grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

        grad_qm_xq = torch.sign(input)
        grad_qm_xq[input_abs <= q_m] = 0
        grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

        # NaN detection
        if torch.allclose(
            torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device),
            torch.tensor([float("nan")], device=device),
            equal_nan=True,
        ):
            error_message = (
                f"Error: NaN appears in gradient!\n"
                f"d: {d_quant.item():.5f}, q_m: {q_m.item():.5f}, q_s: {q_s.item():.5f}\n"
                f"input_abs-max: {torch.max(input_abs).item():.5f}, grad_output-max: {torch.max(grad_output).item():.5f},\n"
                f"input_pow: {torch.min(input_pow)}, range_pow: {range_pow}, input_abs-min: {torch.min(input_abs)}\n"
                f"grad_x: min={torch.min(grad_x):.5f}, max={torch.max(grad_x):.5f}, mean={torch.mean(grad_x):.5f}, std={torch.std(grad_x):.5f}\n"
                f"grad_d: {grad_d.item():.5f}\n"
                f"grad_qm: {grad_qm.item():.5f}\n"
            )
            raise NanInGradientError(error_message)
        return grad_x, grad_d, grad_qm, None, None


# NOTE: this is a experimental WIP, not used in the codebase yet
class DGEQuantizer(torch.autograd.Function):
    """DGE quantizer for weights that replaces STE with differentiable gradient estimation.
    https://arxiv.org/pdf/2501.17116
    Forward: Normal quantization
    Backward: Uses f'(x) = (1/k) · |x - δ/2|^(1/k - 1) for weight gradient computation
    where k scales based on bit width relative to paper's k=5 for 4-bit.
    """

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        d_quant: torch.Tensor,  # Quantization interval (δ)
        q_m: torch.Tensor,
        clip_val: torch.Tensor,
        q_s: torch.Tensor,
        num_bits: torch.Tensor,  # Current target bit width
    ) -> torch.Tensor:
        device = input.device
        input_abs = torch.abs(input)

        # Move tensors to device
        d_quant = d_quant.to(device)
        q_m = q_m.to(device)
        clip_val = clip_val.to(device)
        q_s = q_s.to(device)

        # Scale k relative to paper's k=5 for 4-bit
        k = torch.tensor(5.0 * (4.0 / num_bits)).to(device)
        ctx.save_for_backward(input, d_quant, q_m, k, clip_val, q_s)

        range_pow = torch.abs(q_m - q_s)
        input_pow = input_abs - q_s  # input_abs >= q_s

        output = d_quant * torch.round(input_pow.div(d_quant))
        output[input_abs <= q_s] = 0
        output[input_abs >= q_m] = d_quant * torch.round(range_pow.div(d_quant))
        output = torch.sign(input) * output
        return output

    @staticmethod
    def backward(ctx, grad_output) -> tuple[torch.Tensor]:
        input, d_quant, q_m, k, clip_val, q_s = ctx.saved_tensors
        device = input.device
        input_abs = torch.abs(input)

        # Zero gradients outside clipping range
        grad_x = grad_output.clone()
        grad_x[input.ge(clip_val[1])] = 0
        grad_x[input.le(clip_val[0])] = 0

        # Paper's DGE gradient computation: f'(x) = (1/k) · |x - δ/2|^(1/k - 1)
        x_centered = input - d_quant / 2
        grad_scale = (1 / k) * torch.pow(torch.abs(x_centered), 1 / k - 1)
        grad_x = grad_x * grad_scale
        # TODO: do we modify if we have higher than 4 bits?
        # Cap gradient magnitude at 3.0 as in paper
        grad_x = torch.clamp(grad_x, -3.0, 3.0)

        # Compute d_quant gradient
        range_pow = torch.abs(q_m - q_s)
        input_pow = input_abs - q_s
        grad_d_xq = torch.round(input_pow.div(d_quant)) - input_pow.div(d_quant)
        grad_d_xq[input_abs >= q_m] = torch.round(
            range_pow.div(d_quant)
        ) - range_pow.div(d_quant)
        grad_d_xq[input_abs <= q_s] = 0
        grad_d_xq = torch.sign(input) * grad_d_xq
        grad_d = torch.tensor([torch.sum(grad_output * grad_d_xq)], device=device)

        # Compute q_m gradient
        grad_qm_xq = torch.sign(input)
        grad_qm_xq[input_abs <= q_m] = 0
        grad_qm = torch.tensor([torch.sum(grad_output * grad_qm_xq)], device=device)

        if torch.isnan(grad_x).any():
            raise NanInGradientError(
                f"NaN in gradient computation\n"
                f"input range: [{input.min():.4f}, {input.max():.4f}]\n"
                f"d_quant: {d_quant.item():.4f}\n"
                f"q_m: {q_m.item():.4f}\n"
                f"k: {k.item():.4f}"
            )

        return grad_x, grad_d, grad_qm, None, None, None


def _get_quantizer(qtype: QuantizationType) -> torch.autograd.Function:
    if qtype == QuantizationType.SYMMETRIC_LINEAR:
        return SymQuantizerLinear
    elif qtype == QuantizationType.SYMMETRIC_NONLINEAR:
        return SymQuantizerNonLinear
    elif qtype == QuantizationType.DGE:
        return DGEQuantizer
    else:
        raise NotImplementedError


class QuantizeMixin:
    def init_quantization(
        self,
        d_quant_init: float = 1.0,
        t_quant_init: float = 1.0,
        q_m_init: float = 1.0,
        quant_type: QuantizationType = QuantizationType.SYMMETRIC_LINEAR,
        quant_mode: QuantizationMode = QuantizationMode.WEIGHT_ONLY,
        weight_clip_val: tuple[float, float] = (-2.0, 2.0),
        act_clip_val: tuple[float, float] = (-2.0, 2.0),
    ):
        # Initialize weight quantization parameters
        self.d_quant_wt = nn.Parameter(torch.tensor([d_quant_init]))
        self.q_m_wt = nn.Parameter(torch.tensor([q_m_init]))
        if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            self.t_quant_wt = nn.Parameter(torch.tensor([t_quant_init]))

        # Initialize activation quantization if needed
        if quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
            self.d_quant_act = nn.Parameter(torch.tensor([d_quant_init]))
            self.q_m_act = nn.Parameter(torch.tensor([q_m_init]))
            if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
                self.t_quant_act = nn.Parameter(torch.tensor([t_quant_init]))

        self.quant_type = quant_type
        self.quant_mode = quant_mode
        self.weight_clip_val = weight_clip_val
        self.act_clip_val = act_clip_val

    def quantize_weight(self, weight: torch.tensor) -> torch.Tensor:
        """Quantize the weight tensor."""
        weight_clip_val = torch.tensor(self.weight_clip_val, device=weight.device)
        q_s = torch.tensor(0.0, device=weight.device)
        quantizer = _get_quantizer(self.quant_type)
        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            # Symmetric linear
            return quantizer.apply(
                weight,
                self.d_quant_wt,
                self.q_m_wt,
                weight_clip_val,
                q_s,
            )
        else:  # Symmetric nonlinear
            return quantizer.apply(
                weight,
                self.d_quant_wt,
                self.q_m_wt,
                self.t_quant_wt,
                weight_clip_val,
                q_s,
            )

    def quantize_act(self, activation: torch.tensor) -> torch.Tensor:
        """Quantize the activation tensor."""
        if self.quant_mode != QuantizationMode.WEIGHT_AND_ACTIVATION:
            return activation

        activation_clip_val = torch.tensor(self.act_clip_val, device=activation.device)
        q_s = torch.tensor(0.0, device=activation.device)
        quantizer = _get_quantizer(self.quant_type)

        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            return quantizer.apply(
                activation,
                self.d_quant_act,
                self.q_m_act,
                activation_clip_val,
                q_s,
            )
        else:  # SYMMETRIC_NONLINEAR
            return quantizer.apply(
                activation,
                self.d_quant_act,
                self.q_m_act,
                self.t_quant_act,
                activation_clip_val,
                q_s,
            )

    @property
    def weight_bit(self) -> int:
        d = self.d_quant_wt.item()
        qmax = abs(self.q_m_wt.item())
        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            t = 1.0
        elif self.quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            t = self.t_quant_wt.item()
        else:
            raise NotImplementedError

        return round(math.log2(math.exp(t * math.log(qmax)) / abs(d) + 1) + 1)

    @property
    def activation_bit(self) -> int:
        if self.quant_mode != QuantizationMode.WEIGHT_AND_ACTIVATION:
            return 32

        d = self.d_quant_act.item()
        qmax = abs(self.q_m_act.item())
        if self.quant_type == QuantizationType.SYMMETRIC_LINEAR:
            t = 1.0
        elif self.quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            t = self.t_quant_act.item()
        else:
            raise NotImplementedError

        return round(math.log2(math.exp(t * math.log(qmax)) / abs(d) + 1) + 1)


def initialize_quant_layer(
    layer: Union["QuantizeConv2d", "QuantizeLinear"],
    num_bits: int = 16,
    quant_type: QuantizationType = QuantizationType.SYMMETRIC_LINEAR,
    quant_mode: QuantizationMode = QuantizationMode.WEIGHT_ONLY,
) -> None:
    if not isinstance(layer, (QuantizeConv2d, QuantizeLinear)):
        return

    num_bits = float(num_bits)
    t_quant_init = 1.0
    q_s = torch.tensor(0.0, device=layer.weight.device)

    qm_quant_init = torch.max(torch.abs(layer.weight))
    d_quant_init = (qm_quant_init - q_s) / (2 ** (num_bits - 1) - 1)

    # Weight quantization parameters
    nn.init.constant_(layer.d_quant_wt, d_quant_init)
    nn.init.constant_(layer.q_m_wt, qm_quant_init)
    if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
        nn.init.constant_(layer.t_quant_wt, t_quant_init)

    # Activation quantization parameters if needed
    if quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
        nn.init.constant_(layer.d_quant_act, d_quant_init)
        nn.init.constant_(layer.q_m_act, qm_quant_init)
        if quant_type == QuantizationType.SYMMETRIC_NONLINEAR:
            nn.init.constant_(layer.t_quant_act, t_quant_init)


class QuantizeLinear(nn.Linear, QuantizeMixin):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias)
        self.init_quantization(
            d_quant_init, t_quant_init, q_m_init, quant_type, quant_mode
        )

    @staticmethod
    def from_module(
        module=None,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
        quant_init_by_module=True,
        num_bits=8,
    ):
        quant_linear = QuantizeLinear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=True if module.bias is not None else False,
            d_quant_init=d_quant_init,
            t_quant_init=t_quant_init,
            q_m_init=q_m_init,
            quant_type=quant_type,
            quant_mode=quant_mode,
        )

        quant_linear.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            quant_linear.bias.data.copy_(module.bias.data)

        if quant_init_by_module:
            initialize_quant_layer(
                quant_linear,
                num_bits=num_bits,
                quant_type=quant_type,
                quant_mode=quant_mode,
            )
        return quant_linear

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.quantize_weight(self.weight)
        if self.quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
            input_ = self.quantize_act(input_)
        return nn.functional.linear(input_, weight, self.bias)


class QuantizeConv2d(nn.Conv2d, QuantizeMixin):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
    ):
        nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        )
        self.init_quantization(
            d_quant_init, t_quant_init, q_m_init, quant_type, quant_mode
        )

    @staticmethod
    def from_module(
        module=None,
        d_quant_init=1.0,
        t_quant_init=1.0,
        q_m_init=1.0,
        quant_type=QuantizationType.SYMMETRIC_LINEAR,
        quant_mode=QuantizationMode.WEIGHT_ONLY,
        quant_init_by_module=True,
        num_bits=8,
    ):
        # Fixed argument names in constructor call
        quant_conv2d = QuantizeConv2d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=True if module.bias is not None else False,
            d_quant_init=d_quant_init,
            t_quant_init=t_quant_init,
            q_m_init=q_m_init,
            quant_type=quant_type,
            quant_mode=quant_mode,
        )

        quant_conv2d.weight.data.copy_(module.weight.data)
        if module.bias is not None:
            quant_conv2d.bias.data.copy_(module.bias.data)

        if quant_init_by_module:
            initialize_quant_layer(
                quant_conv2d,
                num_bits=num_bits,
                quant_type=quant_type,
                quant_mode=quant_mode,
            )
        return quant_conv2d

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        weight = self.quantize_weight(self.weight)
        if self.quant_mode == QuantizationMode.WEIGHT_AND_ACTIVATION:
            input_ = self.quantize_act(input_)
        return nn.functional.conv2d(
            input_,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


LAYER_TO_QUANTLAYER = {"Linear": QuantizeLinear, "Conv2d": QuantizeConv2d}
