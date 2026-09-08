import logging
import math

from torch import nn

from .quant_layers import LAYER_TO_QUANTLAYER, QuantizationMode, QuantizationType

"""
The model_to_quantize_model supports 
    - both weight and activation quantization
    - both linear quantization (no t) and nonlinear quantization (yes t)
"""


def model_to_quantize_model(
    model: nn.Module,
    d_quant_init: float = 1e-4,
    t_quant_init: float = 1.0,
    q_m_init: float = 1.0,
    quant_init_by_module: bool = True,
    num_bits: int = 16,
    quant_type: QuantizationType | str = QuantizationType.SYMMETRIC_NONLINEAR,
    quant_mode: QuantizationMode | str = QuantizationMode.WEIGHT_ONLY,
) -> nn.Module:
    """Convert model layers to quantized versions.

    Args:
        model: PyTorch model to convert
        d_quant_init: Initial quantization step size
        t_quant_init: Initial nonlinearity parameter
        q_m_init: Initial quantization range
        quant_init_by_module: Whether to initialize quantization parameters from module weights
        num_bits: Number of bits for quantization
        quant_type: Type of quantization (linear or nonlinear)
        quant_mode: Mode of quantization (weight-only or weight+activation)

    Returns:
        Modified model with quantized layers.
    """

    def _get_submodules(model, key):
        parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name_in_parent_module = key.split(".")[-1]
        target_module = model.get_submodule(key)
        return parent_module, target_module, target_name_in_parent_module

    logger = logging.getLogger(__name__)
    if isinstance(quant_type, str):
        try:
            quant_type = QuantizationType(quant_type)
        except ValueError:
            raise ValueError(
                f"Invalid quantization type: {quant_type}. Must be one of {[t.value for t in QuantizationType]}"
            )

    if isinstance(quant_mode, str):
        try:
            quant_mode = QuantizationMode(quant_mode)
        except ValueError:
            raise ValueError(
                f"Invalid quantization mode: {quant_mode}. Must be one of {[m.value for m in QuantizationMode]}"
            )

    converted_layers = 0
    for name, module in model.named_modules():
        if type(module).__name__ in LAYER_TO_QUANTLAYER:
            parent_module, target_module, target_name = _get_submodules(model, name)
            quant_module = LAYER_TO_QUANTLAYER[type(module).__name__].from_module(
                module=target_module,
                d_quant_init=d_quant_init,
                t_quant_init=t_quant_init,
                q_m_init=q_m_init,
                quant_type=quant_type,
                quant_mode=quant_mode,
                quant_init_by_module=quant_init_by_module,
                num_bits=num_bits,
            )
            setattr(parent_module, target_name, quant_module)
            converted_layers += 1

    logger.info(f"Converted {converted_layers} layers to quantized versions")
    return model


def get_quant_param_dict(model: nn.Module) -> dict[str, dict[str, float]]:
    """Extract quantization parameters from a model.

    Args:
        model: PyTorch model with quantized layers
    Returns:
        Dictionary mapping layer names to their quantization parameters
    """
    param_dict = {}
    for name, param in model.named_parameters():
        if any(qtype in name for qtype in ["d_quant", "t_quant", "q_m"]):
            layer_name = ".".join(name.split(".")[:-1])
            param_name = name.split(".")[-1]
            if layer_name not in param_dict:
                param_dict[layer_name] = {}
            param_dict[layer_name][param_name] = param.item()
    return param_dict


def get_bitwidth_dict(
    param_dict: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Calculate bitwidths for weights and activations of each layer.

    Args:
        param_dict: Dictionary mapping layer names to their quantization parameters

    Returns:
        Dictionary mapping layer names to their calculated bitwidths for weights and activations
    """
    bit_dict: dict[str, dict[str, float]] = {}

    def _calculate_bitwidth(d_quant: float, q_m: float, t_quant: float = 1.0) -> float:
        return math.log2(math.exp(t_quant * math.log(abs(q_m))) / abs(d_quant) + 1) + 1

    for key, params in param_dict.items():
        bit_dict[key] = {}
        # weight bitwidths
        d_quant_wt = params["d_quant_wt"]
        q_m_wt = abs(params["q_m_wt"])
        t_quant_wt = params.get("t_quant_wt", 1.0)
        bit_width_wt = _calculate_bitwidth(d_quant_wt, q_m_wt, t_quant_wt)
        bit_dict[key]["weight"] = bit_width_wt
        # activation bitwidths
        if "d_quant_act" in params:
            d_quant_act = params["d_quant_act"]
            q_m_act = abs(params["q_m_act"])
            t_quant_act = params.get("t_quant_act", 1.0)
            bit_width_act = _calculate_bitwidth(d_quant_act, q_m_act, t_quant_act)
            bit_dict[key]["activation"] = bit_width_act

    return bit_dict
