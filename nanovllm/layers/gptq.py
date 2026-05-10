import torch
from torch import nn


def get_quant_config(config) -> dict:
    quant_config = getattr(config, "quantization_config", None)
    if isinstance(quant_config, dict):
        return quant_config
    if hasattr(quant_config, "to_dict"):
        return quant_config.to_dict()
    if quant_config is not None:
        return vars(quant_config)
    return {}


def is_gptq_config(config) -> bool:
    return get_quant_config(config).get("quant_method") == "gptq"


def _config_dtype(config) -> torch.dtype:
    dtype = getattr(config, "dtype", None) or getattr(config, "torch_dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        return getattr(torch, dtype.removeprefix("torch."))
    return torch.get_default_dtype()


def _make_backend_linear(backend_name: str, quant_config: dict, config, input_size: int, output_size: int, bias: bool):
    bits = quant_config.get("bits", 4)
    group_size = quant_config.get("group_size", 128)
    desc_act = quant_config.get("desc_act", False)
    sym = quant_config.get("sym", True)
    dtype = _config_dtype(config)

    if backend_name == "machete":
        from gptqmodel.nn_modules.qlinear.machete import MacheteLinear

        linear = MacheteLinear(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            dtype=dtype,
        )
    elif backend_name == "marlin":
        from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

        linear = MarlinLinear(
            bits=bits,
            group_size=group_size,
            desc_act=desc_act,
            sym=sym,
            in_features=input_size,
            out_features=output_size,
            bias=bias,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown GPTQ backend: {backend_name}")

    linear.is_gptq_linear = True
    linear.gptq_backend = backend_name
    linear.gptq_post_init_done = False
    return linear


def GPTQModelLinear(config, input_size: int, output_size: int, bias: bool = True) -> nn.Module:
    quant_config = get_quant_config(config)
    errors = []
    for backend_name in ("machete", "marlin"):
        try:
            return _make_backend_linear(backend_name, quant_config, config, input_size, output_size, bias)
        except Exception as exc:
            errors.append(f"{backend_name}: {exc}")
    message = "\n".join(errors)
    raise RuntimeError(
        "No fused GPTQModel backend is available. Install GPTQModel with a CUDA fused backend; "
        "nano-vLLM will not fall back to bf16/fp16 dequantized weights.\n"
        f"{message}"
    )


def post_init_gptq_modules(model: nn.Module):
    for module in model.modules():
        if not getattr(module, "is_gptq_linear", False):
            continue
        if getattr(module, "gptq_post_init_done", False):
            continue
        post_init = getattr(module, "post_init", None)
        if post_init is not None:
            post_init()
        module.gptq_post_init_done = True
