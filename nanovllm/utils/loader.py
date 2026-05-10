import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open

from nanovllm.layers.gptq import post_init_gptq_modules


def default_weight_loader(param: nn.Parameter | torch.Tensor, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def get_parameter_or_buffer(model: nn.Module, name: str) -> nn.Parameter | torch.Tensor:
    try:
        return model.get_parameter(name)
    except AttributeError:
        return model.get_buffer(name)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = get_parameter_or_buffer(model, param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = get_parameter_or_buffer(model, weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
    post_init_gptq_modules(model)
