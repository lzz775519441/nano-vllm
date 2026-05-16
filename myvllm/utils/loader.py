import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter | torch.Tensor,
                          loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def post_init_gptq_modules(model: nn.Module):
    from gptqmodel.nn_modules.qlinear.marlin import MarlinLinear

    for module in model.modules():
        if isinstance(module, MarlinLinear):
            module.post_init()


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    params = dict(model.named_parameters())
    params.update(dict(model.named_buffers()))

    for file in sorted(glob(os.path.join(path, "*.safetensors"))):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k, (v, shard_id) in packed_modules_mapping.items():
                    if k in weight_name:
                        param_name = weight_name.replace(k, v)
                        param = params[param_name]
                        param.weight_loader(param, f.get_tensor(weight_name),
                                            shard_id)
                        break
                else:
                    param = params[weight_name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))

    post_init_gptq_modules(model)
