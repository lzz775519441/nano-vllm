import os
import random

import torch

try:
    from nanovllm import _C
except ImportError:
    _C = None


HAS_CUDA_OPS = _C is not None


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in ("0", "false", "off", "no")


USE_CUDA_SAMPLE = _env_flag("NANOVLLM_CUDA_SAMPLE")
USE_CUDA_RMSNORM = _env_flag("NANOVLLM_CUDA_RMSNORM")
USE_CUDA_ROPE = _env_flag("NANOVLLM_CUDA_ROPE")
USE_CUDA_KVSTORE = _env_flag("NANOVLLM_CUDA_KVSTORE")
USE_STRIDE_AWARE_KERNELS = _env_flag("NANOVLLM_CUDA_STRIDE_AWARE", False)

# The current native kernels assume contiguous tensor layouts.  Keep this off
# by default so the wrapper never hides an expensive copy behind a fast-looking
# custom op call.  Turn it on only for A/B testing the old behavior.
ALLOW_CONTIGUOUS_COPY = _env_flag("NANOVLLM_CUDA_ALLOW_CONTIGUOUS_COPY", False)


def _usable_cuda(*tensors: torch.Tensor) -> bool:
    return HAS_CUDA_OPS and all(t.is_cuda for t in tensors)


def _contiguous_or_none(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...] | None:
    if not _usable_cuda(*tensors):
        return None
    if all(t.is_contiguous() for t in tensors):
        return tensors
    if ALLOW_CONTIGUOUS_COPY:
        return tuple(t.contiguous() for t in tensors)
    return None


def _native_layout_ready(*tensors: torch.Tensor) -> bool:
    return _usable_cuda(*tensors) and (USE_STRIDE_AWARE_KERNELS or all(t.is_contiguous() for t in tensors))


def _stride3(tensor: torch.Tensor) -> tuple[int, int, int]:
    return tensor.stride(0), tensor.stride(1), tensor.stride(2)


def _next_seed() -> int:
    return random.getrandbits(63)


def sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    if USE_CUDA_SAMPLE:
        native_tensors = _contiguous_or_none(logits, temperatures)
        if native_tensors is not None:
            logits, temperatures = native_tensors
            return _C.sample(logits, temperatures, _next_seed())
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if USE_CUDA_RMSNORM:
        native_tensors = _contiguous_or_none(x, weight)
        if native_tensors is not None:
            x, weight = native_tensors
            return _C.rms_norm(x, weight, eps)
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return x.to(orig_dtype).mul_(weight)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if USE_CUDA_RMSNORM:
        native_tensors = _contiguous_or_none(x, residual, weight)
        if native_tensors is not None:
            x, residual, weight = native_tensors
            out, new_residual = _C.add_rms_norm(x, residual, weight, eps)
            return out, new_residual
    orig_dtype = x.dtype
    x = x.float().add_(residual.float())
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return x.to(orig_dtype).mul_(weight), residual


def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if USE_CUDA_ROPE and _usable_cuda(positions, query, key, cos_sin_cache):
        if positions.is_contiguous() and cos_sin_cache.is_contiguous() and _native_layout_ready(query, key):
            return _C.rotary_embedding(
                positions,
                query,
                key,
                cos_sin_cache,
                *_stride3(query),
                *_stride3(key),
            )
        if ALLOW_CONTIGUOUS_COPY:
            positions = positions.contiguous()
            query = query.contiguous()
            key = key.contiguous()
            cos_sin_cache = cos_sin_cache.contiguous()
            return _C.rotary_embedding(
                positions,
                query,
                key,
                cos_sin_cache,
                *_stride3(query),
                *_stride3(key),
            )

    cos_sin = cos_sin_cache[positions]
    cos, sin = cos_sin.chunk(2, dim=-1)
    q1, q2 = torch.chunk(query.float(), 2, dim=-1)
    k1, k2 = torch.chunk(key.float(), 2, dim=-1)
    query = torch.cat((q1 * cos - q2 * sin, q2 * cos + q1 * sin), dim=-1).to(query.dtype)
    key = torch.cat((k1 * cos - k2 * sin, k2 * cos + k1 * sin), dim=-1).to(key.dtype)
    return query, key


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    if USE_CUDA_KVSTORE and _usable_cuda(key, value, k_cache, v_cache, slot_mapping):
        _C.store_kvcache(key, value, k_cache, v_cache, slot_mapping)
        return

    valid = slot_mapping >= 0
    if not valid.any():
        return
    slots = slot_mapping[valid].long()
    k_cache.view(-1, key.size(1), key.size(2))[slots] = key[valid]
    v_cache.view(-1, value.size(1), value.size(2))[slots] = value[valid]
