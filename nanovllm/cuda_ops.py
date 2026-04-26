import torch

try:
    from nanovllm import _C
except ImportError:
    _C = None


HAS_CUDA_OPS = _C is not None


def _usable(*tensors: torch.Tensor) -> bool:
    return HAS_CUDA_OPS and all(t.is_cuda for t in tensors)


def sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    if _usable(logits, temperatures):
        seed = torch.empty((), dtype=torch.int64, device="cpu").random_().item()
        return _C.sample(logits.contiguous(), temperatures.contiguous(), seed)
    logits = logits.float().div_(temperatures.unsqueeze(dim=1))
    probs = torch.softmax(logits, dim=-1)
    return probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    if _usable(x, weight):
        return _C.rms_norm(x.contiguous(), weight.contiguous(), eps)
    orig_dtype = x.dtype
    x = x.float()
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return x.to(orig_dtype).mul_(weight)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    if _usable(x, residual, weight):
        out, new_residual = _C.add_rms_norm(x.contiguous(), residual.contiguous(), weight.contiguous(), eps)
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
    if _usable(positions, query, key, cos_sin_cache):
        q, k = _C.rotary_embedding(
            positions.contiguous(),
            query.contiguous(),
            key.contiguous(),
            cos_sin_cache.contiguous(),
        )
        return q, k

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
    if _usable(key, value, k_cache, v_cache, slot_mapping):
        _C.store_kvcache(
            key.contiguous(),
            value.contiguous(),
            k_cache,
            v_cache,
            slot_mapping.contiguous(),
        )
        return

    valid = slot_mapping >= 0
    if not valid.any():
        return
    slots = slot_mapping[valid].long()
    k_cache.view(-1, key.size(1), key.size(2))[slots] = key[valid]
    v_cache.view(-1, value.size(1), value.size(2))[slots] = value[valid]
