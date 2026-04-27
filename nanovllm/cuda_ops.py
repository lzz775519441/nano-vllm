import random
import torch
import nanovllm._C

@torch.library.register_fake("nanovllm::sample")
def _sample_fake(logits, temperatures, seed):
    return torch.empty(logits.size(0), dtype=torch.int64, device=logits.device)

@torch.library.register_fake("nanovllm::rms_norm")
def _rms_norm_fake(x, weight, eps):
    return torch.empty_like(x)

@torch.library.register_fake("nanovllm::add_rms_norm")
def _add_rms_norm_fake(x, residual, weight, eps):
    return torch.empty_like(x), torch.empty_like(residual)

@torch.library.register_fake("nanovllm::rotary_embedding")
def _rotary_embedding_fake(positions, query, key, cos_sin_cache):
    return torch.empty_like(query), torch.empty_like(key)

@torch.library.register_fake("nanovllm::store_kvcache")
def _store_kvcache_fake(key, value, k_cache, v_cache, slot_mapping):
    pass


def sample(logits: torch.Tensor, temperatures: torch.Tensor) -> torch.Tensor:
    seed = random.getrandbits(63)
    return torch.ops.nanovllm.sample(
        logits.contiguous(), temperatures.contiguous(), seed
    )

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.ops.nanovllm.rms_norm(
        x.contiguous(), weight.contiguous(), eps
    )

def add_rms_norm(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.nanovllm.add_rms_norm(
        x.contiguous(), residual.contiguous(), weight.contiguous(), eps
    )

def rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.nanovllm.rotary_embedding(
        positions,
        query,
        key,
        cos_sin_cache,
    )

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    torch.ops.nanovllm.store_kvcache(
        key.contiguous(),
        value.contiguous(),
        k_cache,
        v_cache,
        slot_mapping.contiguous(),
    )