import os
import torch
from torch import nn
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except ImportError:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
from nanovllm.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def _check_store_kvcache_inputs(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    return N, D


def store_kvcache_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, D = _check_store_kvcache_inputs(key, value, k_cache, v_cache, slot_mapping)
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


def store_kvcache_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Naive PyTorch baseline that preserves the Triton backend's -1 skip semantics."""
    _, D = _check_store_kvcache_inputs(key, value, k_cache, v_cache, slot_mapping)
    mask = slot_mapping != -1
    slots = slot_mapping[mask].long()
    k_cache.view(-1, D).index_copy_(0, slots, key.view(-1, D)[mask])
    v_cache.view(-1, D).index_copy_(0, slots, value.view(-1, D)[mask])


@torch.compile
def _store_kvcache_torch_compile_impl(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N = key.shape[0]
    D = key.shape[1] * key.shape[2]
    slots = slot_mapping.long()
    k_cache.view(-1, D).index_copy_(0, slots, key.view(N, D))
    v_cache.view(-1, D).index_copy_(0, slots, value.view(N, D))


def store_kvcache_torch_compile(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    """Compiled PyTorch baseline for profiling; expects all slots to be valid."""
    _check_store_kvcache_inputs(key, value, k_cache, v_cache, slot_mapping)
    _store_kvcache_torch_compile_impl(key, value, k_cache, v_cache, slot_mapping)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    backend: str | None = None,
):
    backend = backend or os.getenv("NANOVLLM_STORE_KVCACHE_BACKEND", "triton")
    if backend == "triton":
        store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
    elif backend in ("torch", "naive"):
        store_kvcache_torch(key, value, k_cache, v_cache, slot_mapping)
    elif backend in ("torch_compile", "compile", "compiled"):
        store_kvcache_torch_compile(key, value, k_cache, v_cache, slot_mapping)
    else:
        raise ValueError(f"Unknown store_kvcache backend: {backend}")


def _require_flash_attn():
    if flash_attn_varlen_func is None or flash_attn_with_kvcache is None:
        raise ImportError("flash_attn is required for Attention.forward")


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        _require_flash_attn()
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if context.attn_mode == "varlen" and k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.attn_mode == "varlen":
            if context.block_tables is not None:    # paged KV cache
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        elif context.attn_mode == "decode":
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, k.unsqueeze(1), v.unsqueeze(1),
                                        cache_seqlens=context.cache_seqlens, block_table=context.block_tables,
                                        softmax_scale=self.scale, causal=True).squeeze(1)
        else:
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True)
        return o
