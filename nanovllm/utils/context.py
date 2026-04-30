from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    attn_mode: str | None = None
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    cache_seqlens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    attn_mode=None,
    is_prefill=False,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    cache_seqlens=None,
    block_tables=None,
):
    global _CONTEXT
    _CONTEXT = Context(
        attn_mode=attn_mode,
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        cache_seqlens=cache_seqlens,
        block_tables=block_tables,
    )

def set_varlen_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables=None):
    set_context(
        attn_mode="varlen",
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        block_tables=block_tables,
    )

def set_decode_context(slot_mapping, cache_seqlens, block_tables):
    set_context(
        attn_mode="decode",
        slot_mapping=slot_mapping,
        cache_seqlens=cache_seqlens,
        block_tables=block_tables,
    )

def set_context_obj(context: Context):
    global _CONTEXT
    _CONTEXT = context

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
