from dataclasses import dataclass
from enum import Enum

from nanovllm.engine.sequence import Sequence


class CUDAGraphMode(str, Enum):
    NONE = "none"
    FULL_DECODE_ONLY = "full_decode_only"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"


class RuntimeMode(str, Enum):
    NONE = "none"
    FULL_DECODE = "full_decode"
    PIECEWISE = "piecewise"


@dataclass(frozen=True, slots=True)
class BatchDescriptor:
    num_seqs: int
    num_batched_tokens: int
    is_uniform_decode: bool
    max_query_len: int
    max_kv_len: int

    @classmethod
    def from_sequences(cls, seqs: list[Sequence]):
        num_batched_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
        is_uniform_decode = all(not seq.is_prefill and seq.num_scheduled_tokens == 1 for seq in seqs)
        max_query_len = max(seq.num_scheduled_tokens for seq in seqs)
        max_kv_len = max(seq.num_cached_tokens + seq.num_scheduled_tokens for seq in seqs)
        return cls(len(seqs), num_batched_tokens, is_uniform_decode, max_query_len, max_kv_len)


class CUDAGraphDispatcher:

    def __init__(self, mode: str):
        self.mode = CUDAGraphMode(mode)

    def dispatch(self, descriptor: BatchDescriptor) -> RuntimeMode:
        if self.mode == CUDAGraphMode.NONE:
            return RuntimeMode.NONE
        if descriptor.is_uniform_decode:
            if self.mode in (CUDAGraphMode.FULL_DECODE_ONLY, CUDAGraphMode.FULL_AND_PIECEWISE):
                return RuntimeMode.FULL_DECODE
            if self.mode == CUDAGraphMode.PIECEWISE:
                return RuntimeMode.PIECEWISE
        if self.mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL_AND_PIECEWISE):
            return RuntimeMode.PIECEWISE
        return RuntimeMode.NONE
