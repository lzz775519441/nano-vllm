from enum import Enum

from nanovllm.engine.sequence import Sequence


class RuntimeMode(str, Enum):
    NONE = "none"
    FULL_DECODE = "full_decode"
    PIECEWISE = "piecewise"


def eligible_full_decode_graph(seqs: list[Sequence]) -> bool:
    """Matches prepare_decode: all decode steps with one scheduled token per request."""
    return bool(seqs) and all(not seq.is_prefill and seq.num_scheduled_tokens == 1 for seq in seqs)


class CUDAGraphDispatcher:

    def __init__(self, mode: str):
        self.enabled = mode != "none"

    def dispatch(self, seqs: list[Sequence]) -> RuntimeMode:
        if not self.enabled:
            return RuntimeMode.NONE
        if eligible_full_decode_graph(seqs):
            return RuntimeMode.FULL_DECODE
        return RuntimeMode.PIECEWISE
