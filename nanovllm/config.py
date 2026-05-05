import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_chunked_prefill_tokens: int = 2048
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    cudagraph_mode: str = "full_and_piecewise"
    max_decode_cudagraph_tokens: int = 512
    max_piecewise_cudagraph_tokens: int = 320
    hf_config: Any | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.max_chunked_prefill_tokens >= 0
        assert self.cudagraph_mode in {"none", "full_decode_only", "piecewise", "full_and_piecewise"}
        assert self.max_decode_cudagraph_tokens >= 0
        assert self.max_piecewise_cudagraph_tokens >= 0
        from transformers import AutoConfig
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
