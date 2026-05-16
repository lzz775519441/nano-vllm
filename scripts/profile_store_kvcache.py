#!/usr/bin/env python3
import argparse
import math
import sys
from pathlib import Path
from typing import Callable

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from myvllm.layers.attention import (
    store_kvcache_torch,
    store_kvcache_torch_compile,
    store_kvcache_triton,
)


BACKENDS: dict[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]] = {
    "torch": store_kvcache_torch,
    "naive": store_kvcache_torch,
    "torch_compile": store_kvcache_torch_compile,
    "triton": store_kvcache_triton,
}


DTYPES = {
    "fp16": torch.float16,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Profile myvllm store_kvcache backends.")
    parser.add_argument("--backend", choices=sorted(BACKENDS), required=True)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--num-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=0)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slot-order", choices=("sequential", "random"), default="sequential")
    parser.add_argument("--no-check", action="store_true")
    parser.add_argument("--no-cuda-profiler-range", action="store_true")
    return parser.parse_args()


def make_slot_mapping(args, total_slots: int, device: torch.device):
    if args.slot_order == "sequential":
        return torch.arange(args.tokens, device=device, dtype=torch.int32)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    return torch.randperm(total_slots, device=device, generator=generator, dtype=torch.int64)[:args.tokens].to(torch.int32)


def check_correctness(fn, key, value, k_cache, v_cache, slot_mapping):
    k_cache.zero_()
    v_cache.zero_()
    fn(key, value, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize()

    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    slots = slot_mapping.long()
    expected_k = key.view(n, d)
    expected_v = value.view(n, d)
    actual_k = k_cache.view(-1, d).index_select(0, slots)
    actual_v = v_cache.view(-1, d).index_select(0, slots)
    torch.testing.assert_close(actual_k, expected_k)
    torch.testing.assert_close(actual_v, expected_v)


def cuda_profiler_start(enabled: bool):
    if enabled:
        torch.cuda.cudart().cudaProfilerStart()


def cuda_profiler_stop(enabled: bool):
    if enabled:
        torch.cuda.cudart().cudaProfilerStop()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for store_kvcache profiling.")

    requested_device = torch.device(args.device)
    if requested_device.type != "cuda":
        raise RuntimeError("This profiler is intended for CUDA devices.")

    device_index = 0 if requested_device.index is None else requested_device.index
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    dtype = DTYPES[args.dtype]
    d = args.num_kv_heads * args.head_dim
    min_blocks = math.ceil(args.tokens / args.block_size)
    num_blocks = args.num_blocks if args.num_blocks > 0 else max(min_blocks + 1, 2)
    total_slots = num_blocks * args.block_size
    if args.tokens > total_slots:
        raise ValueError(f"--tokens ({args.tokens}) exceeds cache capacity ({total_slots}).")

    key = torch.randn(args.tokens, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    k_cache = torch.empty(num_blocks, args.block_size, args.num_kv_heads, args.head_dim, device=device, dtype=dtype)
    v_cache = torch.empty_like(k_cache)
    slot_mapping = make_slot_mapping(args, total_slots, device)

    fn = BACKENDS[args.backend]
    if not args.no_check:
        check_correctness(fn, key, value, k_cache, v_cache, slot_mapping)

    for _ in range(args.warmup):
        fn(key, value, k_cache, v_cache, slot_mapping)
    torch.cuda.synchronize()

    use_profiler_range = not args.no_cuda_profiler_range
    cuda_profiler_start(use_profiler_range)
    torch.cuda.nvtx.range_push(f"store_kvcache:{args.backend}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        fn(key, value, k_cache, v_cache, slot_mapping)
    end.record()
    end.synchronize()
    torch.cuda.nvtx.range_pop()
    cuda_profiler_stop(use_profiler_range)

    elapsed_ms = start.elapsed_time(end)
    avg_us = elapsed_ms * 1000.0 / args.iters
    min_bytes = 4 * args.tokens * d * torch.tensor([], dtype=dtype).element_size()
    bandwidth_gbs = min_bytes / (avg_us * 1e-6) / 1e9
    print(
        f"backend={args.backend} tokens={args.tokens} heads={args.num_kv_heads} "
        f"head_dim={args.head_dim} dtype={args.dtype} iters={args.iters} "
        f"avg_us={avg_us:.3f} min_bandwidth_GBps={bandwidth_gbs:.2f}"
    )


if __name__ == "__main__":
    main()
