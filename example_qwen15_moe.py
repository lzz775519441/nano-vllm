import argparse
import os
from time import perf_counter

import torch
from transformers import AutoTokenizer


def cuda_sync():
    torch.cuda.synchronize()


def nvtx_push(name: str):
    torch.cuda.nvtx.range_push(name)


def nvtx_pop():
    torch.cuda.nvtx.range_pop()


def generate_once(llm, prompts, sampling_params, name: str, use_tqdm: bool):
    nvtx_push(name)
    cuda_sync()
    start = perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=use_tqdm)
    cuda_sync()
    elapsed = perf_counter() - start
    nvtx_pop()
    return outputs, elapsed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="~/autodl-tmp/huggingface/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4/",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--warmup-iters", type=int, default=1)
    parser.add_argument("--profile-iters", type=int, default=3)
    parser.add_argument("--cuda-profiler-api", action="store_true")
    parser.add_argument("--tqdm", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    path = os.path.expanduser(args.model)

    from nanovllm import LLM, SamplingParams

    nvtx_push("tokenizer_load")
    tokenizer = AutoTokenizer.from_pretrained(path)
    nvtx_pop()

    nvtx_push("llm_init")
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    cuda_sync()
    nvtx_pop()

    sampling_params = SamplingParams(temperature=0.6, max_tokens=args.max_tokens)
    prompts = [
        "introduce yourself briefly",
        "write a short Python function that checks whether a number is prime",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{
                "role": "user",
                "content": prompt
            }],
            tokenize=False,
            add_generation_prompt=True,
        ) for prompt in prompts
    ]

    for i in range(args.warmup_iters):
        _, elapsed = generate_once(llm, prompts, sampling_params, f"warmup_{i}", args.tqdm)
        print(f"warmup {i}: {elapsed:.3f}s")

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStart()

    outputs = None
    profile_start = perf_counter()
    for i in range(args.profile_iters):
        outputs, elapsed = generate_once(llm, prompts, sampling_params, f"profile_{i}", args.tqdm)
        print(f"profile {i}: {elapsed:.3f}s")
    cuda_sync()
    profile_elapsed = perf_counter() - profile_start

    if args.cuda_profiler_api:
        torch.cuda.cudart().cudaProfilerStop()

    print(f"profile total: {profile_elapsed:.3f}s")

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
