import argparse
import os
import random
import time
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class RequestState:
    index: int
    arrival_time: float
    prompt_len: int
    target_output_len: int
    first_token_time: float | None = None
    finish_time: float | None = None
    actual_output_len: int = 0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int((len(values) - 1) * pct / 100))
    return values[idx]


def build_requests(args):
    from nanovllm import SamplingParams

    random.seed(args.seed)
    prompt_token_ids = [
        [random.randint(0, args.vocab_range) for _ in range(random.randint(args.min_input_len, args.max_input_len))]
        for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(
            temperature=args.temperature,
            ignore_eos=True,
            max_tokens=random.randint(args.min_output_len, args.max_output_len),
        )
        for _ in range(args.num_seqs)
    ]
    return prompt_token_ids, sampling_params


def build_arrivals(num_requests: int, request_rate: float) -> list[float]:
    if request_rate <= 0:
        return [0.0] * num_requests
    arrivals = [0.0]
    for _ in range(1, num_requests):
        arrivals.append(arrivals[-1] + random.expovariate(request_rate))
    return arrivals


def make_llm(args):
    from nanovllm import LLM

    return LLM(
        os.path.expanduser(args.model),
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        cudagraph_mode=args.cudagraph_mode,
    )


def warmup(llm):
    from nanovllm import SamplingParams

    llm.generate(["Benchmark: "], SamplingParams(), use_tqdm=False)
    torch.cuda.synchronize()


def run_offline(args):
    prompt_token_ids, sampling_params = build_requests(args)
    llm = make_llm(args)
    warmup(llm)

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    prompt_tokens = sum(len(prompt) for prompt in prompt_token_ids)
    output_tokens = sum(sp.max_tokens for sp in sampling_params)
    print("Offline benchmark")
    print(f"Requests: {args.num_seqs}")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Wall time: {elapsed:.2f}s")
    print(f"Output throughput: {output_tokens / elapsed:.2f} tok/s")
    print(f"Total token throughput: {(prompt_tokens + output_tokens) / elapsed:.2f} tok/s")


def run_online_step(llm, states: dict[int, RequestState]):
    step_start = time.perf_counter()
    scheduler_output = llm.scheduler.schedule()
    if hasattr(scheduler_output, "seqs"):
        seqs = scheduler_output.seqs
        sampled_seq_ids = [seq.seq_id for seq in seqs if seq.needs_sampling]
        token_ids = llm.model_runner.call("run", seqs)
        llm.scheduler.postprocess(scheduler_output, token_ids)
        num_prefill_tokens = scheduler_output.num_prefill_tokens
        num_decode_tokens = scheduler_output.num_decode_tokens
    else:
        seqs, is_prefill = scheduler_output
        if is_prefill:
            sampled_seq_ids = [
                seq.seq_id
                for seq in seqs
                if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens
            ]
            num_prefill_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            num_decode_tokens = 0
        else:
            sampled_seq_ids = [seq.seq_id for seq in seqs]
            num_prefill_tokens = 0
            num_decode_tokens = len(seqs)
        token_ids = llm.model_runner.call("run", seqs, is_prefill)
        llm.scheduler.postprocess(seqs, token_ids, is_prefill)
    step_end = time.perf_counter()

    for seq_id in sampled_seq_ids:
        state = states.get(seq_id)
        if state is not None and state.first_token_time is None:
            state.first_token_time = step_end

    for seq in seqs:
        if seq.is_finished:
            state = states.get(seq.seq_id)
            if state is not None and state.finish_time is None:
                state.finish_time = step_end
                state.actual_output_len = len(seq.completion_token_ids)

    return num_prefill_tokens, num_decode_tokens, step_end - step_start


def print_online_report(states: dict[int, RequestState], wall_time: float, busy_time: float, prefill_tokens: int, decode_tokens: int):
    requests = list(states.values())
    completed = [state for state in requests if state.finish_time is not None]
    ttfts = [state.first_token_time - state.arrival_time for state in completed if state.first_token_time is not None]
    latencies = [state.finish_time - state.arrival_time for state in completed]
    output_tokens = sum(state.actual_output_len for state in completed)
    prompt_tokens = sum(state.prompt_len for state in completed)
    tpots = [
        (state.finish_time - state.first_token_time) / max(1, state.actual_output_len - 1)
        for state in completed
        if state.first_token_time is not None and state.actual_output_len > 0
    ]

    print("Online benchmark")
    print(f"Completed requests: {len(completed)}/{len(requests)}")
    print(f"Wall time: {wall_time:.2f}s")
    print(f"Engine busy: {busy_time:.2f}s ({busy_time / wall_time * 100:.1f}%)")
    print(f"Prompt tokens: {prompt_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Scheduled prefill/decode tokens: {prefill_tokens}/{decode_tokens}")
    print(f"Request throughput: {len(completed) / wall_time:.2f} req/s")
    print(f"Output throughput: {output_tokens / wall_time:.2f} tok/s")
    print(f"Total scheduled throughput: {(prefill_tokens + decode_tokens) / wall_time:.2f} tok/s")
    print(f"TTFT p50/p90/p99: {percentile(ttfts, 50):.3f}s / {percentile(ttfts, 90):.3f}s / {percentile(ttfts, 99):.3f}s")
    print(f"Latency p50/p90/p99: {percentile(latencies, 50):.3f}s / {percentile(latencies, 90):.3f}s / {percentile(latencies, 99):.3f}s")
    print(f"TPOT p50/p90/p99: {percentile(tpots, 50) * 1000:.2f}ms / {percentile(tpots, 90) * 1000:.2f}ms / {percentile(tpots, 99) * 1000:.2f}ms")


def run_online(args):
    prompt_token_ids, sampling_params = build_requests(args)
    arrivals = build_arrivals(args.num_seqs, args.request_rate)
    llm = make_llm(args)
    warmup(llm)

    states: dict[int, RequestState] = {}
    next_request = 0
    total_prefill_tokens = 0
    total_decode_tokens = 0
    busy_time = 0.0

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()
    start = time.perf_counter()
    while next_request < args.num_seqs or not llm.is_finished():
        now = time.perf_counter()
        elapsed = now - start
        while next_request < args.num_seqs and arrivals[next_request] <= elapsed:
            seq_id = llm.add_request(prompt_token_ids[next_request], sampling_params[next_request])
            states[seq_id] = RequestState(
                index=next_request,
                arrival_time=start + arrivals[next_request],
                prompt_len=len(prompt_token_ids[next_request]),
                target_output_len=sampling_params[next_request].max_tokens,
            )
            next_request += 1

        if llm.is_finished():
            if next_request < args.num_seqs:
                sleep_time = start + arrivals[next_request] - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
            continue

        num_prefill_tokens, num_decode_tokens, step_time = run_online_step(llm, states)
        total_prefill_tokens += num_prefill_tokens
        total_decode_tokens += num_decode_tokens
        busy_time += step_time

    torch.cuda.synchronize()
    wall_time = time.perf_counter() - start
    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()

    print_online_report(states, wall_time, busy_time, total_prefill_tokens, total_decode_tokens)


def parse_args():
    parser = argparse.ArgumentParser(description="nano-vLLM offline and online benchmark")
    parser.add_argument("--mode", choices=["online", "offline"], default="online")
    parser.add_argument("--model", default="~/autodl-tmp/huggingface/Qwen3-8B/")
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--request-rate", type=float, default=32.0, help="Poisson arrival rate in requests/s. Use 0 for a burst.")
    parser.add_argument("--min-input-len", type=int, default=100)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--min-output-len", type=int, default=100)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--vocab-range", type=int, default=10000)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cudagraph-mode", default="full_and_piecewise", choices=["none", "full_decode_only", "piecewise", "full_and_piecewise"])
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "offline":
        run_offline(args)
    else:
        run_online(args)


if __name__ == "__main__":
    main()
