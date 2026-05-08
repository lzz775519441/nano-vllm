import argparse
import csv
import itertools
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


WORKLOAD_ARGS = [
    "--mode", "online",
    "--duration", "60.0",
    "--request-rate", "6.0",
    "--min-input-len", "1024",
    "--max-input-len", "3072",
    "--min-output-len", "128",
    "--max-output-len", "512",
    "--length-distribution", "lognormal",
    "--seed", "0",
    "--max-model-len", "4096",
    "--max-num-seqs", "512",
    "--max-decode-cudagraph-tokens", "512",
]


REFERENCE_METRICS = {
    "name": "baseline",
    "completed": 347,
    "total_requests": 347,
    "wall_time": 78.97,
    "busy_pct": 99.9,
    "request_throughput": 4.39,
    "output_throughput": 905.30,
    "scheduled_throughput": 7392.36,
    "ttft_p50": 5.473,
    "ttft_p90": 11.577,
    "ttft_p99": 12.940,
    "latency_p50": 14.864,
    "latency_p90": 20.096,
    "latency_p99": 28.246,
    "tpot_p50": 45.80,
    "tpot_p90": 49.27,
    "tpot_p99": 52.12,
}


PATTERNS = {
    "completed": re.compile(r"Completed requests: (?P<done>\d+)/(?P<total>\d+)"),
    "wall_time": re.compile(r"Wall time: (?P<value>[\d.]+)s"),
    "busy_pct": re.compile(r"Engine busy: [\d.]+s \((?P<value>[\d.]+)%\)"),
    "request_throughput": re.compile(r"Request throughput: (?P<value>[\d.]+) req/s"),
    "output_throughput": re.compile(r"Output throughput: (?P<value>[\d.]+) tok/s"),
    "scheduled_throughput": re.compile(r"Total scheduled throughput: (?P<value>[\d.]+) tok/s"),
    "ttft": re.compile(r"TTFT p50/p90/p99: (?P<p50>[\d.]+)s / (?P<p90>[\d.]+)s / (?P<p99>[\d.]+)s"),
    "latency": re.compile(r"Latency p50/p90/p99: (?P<p50>[\d.]+)s / (?P<p90>[\d.]+)s / (?P<p99>[\d.]+)s"),
    "tpot": re.compile(r"TPOT p50/p90/p99: (?P<p50>[\d.]+)ms / (?P<p90>[\d.]+)ms / (?P<p99>[\d.]+)ms"),
    "capture_oom": re.compile(r"Piecewise CUDA graph capture stopped at token_bucket=(?P<bucket>\d+)"),
}


def pct_delta(value, reference):
    if value is None or reference in (None, 0):
        return None
    return (float(value) - float(reference)) / float(reference) * 100.0


def fmt(value, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    return f"{float(value):.2f}{suffix}"


def parse_list(raw: str, cast):
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_metrics(text: str) -> dict:
    metrics: dict[str, float | int | bool | str] = {}
    metrics["oom"] = "OutOfMemoryError" in text or "CUDA out of memory" in text

    if match := PATTERNS["completed"].search(text):
        metrics["completed"] = int(match.group("done"))
        metrics["total_requests"] = int(match.group("total"))
    for key in ["wall_time", "busy_pct", "request_throughput", "output_throughput", "scheduled_throughput"]:
        if match := PATTERNS[key].search(text):
            metrics[key] = float(match.group("value"))
    for key in ["ttft", "latency", "tpot"]:
        if match := PATTERNS[key].search(text):
            metrics[f"{key}_p50"] = float(match.group("p50"))
            metrics[f"{key}_p90"] = float(match.group("p90"))
            metrics[f"{key}_p99"] = float(match.group("p99"))
    capture_buckets = [int(match.group("bucket")) for match in PATTERNS["capture_oom"].finditer(text)]
    metrics["capture_oom_bucket"] = capture_buckets[-1] if capture_buckets else 0

    completed = metrics.get("completed", 0)
    total = metrics.get("total_requests", 1)
    if completed != total or metrics["oom"]:
        metrics["score"] = 0.0
    else:
        req_tput = float(metrics.get("request_throughput", 0.0))
        ttft_p90 = max(float(metrics.get("ttft_p90", 1e9)), 1e-9)
        tpot_p90 = max(float(metrics.get("tpot_p90", 1e9)), 1e-9)
        metrics["score"] = req_tput * 1000.0 / (ttft_p90 * tpot_p90)
    return metrics


def add_reference_deltas(row: dict):
    row["request_throughput_delta_pct"] = pct_delta(
        row.get("request_throughput"), REFERENCE_METRICS["request_throughput"]
    )
    row["output_throughput_delta_pct"] = pct_delta(
        row.get("output_throughput"), REFERENCE_METRICS["output_throughput"]
    )
    row["scheduled_throughput_delta_pct"] = pct_delta(
        row.get("scheduled_throughput"), REFERENCE_METRICS["scheduled_throughput"]
    )
    row["ttft_p90_delta_pct"] = pct_delta(row.get("ttft_p90"), REFERENCE_METRICS["ttft_p90"])
    row["latency_p90_delta_pct"] = pct_delta(row.get("latency_p90"), REFERENCE_METRICS["latency_p90"])
    row["tpot_p90_delta_pct"] = pct_delta(row.get("tpot_p90"), REFERENCE_METRICS["tpot_p90"])


def run_one(cmd: list[str], log_path: Path) -> tuple[int, str]:
    print("+ " + shlex.join(cmd), flush=True)
    with log_path.open("w") as log_file:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        assert process.stdout is not None
        chunks = []
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            chunks.append(line)
        return process.wait(), "".join(chunks)


def write_summary(path: Path, rows: list[dict]):
    fieldnames = [
        "rank",
        "score",
        "gpu_memory_utilization",
        "max_num_batched_tokens",
        "max_piecewise_cudagraph_tokens",
        "returncode",
        "oom",
        "capture_oom_bucket",
        "completed",
        "total_requests",
        "wall_time",
        "busy_pct",
        "request_throughput",
        "output_throughput",
        "scheduled_throughput",
        "request_throughput_delta_pct",
        "output_throughput_delta_pct",
        "scheduled_throughput_delta_pct",
        "ttft_p50",
        "ttft_p90",
        "ttft_p99",
        "ttft_p90_delta_pct",
        "latency_p50",
        "latency_p90",
        "latency_p99",
        "latency_p90_delta_pct",
        "tpot_p50",
        "tpot_p90",
        "tpot_p99",
        "tpot_p90_delta_pct",
        "log_path",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep mixed-batch bench serving parameters.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--bench", default="bench.py")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--gpu-memory-utilizations", default="0.68,0.72,0.76,0.80,0.84")
    parser.add_argument("--max-piecewise-cudagraph-tokens", default="256,304,352,384")
    parser.add_argument("--max-num-batched-tokens", default="1024,1280,1536")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir or f"bench-results/mixed-sweep-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_values = parse_list(args.gpu_memory_utilizations, float)
    piecewise_values = parse_list(args.max_piecewise_cudagraph_tokens, int)
    batched_values = parse_list(args.max_num_batched_tokens, int)
    combos = list(itertools.product(batched_values, gpu_values, piecewise_values))
    if args.limit > 0:
        combos = combos[:args.limit]

    rows = []
    for index, (max_batched, gpu_util, max_piecewise) in enumerate(combos, start=1):
        name = f"{index:03d}-bt{max_batched}-gpu{gpu_util:.2f}-pw{max_piecewise}"
        log_path = output_dir / f"{name}.log"
        cmd = [
            args.python,
            args.bench,
            *WORKLOAD_ARGS,
            "--max-num-batched-tokens", str(max_batched),
            "--gpu-memory-utilization", str(gpu_util),
            "--max-piecewise-cudagraph-tokens", str(max_piecewise),
        ]
        print(f"\n[{index}/{len(combos)}] {name}", flush=True)
        if args.dry_run:
            print("+ " + shlex.join(cmd), flush=True)
            continue

        returncode, text = run_one(cmd, log_path)
        row = {
            "gpu_memory_utilization": gpu_util,
            "max_num_batched_tokens": max_batched,
            "max_piecewise_cudagraph_tokens": max_piecewise,
            "returncode": returncode,
            "log_path": str(log_path),
            **parse_metrics(text),
        }
        add_reference_deltas(row)
        rows.append(row)

        ranked = sorted(rows, key=lambda item: float(item.get("score", 0.0)), reverse=True)
        for rank, ranked_row in enumerate(ranked, start=1):
            ranked_row["rank"] = rank
        write_summary(output_dir / "summary.csv", ranked)
        (output_dir / "summary.json").write_text(json.dumps(ranked, indent=2, sort_keys=True))

        best = ranked[0]
        print(
            "Current best: "
            f"score={fmt(best.get('score'))}, "
            f"bt={best['max_num_batched_tokens']}, "
            f"gpu={best['gpu_memory_utilization']}, "
            f"pw={best['max_piecewise_cudagraph_tokens']}, "
            f"TTFT p90={fmt(best.get('ttft_p90'))}, "
            f"TPOT p90={fmt(best.get('tpot_p90'))}, "
            f"req tput delta={fmt(best.get('request_throughput_delta_pct'), '%')}, "
            f"TTFT p90 delta={fmt(best.get('ttft_p90_delta_pct'), '%')}, "
            f"TPOT p90 delta={fmt(best.get('tpot_p90_delta_pct'), '%')}",
            flush=True,
        )

    if rows:
        print(f"\nDone. Summary: {output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
