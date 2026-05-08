#!/usr/bin/env bash
set -euo pipefail

timestamp="$(date +%Y%m%d-%H%M%S)"
output_dir="${1:-bench-results/mixed-sweep-${timestamp}}"
shift || true

mkdir -p "${output_dir}"
nohup python3 bench_sweep.py --output-dir "${output_dir}" "$@" > "${output_dir}/driver.log" 2>&1 &
pid="$!"

echo "${pid}" > "${output_dir}/pid"
echo "Started bench sweep in background."
echo "PID: ${pid}"
echo "Output dir: ${output_dir}"
echo "Driver log: ${output_dir}/driver.log"
echo "Summary: ${output_dir}/summary.csv"
echo
echo "Watch with:"
echo "  tail -f ${output_dir}/driver.log"
