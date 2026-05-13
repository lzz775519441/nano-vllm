#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found in PATH. Install Nsight Systems or load the CUDA/NVIDIA profiling module first." >&2
  exit 1
fi

PROFILE_DIR="${1:-${PROFILE_DIR:-}}"
if [[ -z "${PROFILE_DIR}" ]]; then
  if [[ ! -d profiles/store_kvcache ]]; then
    echo "No profile directory was provided and profiles/store_kvcache does not exist." >&2
    echo "Usage: $0 profiles/store_kvcache/<timestamp>" >&2
    exit 1
  fi
  PROFILE_DIR="$(
    find profiles/store_kvcache -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' \
      | sort -nr \
      | head -n 1 \
      | cut -d' ' -f2-
  )"
fi

if [[ -z "${PROFILE_DIR}" || ! -d "${PROFILE_DIR}" ]]; then
  echo "Profile directory not found: ${PROFILE_DIR}" >&2
  echo "Usage: $0 profiles/store_kvcache/<timestamp>" >&2
  exit 1
fi

BACKENDS="${BACKENDS:-torch torch_compile triton}"
REPORT="${REPORT:-cuda_gpu_kern_sum}"
FORMATS="${FORMATS:-column csv}"
STATS_DIR="${STATS_DIR:-${PROFILE_DIR}/stats}"
COMBINED_TXT="${STATS_DIR}/${REPORT}_all.txt"

mkdir -p "${STATS_DIR}"
: > "${COMBINED_TXT}"

echo "Reading Nsight Systems reports from: ${PROFILE_DIR}"
echo "Writing stats to: ${STATS_DIR}"
echo "Report: ${REPORT}"
echo "Formats: ${FORMATS}"

processed=0
for backend in ${BACKENDS}; do
  rep="${PROFILE_DIR}/store_kvcache_${backend}.nsys-rep"
  if [[ ! -f "${rep}" ]]; then
    echo
    echo "Skipping ${backend}: missing ${rep}" >&2
    continue
  fi

  output_base="${STATS_DIR}/store_kvcache_${backend}"
  reports_arg=""
  formats_arg=""
  outputs_arg=""
  for format in ${FORMATS}; do
    reports_arg="${reports_arg:+${reports_arg},}${REPORT}"
    formats_arg="${formats_arg:+${formats_arg},}${format}"
    outputs_arg="${outputs_arg:+${outputs_arg},}${output_base}"
  done

  echo
  echo "==> ${backend}"
  nsys stats \
    --force-overwrite=true \
    --report "${reports_arg}" \
    --format "${formats_arg}" \
    --output "${outputs_arg}" \
    "${rep}"

  txt="${output_base}_${REPORT}.txt"
  if [[ -f "${txt}" ]]; then
    {
      echo "== ${backend} =="
      cat "${txt}"
      echo
    } >> "${COMBINED_TXT}"
  fi
  processed=$((processed + 1))
done

if [[ "${processed}" -eq 0 ]]; then
  echo "No .nsys-rep files were processed." >&2
  exit 1
fi

echo
echo "Done."
echo "Combined text summary:"
echo "  ${COMBINED_TXT}"
echo
echo "Per-backend files:"
find "${STATS_DIR}" -maxdepth 1 -type f -name "store_kvcache_*_${REPORT}.*" -printf "  %p\n" | sort
