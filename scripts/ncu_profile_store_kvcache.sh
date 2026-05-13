#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu was not found in PATH. Install Nsight Compute or load the CUDA/NVIDIA profiling module first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
BACKENDS="${BACKENDS:-torch torch_compile triton}"
OUT_DIR="${OUT_DIR:-profiles/store_kvcache_ncu/$(date +%Y%m%d_%H%M%S)}"
TOKENS="${TOKENS:-4096}"
NUM_KV_HEADS="${NUM_KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
DTYPE="${DTYPE:-fp16}"
WARMUP="${WARMUP:-10}"
ITERS="${ITERS:-1}"
SLOT_ORDER="${SLOT_ORDER:-sequential}"
NCU_SET="${NCU_SET:-basic}"

mkdir -p "${OUT_DIR}"

echo "Writing Nsight Compute profiles to: ${OUT_DIR}"
echo "Backends: ${BACKENDS}"
echo "NCU set: ${NCU_SET}"

for backend in ${BACKENDS}; do
  output="${OUT_DIR}/store_kvcache_${backend}"
  echo
  echo "==> Profiling ${backend}"
  ncu \
    --force-overwrite \
    --target-processes all \
    --profile-from-start off \
    --nvtx \
    --set "${NCU_SET}" \
    --export "${output}" \
    "${PYTHON_BIN}" scripts/profile_store_kvcache.py \
      --backend "${backend}" \
      --tokens "${TOKENS}" \
      --num-kv-heads "${NUM_KV_HEADS}" \
      --head-dim "${HEAD_DIM}" \
      --block-size "${BLOCK_SIZE}" \
      --dtype "${DTYPE}" \
      --warmup "${WARMUP}" \
      --iters "${ITERS}" \
      --slot-order "${SLOT_ORDER}" \
      "$@"
done

echo
echo "Done."
echo "Open an interactive report with:"
echo "  ncu-ui ${OUT_DIR}/store_kvcache_triton.ncu-rep"
echo
echo "Print report details with:"
for backend in ${BACKENDS}; do
  echo "  ncu --import ${OUT_DIR}/store_kvcache_${backend}.ncu-rep --page details"
done
