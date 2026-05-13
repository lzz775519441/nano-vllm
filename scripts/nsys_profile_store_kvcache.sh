#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v nsys >/dev/null 2>&1; then
  echo "nsys was not found in PATH. Install Nsight Systems or load the CUDA/NVIDIA profiling module first." >&2
  exit 1
fi

PYTHON_BIN="${PYTHON:-python}"
BACKENDS="${BACKENDS:-torch torch_compile triton}"
OUT_DIR="${OUT_DIR:-profiles/store_kvcache/$(date +%Y%m%d_%H%M%S)}"
TOKENS="${TOKENS:-4096}"
NUM_KV_HEADS="${NUM_KV_HEADS:-8}"
HEAD_DIM="${HEAD_DIM:-128}"
BLOCK_SIZE="${BLOCK_SIZE:-256}"
DTYPE="${DTYPE:-fp16}"
WARMUP="${WARMUP:-50}"
ITERS="${ITERS:-500}"
SLOT_ORDER="${SLOT_ORDER:-sequential}"

mkdir -p "${OUT_DIR}"

echo "Writing Nsight Systems profiles to: ${OUT_DIR}"
echo "Backends: ${BACKENDS}"

for backend in ${BACKENDS}; do
  output="${OUT_DIR}/store_kvcache_${backend}"
  echo
  echo "==> Profiling ${backend}"
  nsys profile \
    --force-overwrite=true \
    --trace=cuda,nvtx,osrt \
    --sample=none \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --output="${output}" \
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
echo "Open a timeline with:"
echo "  nsys-ui ${OUT_DIR}/store_kvcache_triton.nsys-rep"
echo
echo "Print CUDA kernel summaries with:"
for backend in ${BACKENDS}; do
  echo "  nsys stats --report cuda_gpu_kern_sum ${OUT_DIR}/store_kvcache_${backend}.nsys-rep"
done
