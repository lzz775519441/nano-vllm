#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cfloat>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kBlockSize = 256;
constexpr int kQwen3QHeads = 32;
constexpr int kQwen3KvHeads = 8;
constexpr int kQwen3HeadDim = 128;
constexpr int kQwen3RotaryPairs = kQwen3HeadDim / 2;
constexpr int kQwen3KvWidth = kQwen3KvHeads * kQwen3HeadDim;
constexpr int kQwen3KvVecElems = sizeof(uint4) / sizeof(c10::BFloat16);
constexpr int kQwen3KvWidthVec = kQwen3KvWidth / kQwen3KvVecElems;

inline void check_cuda(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_contiguous(const torch::Tensor &t, const char *name) {
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform01(std::uint64_t seed, int row,
                                           int col) {
  std::uint64_t x = seed;
  x ^= static_cast<std::uint64_t>(row) * 0xd1b54a32d192ed03ULL;
  x ^= static_cast<std::uint64_t>(col) * 0xabc98388fb8fac03ULL;
  x = splitmix64(x);
  constexpr double denom = 1.0 / 9007199254740992.0;  // 2^53
  double u = static_cast<double>(x >> 11) * denom;
  u = fmin(fmax(u, 1.0e-12), 1.0 - 1.0e-12);
  return static_cast<float>(u);
}

template <typename scalar_t>
__global__ void sample_kernel(const scalar_t *__restrict__ logits,
                              const float *__restrict__ temperatures,
                              int64_t *__restrict__ output, int batch,
                              int vocab, std::uint64_t seed) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float inv_temp = 1.0f / temperatures[row];
  float best_score = -FLT_MAX;
  int best_idx = 0;

  for (int col = tid; col < vocab; col += blockDim.x) {
    float logit =
        static_cast<float>(logits[static_cast<int64_t>(row) * vocab + col]) *
        inv_temp;
    float u = uniform01(seed, row, col);
    float gumbel = -logf(-logf(u));
    float score = logit + gumbel;
    if (score > best_score || (score == best_score && col < best_idx)) {
      best_score = score;
      best_idx = col;
    }
  }

  __shared__ float scores[kBlockSize];
  __shared__ int indices[kBlockSize];
  scores[tid] = best_score;
  indices[tid] = best_idx;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      float other_score = scores[tid + stride];
      int other_idx = indices[tid + stride];
      if (other_score > scores[tid] ||
          (other_score == scores[tid] && other_idx < indices[tid])) {
        scores[tid] = other_score;
        indices[tid] = other_idx;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    output[row] = static_cast<int64_t>(indices[0]);
  }
}

template <typename scalar_t>
__global__ void rms_norm_kernel(const scalar_t *__restrict__ x,
                                const scalar_t *__restrict__ weight,
                                scalar_t *__restrict__ out, int rows,
                                int hidden, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += blockDim.x) {
    float v = static_cast<float>(x[static_cast<int64_t>(row) * hidden + col]);
    sum += v * v;
  }

  __shared__ float shared[kBlockSize];
  shared[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  float scale = rsqrtf(shared[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    int64_t offset = static_cast<int64_t>(row) * hidden + col;
    float v =
        static_cast<float>(x[offset]) * scale * static_cast<float>(weight[col]);
    out[offset] = static_cast<scalar_t>(v);
  }
}

template <typename scalar_t>
__global__ void add_rms_norm_kernel(const scalar_t *__restrict__ x,
                                    const scalar_t *__restrict__ residual,
                                    const scalar_t *__restrict__ weight,
                                    scalar_t *__restrict__ out,
                                    scalar_t *__restrict__ new_residual,
                                    int rows, int hidden, float eps) {
  int row = blockIdx.x;
  int tid = threadIdx.x;
  float sum = 0.0f;

  for (int col = tid; col < hidden; col += blockDim.x) {
    int64_t offset = static_cast<int64_t>(row) * hidden + col;
    float v =
        static_cast<float>(x[offset]) + static_cast<float>(residual[offset]);
    sum += v * v;
    new_residual[offset] = static_cast<scalar_t>(v);
  }

  __shared__ float shared[kBlockSize];
  shared[tid] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared[tid] += shared[tid + stride];
    }
    __syncthreads();
  }

  float scale = rsqrtf(shared[0] / static_cast<float>(hidden) + eps);
  for (int col = tid; col < hidden; col += blockDim.x) {
    int64_t offset = static_cast<int64_t>(row) * hidden + col;
    float v = static_cast<float>(new_residual[offset]) * scale *
              static_cast<float>(weight[col]);
    out[offset] = static_cast<scalar_t>(v);
  }
}

__global__ void qwen3_rotary_qk_kernel(
    const int64_t *__restrict__ positions,
    const c10::BFloat16 *__restrict__ query,
    const c10::BFloat16 *__restrict__ key, c10::BFloat16 *__restrict__ q_out,
    c10::BFloat16 *__restrict__ k_out, const float *__restrict__ cos_sin_cache,
    int64_t positions_stride0, int64_t q_stride0, int64_t q_stride1,
    int64_t q_stride2, int64_t k_stride0, int64_t k_stride1, int64_t k_stride2,
    int64_t cos_sin_stride0, int64_t cos_sin_stride2) {
  int token = blockIdx.x;
  int head = blockIdx.y;
  int i = threadIdx.x;
  if (i >= kQwen3RotaryPairs) return;
  int64_t pos = positions[static_cast<int64_t>(token) * positions_stride0];
  const float *cache = cos_sin_cache + pos * cos_sin_stride0;
  float cos = cache[static_cast<int64_t>(i) * cos_sin_stride2];
  float sin =
      cache[static_cast<int64_t>(i + kQwen3RotaryPairs) * cos_sin_stride2];
  int64_t q_base = static_cast<int64_t>(token) * q_stride0 +
                   static_cast<int64_t>(head) * q_stride1;
  int64_t q_output_base =
      (static_cast<int64_t>(token) * kQwen3QHeads + head) * kQwen3HeadDim;
  int64_t q_i = q_base + static_cast<int64_t>(i) * q_stride2;
  int64_t q_half =
      q_base + static_cast<int64_t>(i + kQwen3RotaryPairs) * q_stride2;
  float q1 = static_cast<float>(query[q_i]);
  float q2 = static_cast<float>(query[q_half]);
  q_out[q_output_base + i] = static_cast<c10::BFloat16>(q1 * cos - q2 * sin);
  q_out[q_output_base + i + kQwen3RotaryPairs] =
      static_cast<c10::BFloat16>(q1 * sin + q2 * cos);
  if (head >= kQwen3KvHeads) return;
  int64_t k_base = static_cast<int64_t>(token) * k_stride0 +
                   static_cast<int64_t>(head) * k_stride1;
  int64_t k_output_base =
      (static_cast<int64_t>(token) * kQwen3KvHeads + head) * kQwen3HeadDim;
  int64_t k_i = k_base + static_cast<int64_t>(i) * k_stride2;
  int64_t k_half =
      k_base + static_cast<int64_t>(i + kQwen3RotaryPairs) * k_stride2;
  float k1 = static_cast<float>(key[k_i]);
  float k2 = static_cast<float>(key[k_half]);
  k_out[k_output_base + i] = static_cast<c10::BFloat16>(k1 * cos - k2 * sin);
  k_out[k_output_base + i + kQwen3RotaryPairs] =
      static_cast<c10::BFloat16>(k1 * sin + k2 * cos);
}

__global__ void store_kvcache_vec_kernel(
    const uint4 *__restrict__ key, const uint4 *__restrict__ value,
    uint4 *__restrict__ k_cache, uint4 *__restrict__ v_cache,
    const int32_t *__restrict__ slot_mapping, int64_t k_stride0_vec,
    int64_t v_stride0_vec) {
  int token = blockIdx.x;
  int col = threadIdx.x;
  int slot = slot_mapping[token];
  if (slot < 0) return;
  int64_t k_src = static_cast<int64_t>(token) * k_stride0_vec;
  int64_t v_src = static_cast<int64_t>(token) * v_stride0_vec;
  int64_t dst = static_cast<int64_t>(slot) * kQwen3KvWidthVec;
  for (int i = col; i < kQwen3KvWidthVec; i += blockDim.x) {
    k_cache[dst + i] = key[k_src + i];
    v_cache[dst + i] = value[v_src + i];
  }
}

}  // namespace

torch::Tensor sample(torch::Tensor logits, torch::Tensor temperatures,
                     std::uint64_t seed) {
  check_cuda(logits, "logits");
  check_cuda(temperatures, "temperatures");
  check_contiguous(logits, "logits");
  check_contiguous(temperatures, "temperatures");
  TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
  TORCH_CHECK(temperatures.dim() == 1, "temperatures must be 1D");
  TORCH_CHECK(temperatures.scalar_type() == torch::kFloat32,
              "temperatures must be float32");
  TORCH_CHECK(logits.size(0) == temperatures.size(0), "batch size mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
  auto output =
      torch::empty({logits.size(0)}, logits.options().dtype(torch::kInt64));
  int batch = static_cast<int>(logits.size(0));
  int vocab = static_cast<int>(logits.size(1));
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, logits.scalar_type(), "sample_cuda", [&] {
        sample_kernel<scalar_t><<<batch, kBlockSize, 0, stream>>>(
            logits.data_ptr<scalar_t>(), temperatures.data_ptr<float>(),
            output.data_ptr<int64_t>(), batch, vocab, seed);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor rms_norm(torch::Tensor x, torch::Tensor weight, double eps) {
  check_cuda(x, "x");
  check_cuda(weight, "weight");
  check_contiguous(x, "x");
  check_contiguous(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 1, "weight must be 1D");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "x and weight dtype mismatch");
  int hidden = static_cast<int>(x.size(-1));
  TORCH_CHECK(weight.size(0) == hidden, "weight size mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto out = torch::empty_like(x);
  int64_t rows64 = x.numel() / hidden;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "too many rows");
  int rows = static_cast<int>(rows64);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, x.scalar_type(), "rms_norm_cuda", [&] {
        rms_norm_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
            x.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), rows, hidden, static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

std::vector<torch::Tensor> add_rms_norm(torch::Tensor x, torch::Tensor residual,
                                        torch::Tensor weight, double eps) {
  check_cuda(x, "x");
  check_cuda(residual, "residual");
  check_cuda(weight, "weight");
  check_contiguous(x, "x");
  check_contiguous(residual, "residual");
  check_contiguous(weight, "weight");
  TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual shape mismatch");
  TORCH_CHECK(x.scalar_type() == residual.scalar_type(),
              "x and residual dtype mismatch");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "x and weight dtype mismatch");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  int hidden = static_cast<int>(x.size(-1));
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden,
              "weight size mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
  auto out = torch::empty_like(x);
  auto new_residual = torch::empty_like(x);
  int64_t rows64 = x.numel() / hidden;
  TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "too many rows");
  int rows = static_cast<int>(rows64);
  auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      torch::kHalf, torch::kBFloat16, x.scalar_type(), "add_rms_norm_cuda",
      [&] {
        add_rms_norm_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
            x.data_ptr<scalar_t>(), residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(), out.data_ptr<scalar_t>(),
            new_residual.data_ptr<scalar_t>(), rows, hidden,
            static_cast<float>(eps));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {out, new_residual};
}

std::vector<torch::Tensor> rotary_embedding(torch::Tensor positions,
                                            torch::Tensor query,
                                            torch::Tensor key,
                                            torch::Tensor cos_sin_cache) {
  check_cuda(positions, "positions");
  check_cuda(query, "query");
  check_cuda(key, "key");
  check_cuda(cos_sin_cache, "cos_sin_cache");
  TORCH_CHECK(positions.scalar_type() == torch::kInt64,
              "positions must be int64");
  TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32,
              "cos_sin_cache must be float32");
  TORCH_CHECK(query.scalar_type() == torch::kBFloat16 &&
                  key.scalar_type() == torch::kBFloat16,
              "Qwen3 rotary only supports bfloat16 query/key");
  TORCH_CHECK(query.dim() == 3 && query.size(1) == kQwen3QHeads &&
                  query.size(2) == kQwen3HeadDim,
              "Qwen3 rotary expects query [tokens, 32, 128]");
  TORCH_CHECK(key.dim() == 3 && key.size(1) == kQwen3KvHeads &&
                  key.size(2) == kQwen3HeadDim,
              "Qwen3 rotary expects key [tokens, 8, 128]");
  TORCH_CHECK(query.size(0) == key.size(0), "query/key token count mismatch");
  TORCH_CHECK(positions.size(0) == query.size(0),
              "positions token count mismatch");
  TORCH_CHECK(cos_sin_cache.dim() == 3 && cos_sin_cache.size(1) == 1 &&
                  cos_sin_cache.size(2) == kQwen3HeadDim,
              "cos_sin_cache must be [max_position, 1, 128]");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  auto q_out = torch::empty(query.sizes(), query.options());
  auto k_out = torch::empty(key.sizes(), key.options());
  int tokens = static_cast<int>(query.size(0));
  int64_t positions_stride0 = positions.stride(0);
  int64_t cos_sin_stride0 = cos_sin_cache.stride(0);
  int64_t cos_sin_stride2 = cos_sin_cache.stride(2);
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kQwen3RotaryPairs);
  dim3 grid(tokens, kQwen3QHeads);
  qwen3_rotary_qk_kernel<<<grid, block, 0, stream>>>(
      positions.data_ptr<int64_t>(), query.data_ptr<c10::BFloat16>(),
      key.data_ptr<c10::BFloat16>(), q_out.data_ptr<c10::BFloat16>(),
      k_out.data_ptr<c10::BFloat16>(), cos_sin_cache.data_ptr<float>(),
      positions_stride0, query.stride(0), query.stride(1), query.stride(2),
      key.stride(0), key.stride(1), key.stride(2), cos_sin_stride0,
      cos_sin_stride2);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {q_out, k_out};
}

void store_kvcache(torch::Tensor key, torch::Tensor value,
                   torch::Tensor k_cache, torch::Tensor v_cache,
                   torch::Tensor slot_mapping) {
  check_cuda(key, "key");
  check_cuda(value, "value");
  check_cuda(k_cache, "k_cache");
  check_cuda(v_cache, "v_cache");
  check_cuda(slot_mapping, "slot_mapping");
  TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
  TORCH_CHECK(key.scalar_type() == torch::kBFloat16 &&
                  value.scalar_type() == torch::kBFloat16 &&
                  k_cache.scalar_type() == torch::kBFloat16 &&
                  v_cache.scalar_type() == torch::kBFloat16,
              "Qwen3 KV store only supports bfloat16");
  TORCH_CHECK(key.dim() == 3 && key.size(1) == kQwen3KvHeads &&
                  key.size(2) == kQwen3HeadDim,
              "Qwen3 KV store expects key/value [tokens, 8, 128]");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(),
              "KV cache tensors must be contiguous");
  TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
  TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt32,
              "slot_mapping must be int32");
  TORCH_CHECK(slot_mapping.numel() == key.size(0),
              "slot_mapping length mismatch");

  const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
  int tokens = static_cast<int>(key.size(0));
  TORCH_CHECK(key.stride(1) == kQwen3HeadDim && key.stride(2) == 1,
              "key inner layout must be contiguous");
  TORCH_CHECK(value.stride(1) == kQwen3HeadDim && value.stride(2) == 1,
              "value inner layout must be contiguous");
  TORCH_CHECK(key.stride(0) % kQwen3KvVecElems == 0,
              "key token stride must be 16-byte aligned");
  TORCH_CHECK(value.stride(0) % kQwen3KvVecElems == 0,
              "value token stride must be 16-byte aligned");
  TORCH_CHECK(k_cache.dim() == 4 && v_cache.dim() == 4,
              "KV cache must be [blocks, block_size, heads, head_dim]");
  TORCH_CHECK(
      k_cache.size(2) == kQwen3KvHeads && k_cache.size(3) == kQwen3HeadDim,
      "k_cache shape mismatch");
  TORCH_CHECK(
      v_cache.size(2) == kQwen3KvHeads && v_cache.size(3) == kQwen3HeadDim,
      "v_cache shape mismatch");
  auto stream = at::cuda::getCurrentCUDAStream();
  dim3 block(kBlockSize);
  dim3 grid(tokens);
  store_kvcache_vec_kernel<<<grid, block, 0, stream>>>(
      reinterpret_cast<const uint4 *>(key.data_ptr<c10::BFloat16>()),
      reinterpret_cast<const uint4 *>(value.data_ptr<c10::BFloat16>()),
      reinterpret_cast<uint4 *>(k_cache.data_ptr<c10::BFloat16>()),
      reinterpret_cast<uint4 *>(v_cache.data_ptr<c10::BFloat16>()),
      slot_mapping.data_ptr<int32_t>(), key.stride(0) / kQwen3KvVecElems,
      value.stride(0) / kQwen3KvVecElems);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
