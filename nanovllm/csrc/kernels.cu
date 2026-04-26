#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

constexpr int kBlockSize = 256;

inline void check_cuda(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
}

inline void check_contiguous(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__device__ __forceinline__ float uniform01(std::uint64_t seed, int row, int col) {
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
__global__ void sample_kernel(
    const scalar_t* __restrict__ logits,
    const float* __restrict__ temperatures,
    int64_t* __restrict__ output,
    int batch,
    int vocab,
    std::uint64_t seed) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float inv_temp = 1.0f / temperatures[row];
    float best_score = -FLT_MAX;
    int best_idx = 0;

    for (int col = tid; col < vocab; col += blockDim.x) {
        float logit = static_cast<float>(logits[static_cast<int64_t>(row) * vocab + col]) * inv_temp;
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
            if (other_score > scores[tid] || (other_score == scores[tid] && other_idx < indices[tid])) {
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
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int rows,
    int hidden,
    float eps) {
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
        float v = static_cast<float>(x[offset]) * scale * static_cast<float>(weight[col]);
        out[offset] = static_cast<scalar_t>(v);
    }
}

template <typename scalar_t>
__global__ void add_rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ residual,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ new_residual,
    int rows,
    int hidden,
    float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float sum = 0.0f;

    for (int col = tid; col < hidden; col += blockDim.x) {
        int64_t offset = static_cast<int64_t>(row) * hidden + col;
        float v = static_cast<float>(x[offset]) + static_cast<float>(residual[offset]);
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
        float v = static_cast<float>(new_residual[offset]) * scale * static_cast<float>(weight[col]);
        out[offset] = static_cast<scalar_t>(v);
    }
}

template <typename scalar_t>
__global__ void rotary_one_kernel(
    const int64_t* __restrict__ positions,
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const float* __restrict__ cos_sin_cache,
    int tokens,
    int heads,
    int head_dim) {
    int64_t total = static_cast<int64_t>(tokens) * heads * (head_dim / 2);
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int half = head_dim / 2;
        int i = linear % half;
        int head = (linear / half) % heads;
        int token = linear / (half * heads);
        int64_t base = (static_cast<int64_t>(token) * heads + head) * head_dim;
        int64_t pos = positions[token];
        const float* cache = cos_sin_cache + pos * head_dim;
        float cos = cache[i];
        float sin = cache[i + half];
        float x1 = static_cast<float>(input[base + i]);
        float x2 = static_cast<float>(input[base + i + half]);
        output[base + i] = static_cast<scalar_t>(x1 * cos - x2 * sin);
        output[base + i + half] = static_cast<scalar_t>(x2 * cos + x1 * sin);
    }
}

template <typename scalar_t>
__global__ void store_kvcache_kernel(
    const scalar_t* __restrict__ key,
    const scalar_t* __restrict__ value,
    scalar_t* __restrict__ k_cache,
    scalar_t* __restrict__ v_cache,
    const int32_t* __restrict__ slot_mapping,
    int tokens,
    int width) {
    int64_t total = static_cast<int64_t>(tokens) * width;
    for (int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < total;
         linear += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        int token = linear / width;
        int col = linear % width;
        int slot = slot_mapping[token];
        if (slot < 0) {
            continue;
        }
        int64_t src = static_cast<int64_t>(token) * width + col;
        int64_t dst = static_cast<int64_t>(slot) * width + col;
        k_cache[dst] = key[src];
        v_cache[dst] = value[src];
    }
}

int blocks_for(int64_t n) {
    return static_cast<int>(std::min<int64_t>((n + kBlockSize - 1) / kBlockSize, 65535));
}

}  // namespace

torch::Tensor sample(torch::Tensor logits, torch::Tensor temperatures, std::uint64_t seed) {
    check_cuda(logits, "logits");
    check_cuda(temperatures, "temperatures");
    check_contiguous(logits, "logits");
    check_contiguous(temperatures, "temperatures");
    TORCH_CHECK(logits.dim() == 2, "logits must be 2D");
    TORCH_CHECK(temperatures.dim() == 1, "temperatures must be 1D");
    TORCH_CHECK(temperatures.scalar_type() == torch::kFloat32, "temperatures must be float32");
    TORCH_CHECK(logits.size(0) == temperatures.size(0), "batch size mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(logits));
    auto output = torch::empty({logits.size(0)}, logits.options().dtype(torch::kInt64));
    int batch = static_cast<int>(logits.size(0));
    int vocab = static_cast<int>(logits.size(1));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, logits.scalar_type(), "sample_cuda", [&] {
        sample_kernel<scalar_t><<<batch, kBlockSize, 0, stream>>>(
            logits.data_ptr<scalar_t>(),
            temperatures.data_ptr<float>(),
            output.data_ptr<int64_t>(),
            batch,
            vocab,
            seed);
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
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtype mismatch");
    int hidden = static_cast<int>(x.size(-1));
    TORCH_CHECK(weight.size(0) == hidden, "weight size mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    int64_t rows64 = x.numel() / hidden;
    TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "too many rows");
    int rows = static_cast<int>(rows64);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, x.scalar_type(), "rms_norm_cuda", [&] {
        rms_norm_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            rows,
            hidden,
            static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> add_rms_norm(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, double eps) {
    check_cuda(x, "x");
    check_cuda(residual, "residual");
    check_cuda(weight, "weight");
    check_contiguous(x, "x");
    check_contiguous(residual, "residual");
    check_contiguous(weight, "weight");
    TORCH_CHECK(x.sizes() == residual.sizes(), "x and residual shape mismatch");
    TORCH_CHECK(x.scalar_type() == residual.scalar_type(), "x and residual dtype mismatch");
    TORCH_CHECK(x.scalar_type() == weight.scalar_type(), "x and weight dtype mismatch");
    TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
    int hidden = static_cast<int>(x.size(-1));
    TORCH_CHECK(weight.dim() == 1 && weight.size(0) == hidden, "weight size mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(x));
    auto out = torch::empty_like(x);
    auto new_residual = torch::empty_like(x);
    int64_t rows64 = x.numel() / hidden;
    TORCH_CHECK(rows64 <= std::numeric_limits<int>::max(), "too many rows");
    int rows = static_cast<int>(rows64);
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, x.scalar_type(), "add_rms_norm_cuda", [&] {
        add_rms_norm_kernel<scalar_t><<<rows, kBlockSize, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(),
            new_residual.data_ptr<scalar_t>(),
            rows,
            hidden,
            static_cast<float>(eps));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, new_residual};
}

std::vector<torch::Tensor> rotary_embedding(torch::Tensor positions, torch::Tensor query, torch::Tensor key, torch::Tensor cos_sin_cache) {
    check_cuda(positions, "positions");
    check_cuda(query, "query");
    check_cuda(key, "key");
    check_cuda(cos_sin_cache, "cos_sin_cache");
    check_contiguous(positions, "positions");
    check_contiguous(query, "query");
    check_contiguous(key, "key");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64, "positions must be int64");
    TORCH_CHECK(cos_sin_cache.scalar_type() == torch::kFloat32, "cos_sin_cache must be float32");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 3, "query/key must be [tokens, heads, head_dim]");
    TORCH_CHECK(query.scalar_type() == key.scalar_type(), "query/key dtype mismatch");
    TORCH_CHECK(query.size(0) == key.size(0), "query/key token count mismatch");
    TORCH_CHECK(query.size(2) == key.size(2), "query/key head_dim mismatch");
    TORCH_CHECK(query.size(2) % 2 == 0, "head_dim must be even");
    TORCH_CHECK(positions.size(0) == query.size(0), "positions token count mismatch");
    TORCH_CHECK(cos_sin_cache.dim() == 3 && cos_sin_cache.size(1) == 1 && cos_sin_cache.size(2) == query.size(2),
                "cos_sin_cache must be [max_position, 1, head_dim]");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
    auto q_out = torch::empty_like(query);
    auto k_out = torch::empty_like(key);
    int tokens = static_cast<int>(query.size(0));
    int q_heads = static_cast<int>(query.size(1));
    int k_heads = static_cast<int>(key.size(1));
    int head_dim = static_cast<int>(query.size(2));
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, query.scalar_type(), "rotary_embedding_cuda", [&] {
        int64_t q_work = static_cast<int64_t>(tokens) * q_heads * (head_dim / 2);
        int64_t k_work = static_cast<int64_t>(tokens) * k_heads * (head_dim / 2);
        rotary_one_kernel<scalar_t><<<blocks_for(q_work), kBlockSize, 0, stream>>>(
            positions.data_ptr<int64_t>(),
            query.data_ptr<scalar_t>(),
            q_out.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<float>(),
            tokens,
            q_heads,
            head_dim);
        rotary_one_kernel<scalar_t><<<blocks_for(k_work), kBlockSize, 0, stream>>>(
            positions.data_ptr<int64_t>(),
            key.data_ptr<scalar_t>(),
            k_out.data_ptr<scalar_t>(),
            cos_sin_cache.data_ptr<float>(),
            tokens,
            k_heads,
            head_dim);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {q_out, k_out};
}

void store_kvcache(torch::Tensor key, torch::Tensor value, torch::Tensor k_cache, torch::Tensor v_cache, torch::Tensor slot_mapping) {
    check_cuda(key, "key");
    check_cuda(value, "value");
    check_cuda(k_cache, "k_cache");
    check_cuda(v_cache, "v_cache");
    check_cuda(slot_mapping, "slot_mapping");
    check_contiguous(key, "key");
    check_contiguous(value, "value");
    TORCH_CHECK(key.sizes() == value.sizes(), "key/value shape mismatch");
    TORCH_CHECK(key.scalar_type() == value.scalar_type(), "key/value dtype mismatch");
    TORCH_CHECK(key.scalar_type() == k_cache.scalar_type() && key.scalar_type() == v_cache.scalar_type(), "KV cache dtype mismatch");
    TORCH_CHECK(key.dim() == 3, "key/value must be [tokens, heads, head_dim]");
    TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "KV cache tensors must be contiguous");
    TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
    TORCH_CHECK(slot_mapping.scalar_type() == torch::kInt32, "slot_mapping must be int32");
    TORCH_CHECK(slot_mapping.numel() == key.size(0), "slot_mapping length mismatch");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(key));
    int tokens = static_cast<int>(key.size(0));
    int width = static_cast<int>(key.size(1) * key.size(2));
    int64_t total = static_cast<int64_t>(tokens) * width;
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, key.scalar_type(), "store_kvcache_cuda", [&] {
        store_kvcache_kernel<scalar_t><<<blocks_for(total), kBlockSize, 0, stream>>>(
            key.data_ptr<scalar_t>(),
            value.data_ptr<scalar_t>(),
            k_cache.data_ptr<scalar_t>(),
            v_cache.data_ptr<scalar_t>(),
            slot_mapping.data_ptr<int32_t>(),
            tokens,
            width);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
