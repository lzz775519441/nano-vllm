import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nanovllm import cuda_ops


def _available():
    return torch.cuda.is_available() and cuda_ops.HAS_CUDA_OPS


def test_rms_norm_matches_torch():
    if not _available():
        return
    x = torch.randn(17, 128, device="cuda", dtype=torch.float16)
    weight = torch.randn(128, device="cuda", dtype=torch.float16)
    out = cuda_ops.rms_norm(x, weight, 1e-6)

    ref = x.float()
    ref = ref * torch.rsqrt(ref.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = ref.to(x.dtype) * weight
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


def test_add_rms_norm_matches_torch():
    if not _available():
        return
    x = torch.randn(9, 256, device="cuda", dtype=torch.float16)
    residual = torch.randn_like(x)
    weight = torch.randn(256, device="cuda", dtype=torch.float16)
    out, new_residual = cuda_ops.add_rms_norm(x, residual, weight, 1e-6)

    ref_residual = (x.float() + residual.float()).to(x.dtype)
    ref = x.float() + residual.float()
    ref = ref * torch.rsqrt(ref.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    ref = ref.to(x.dtype) * weight
    torch.testing.assert_close(new_residual, ref_residual, rtol=0, atol=0)
    torch.testing.assert_close(out, ref, rtol=2e-3, atol=2e-3)


def test_rotary_embedding_matches_torch():
    if not _available():
        return
    tokens, q_heads, k_heads, head_dim = 11, 32, 8, 128
    positions_storage = torch.empty(tokens * 2, device="cuda", dtype=torch.int64)
    positions = positions_storage[::2]
    positions.copy_(torch.arange(tokens, device="cuda", dtype=torch.int64))
    query_storage = torch.randn(tokens, q_heads + 1, head_dim * 2, device="cuda", dtype=torch.bfloat16)
    key_storage = torch.randn(tokens, k_heads + 1, head_dim * 2, device="cuda", dtype=torch.bfloat16)
    query = query_storage[:, :q_heads, ::2]
    key = key_storage[:, :k_heads, ::2]
    freqs = torch.randn(64, head_dim // 2, device="cuda")
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze(1)[::2]
    assert not positions.is_contiguous()
    assert not query.is_contiguous()
    assert not key.is_contiguous()
    assert not cos_sin_cache.is_contiguous()

    q, k = cuda_ops.rotary_embedding(positions, query, key, cos_sin_cache)
    cos, sin = cos_sin_cache[positions].chunk(2, dim=-1)

    def ref(x):
        x1, x2 = x.float().chunk(2, dim=-1)
        return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1).to(x.dtype)

    torch.testing.assert_close(q, ref(query), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(k, ref(key), rtol=2e-2, atol=2e-2)


def test_store_kvcache_matches_reference():
    if not _available():
        return
    tokens, block_size, heads, head_dim = 6, 16, 8, 128
    key = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    value_storage = torch.randn(tokens, heads + 1, head_dim, device="cuda", dtype=torch.bfloat16)
    value = value_storage[:, :heads, :]
    assert key.stride(0) != value.stride(0)
    k_cache = torch.zeros(3, block_size, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    slot_mapping = torch.tensor([0, 5, -1, 7, 8, 10], device="cuda", dtype=torch.int32)

    ref_k = k_cache.clone()
    ref_v = v_cache.clone()
    valid = slot_mapping >= 0
    ref_k.view(-1, heads, head_dim)[slot_mapping[valid].long()] = key[valid]
    ref_v.view(-1, heads, head_dim)[slot_mapping[valid].long()] = value[valid]

    cuda_ops.store_kvcache(key, value, k_cache, v_cache, slot_mapping)
    torch.testing.assert_close(k_cache, ref_k, rtol=0, atol=0)
    torch.testing.assert_close(v_cache, ref_v, rtol=0, atol=0)


def test_sample_shape_and_range():
    if not _available():
        return
    logits = torch.randn(8, 1024, device="cuda", dtype=torch.float16)
    temperatures = torch.full((8,), 0.7, device="cuda")
    tokens = cuda_ops.sample(logits, temperatures)
    seeded_tokens = cuda_ops._C.sample(logits, temperatures, 1234)
    seeded_tokens_again = cuda_ops._C.sample(logits, temperatures, 1234)
    assert tokens.shape == (8,)
    assert tokens.dtype == torch.int64
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < logits.size(1)
    torch.testing.assert_close(seeded_tokens, seeded_tokens_again, rtol=0, atol=0)


if __name__ == "__main__":
    if not _available():
        print("CUDA extension is not available; skipped")
    else:
        test_rms_norm_matches_torch()
        test_add_rms_norm_matches_torch()
        test_rotary_embedding_matches_torch()
        test_store_kvcache_matches_reference()
        test_sample_shape_and_range()
        print("CUDA op tests passed")
