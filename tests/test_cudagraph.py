import importlib
import sys
import types
import unittest

import torch

from nanovllm.engine.cudagraph import BatchDescriptor, CUDAGraphDispatcher, RuntimeMode
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, set_decode_context, set_varlen_context


class CUDAGraphDispatcherTest(unittest.TestCase):

    def make_seq(self, *, is_prefill: bool, scheduled: int, cached: int = 0):
        seq = Sequence([1, 2, 3, 4])
        seq.is_prefill = is_prefill
        seq.num_scheduled_tokens = scheduled
        seq.num_cached_tokens = cached
        return seq

    def test_full_and_piecewise_dispatches_decode_to_full_graph(self):
        seqs = [self.make_seq(is_prefill=False, scheduled=1, cached=4) for _ in range(3)]
        descriptor = BatchDescriptor.from_sequences(seqs)

        self.assertTrue(descriptor.is_uniform_decode)
        self.assertEqual(CUDAGraphDispatcher("full_and_piecewise").dispatch(descriptor), RuntimeMode.FULL_DECODE)

    def test_full_and_piecewise_dispatches_mixed_to_piecewise(self):
        seqs = [
            self.make_seq(is_prefill=False, scheduled=1, cached=4),
            self.make_seq(is_prefill=True, scheduled=3, cached=2),
        ]
        descriptor = BatchDescriptor.from_sequences(seqs)

        self.assertFalse(descriptor.is_uniform_decode)
        self.assertEqual(descriptor.num_batched_tokens, 4)
        self.assertEqual(CUDAGraphDispatcher("full_and_piecewise").dispatch(descriptor), RuntimeMode.PIECEWISE)

    def test_full_decode_only_downgrades_prefill_to_eager(self):
        seqs = [self.make_seq(is_prefill=True, scheduled=4, cached=0)]
        descriptor = BatchDescriptor.from_sequences(seqs)

        self.assertEqual(CUDAGraphDispatcher("full_decode_only").dispatch(descriptor), RuntimeMode.NONE)


class AttentionModeTest(unittest.TestCase):

    def setUp(self):
        self.calls = []
        fake_flash_attn = types.ModuleType("flash_attn")

        def fake_varlen(q, k, v, **kwargs):
            self.calls.append(("varlen", kwargs))
            return q

        def fake_kvcache(q, k_cache, v_cache, k=None, v=None, **kwargs):
            self.calls.append(("decode", k is not None, v is not None, kwargs))
            return q

        fake_flash_attn.flash_attn_varlen_func = fake_varlen
        fake_flash_attn.flash_attn_with_kvcache = fake_kvcache
        sys.modules["flash_attn"] = fake_flash_attn
        sys.modules.pop("nanovllm.layers.attention", None)
        self.attention_mod = importlib.import_module("nanovllm.layers.attention")

    def tearDown(self):
        reset_context()
        sys.modules.pop("nanovllm.layers.attention", None)
        sys.modules.pop("flash_attn", None)

    def test_varlen_context_uses_varlen_attention(self):
        attn = self.attention_mod.Attention(1, 4, 1.0, 1)
        q = k = v = torch.zeros(2, 1, 4)
        cu = torch.tensor([0, 2], dtype=torch.int32)
        set_varlen_context(cu, cu, 2, 2, torch.tensor([0, 1], dtype=torch.int32))

        out = attn(q, k, v)

        self.assertIs(out, q)
        self.assertEqual(self.calls[0][0], "varlen")

    def test_decode_context_uses_kvcache_attention_with_kv_update(self):
        attn = self.attention_mod.Attention(1, 4, 1.0, 1)
        q = k = v = torch.zeros(2, 1, 4)
        block_tables = torch.zeros(2, 1, dtype=torch.int32)
        set_decode_context(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([4, 5], dtype=torch.int32),
            block_tables,
        )

        out = attn(q, k, v)

        self.assertEqual(out.shape, q.shape)
        self.assertEqual(self.calls[0][0], "decode")
        self.assertTrue(self.calls[0][1])
        self.assertTrue(self.calls[0][2])

    def test_mixed_layout_still_uses_single_varlen_attention(self):
        store_calls = []
        self.attention_mod.store_kvcache = lambda *args: store_calls.append(args)
        attn = self.attention_mod.Attention(1, 4, 1.0, 1)
        attn.k_cache = torch.zeros(1)
        attn.v_cache = torch.zeros(1)
        q = k = v = torch.zeros(3, 1, 4)
        cu = torch.tensor([0, 1, 3], dtype=torch.int32)
        block_tables = torch.zeros(2, 1, dtype=torch.int32)
        slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
        set_varlen_context(
            cu,
            torch.tensor([0, 4, 6], dtype=torch.int32),
            2,
            6,
            slot_mapping,
            block_tables,
        )

        out = attn(q, k, v)

        self.assertIs(out, q)
        self.assertEqual([call[0] for call in self.calls], ["varlen"])
        self.assertEqual(self.calls[0][1]["cu_seqlens_q"].tolist(), [0, 1, 3])
        self.assertEqual(len(store_calls), 1)
        self.assertEqual(store_calls[0][-1].tolist(), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
