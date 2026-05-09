import importlib
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist

from nanovllm.utils.context import reset_context, set_varlen_context


_OLD_FLASH_ATTN = None
_HAD_FLASH_ATTN = False
_DIST_DIR = None
_OWNS_PROCESS_GROUP = False
qwen2_moe = None


def setUpModule():
    global _OLD_FLASH_ATTN, _HAD_FLASH_ATTN, _DIST_DIR, _OWNS_PROCESS_GROUP, qwen2_moe
    _HAD_FLASH_ATTN = "flash_attn" in sys.modules
    _OLD_FLASH_ATTN = sys.modules.get("flash_attn")

    fake_flash_attn = types.ModuleType("flash_attn")

    def fake_varlen(q, k, v, **kwargs):
        return q

    def fake_kvcache(q, k_cache, v_cache, k=None, v=None, **kwargs):
        return q

    fake_flash_attn.flash_attn_varlen_func = fake_varlen
    fake_flash_attn.flash_attn_with_kvcache = fake_kvcache
    sys.modules["flash_attn"] = fake_flash_attn
    sys.modules.pop("nanovllm.layers.attention", None)
    sys.modules.pop("nanovllm.models.qwen2_moe", None)

    if dist.is_available() and not dist.is_initialized():
        _DIST_DIR = tempfile.TemporaryDirectory()
        init_file = os.path.join(_DIST_DIR.name, "dist_init")
        dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=0, world_size=1)
        _OWNS_PROCESS_GROUP = True

    qwen2_moe = importlib.import_module("nanovllm.models.qwen2_moe")


def tearDownModule():
    reset_context()
    sys.modules.pop("nanovllm.models.qwen2_moe", None)
    sys.modules.pop("nanovllm.layers.attention", None)
    if _HAD_FLASH_ATTN:
        sys.modules["flash_attn"] = _OLD_FLASH_ATTN
    else:
        sys.modules.pop("flash_attn", None)
    if _OWNS_PROCESS_GROUP and dist.is_initialized():
        dist.destroy_process_group()
    if _DIST_DIR is not None:
        _DIST_DIR.cleanup()


def tiny_config(**overrides):
    config = dict(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        decoder_sparse_step=1,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=10,
        num_experts_per_tok=2,
        num_experts=3,
        norm_topk_prob=False,
        mlp_only_layers=[],
        model_type="qwen2_moe",
        dtype=torch.float32,
    )
    config.update(overrides)
    return SimpleNamespace(**config)


class Qwen2MoeTest(unittest.TestCase):

    def tearDown(self):
        reset_context()

    def test_sparse_moe_block_shape_with_and_without_topk_norm(self):
        x = torch.randn(5, 8)
        for norm_topk_prob in (False, True):
            block = qwen2_moe.Qwen2MoeSparseMoeBlock(tiny_config(norm_topk_prob=norm_topk_prob))

            out = block(x)

            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, x.dtype)

    def test_shared_expert_branch_participates_in_output(self):
        block = qwen2_moe.Qwen2MoeSparseMoeBlock(tiny_config(num_experts_per_tok=1))
        x = torch.ones(3, 8)
        with torch.no_grad():
            for expert in block.experts:
                expert.gate_proj.weight.zero_()
                expert.up_proj.weight.zero_()
                expert.down_proj.weight.zero_()
            block.shared_expert.gate_proj.weight.fill_(0.2)
            block.shared_expert.up_proj.weight.fill_(0.2)
            block.shared_expert.down_proj.weight.fill_(0.2)
            block.shared_expert_gate.weight.fill_(2.0)
            enabled = block(x)
            block.shared_expert_gate.weight.fill_(-2.0)
            disabled = block(x)

        self.assertTrue(hasattr(block, "shared_expert_gate"))
        self.assertFalse(torch.allclose(enabled, disabled))

    def test_decoder_layer_selects_sparse_or_dense_mlp(self):
        sparse = qwen2_moe.Qwen2MoeDecoderLayer(tiny_config(decoder_sparse_step=1), layer_idx=0)
        dense_by_step = qwen2_moe.Qwen2MoeDecoderLayer(tiny_config(decoder_sparse_step=2), layer_idx=0)
        dense_by_override = qwen2_moe.Qwen2MoeDecoderLayer(
            tiny_config(decoder_sparse_step=1, mlp_only_layers=[0]),
            layer_idx=0,
        )

        self.assertIsInstance(sparse.mlp, qwen2_moe.Qwen2MoeSparseMoeBlock)
        self.assertIsInstance(dense_by_step.mlp, qwen2_moe.Qwen2MoeMLP)
        self.assertIsInstance(dense_by_override.mlp, qwen2_moe.Qwen2MoeMLP)

    def test_tiny_model_forward_with_fake_attention(self):
        model = qwen2_moe.Qwen2MoeForCausalLM(tiny_config(num_hidden_layers=1))
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        positions = torch.arange(input_ids.numel(), dtype=torch.int64)
        cu = torch.tensor([0, input_ids.numel()], dtype=torch.int32)
        set_varlen_context(cu, cu, input_ids.numel(), input_ids.numel(), torch.arange(input_ids.numel(), dtype=torch.int32))

        with torch.inference_mode():
            out = model(input_ids, positions)

        self.assertEqual(out.shape, (input_ids.numel(), 8))

    def test_loader_packs_attention_but_not_expert_mlp_weights(self):
        try:
            from safetensors.torch import save_file
        except ImportError:
            self.skipTest("safetensors is not installed")
        from nanovllm.utils.loader import load_model

        model = qwen2_moe.Qwen2MoeForCausalLM(tiny_config(num_hidden_layers=1))
        q = torch.full((8, 8), 1.0)
        k = torch.full((8, 8), 2.0)
        v = torch.full((8, 8), 3.0)
        expert_gate = torch.full((6, 8), 4.0)
        expert_up = torch.full((6, 8), 5.0)
        shared_gate = torch.full((1, 8), 6.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(
                {
                    "model.layers.0.self_attn.q_proj.weight": q,
                    "model.layers.0.self_attn.k_proj.weight": k,
                    "model.layers.0.self_attn.v_proj.weight": v,
                    "model.layers.0.mlp.experts.0.gate_proj.weight": expert_gate,
                    "model.layers.0.mlp.experts.0.up_proj.weight": expert_up,
                    "model.layers.0.mlp.shared_expert_gate.weight": shared_gate,
                },
                os.path.join(tmpdir, "model.safetensors"),
            )
            load_model(model, tmpdir)

        qkv = model.get_parameter("model.layers.0.self_attn.qkv_proj.weight")
        self.assertTrue(torch.equal(qkv[:8], q))
        self.assertTrue(torch.equal(qkv[8:16], k))
        self.assertTrue(torch.equal(qkv[16:24], v))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.mlp.experts.0.gate_proj.weight"), expert_gate))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.mlp.experts.0.up_proj.weight"), expert_up))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.mlp.shared_expert_gate.weight"), shared_gate))


if __name__ == "__main__":
    unittest.main()
