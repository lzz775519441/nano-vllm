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
_GPTQ_MODULES = [
    "gptqmodel",
    "gptqmodel.nn_modules",
    "gptqmodel.nn_modules.qlinear",
    "gptqmodel.nn_modules.qlinear.marlin",
]


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

    install_fake_gptqmodel()
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
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        moe_intermediate_size=6,
        shared_expert_intermediate_size=10,
        num_experts_per_tok=2,
        num_experts=3,
        norm_topk_prob=False,
        model_type="qwen2_moe",
        dtype=torch.float32,
        quantization_config={
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
            "desc_act": False,
            "sym": True,
        },
    )
    config.update(overrides)
    return SimpleNamespace(**config)


def clear_fake_gptqmodel():
    for name in _GPTQ_MODULES:
        sys.modules.pop(name, None)


def install_fake_gptqmodel():
    if "gptqmodel.nn_modules.qlinear.marlin" in sys.modules:
        return
    clear_fake_gptqmodel()
    sys.modules["gptqmodel"] = types.ModuleType("gptqmodel")
    sys.modules["gptqmodel.nn_modules"] = types.ModuleType("gptqmodel.nn_modules")
    sys.modules["gptqmodel.nn_modules.qlinear"] = types.ModuleType("gptqmodel.nn_modules.qlinear")
    marlin_mod = types.ModuleType("gptqmodel.nn_modules.qlinear.marlin")

    class FakeQuantLinear(torch.nn.Module):
        def __init__(self, bits, group_size, desc_act, sym, in_features, out_features, bias=False, **kwargs):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.bits = bits
            self.group_size = group_size
            self.desc_act = desc_act
            self.sym = sym
            self.register_parameter("qweight", torch.nn.Parameter(torch.empty(1, out_features, dtype=torch.int32), requires_grad=False))
            self.register_parameter("qzeros", torch.nn.Parameter(torch.empty(1, max(1, out_features // 8), dtype=torch.int32), requires_grad=False))
            self.register_parameter("scales", torch.nn.Parameter(torch.empty(1, out_features), requires_grad=False))
            self.register_parameter("g_idx", torch.nn.Parameter(torch.empty(in_features, dtype=torch.int32), requires_grad=False))
            if bias:
                self.register_buffer("bias", torch.empty(out_features))
            else:
                self.bias = None
            self.post_init_calls = 0

        def post_init(self):
            self.post_init_calls += 1

        def forward(self, x):
            return torch.zeros(x.shape[:-1] + (self.out_features,), dtype=x.dtype, device=x.device)

    class FakeMarlinLinear(FakeQuantLinear):
        pass

    marlin_mod.MarlinLinear = FakeMarlinLinear
    sys.modules["gptqmodel.nn_modules.qlinear.marlin"] = marlin_mod


class Qwen2MoeTest(unittest.TestCase):

    def tearDown(self):
        reset_context()

    def test_sparse_moe_block_shape_with_and_without_topk_norm(self):
        install_fake_gptqmodel()
        x = torch.randn(5, 8)
        for norm_topk_prob in (False, True):
            block = qwen2_moe.Qwen2MoeSparseMoeBlock(tiny_config(norm_topk_prob=norm_topk_prob))

            out = block(x)

            self.assertEqual(out.shape, x.shape)
            self.assertEqual(out.dtype, x.dtype)

    def test_shared_expert_branch_keeps_expected_modules(self):
        install_fake_gptqmodel()
        block = qwen2_moe.Qwen2MoeSparseMoeBlock(tiny_config(num_experts_per_tok=1))

        self.assertTrue(hasattr(block, "shared_expert_gate"))
        self.assertIsInstance(block.shared_expert.gate_proj, qwen2_moe.MarlinLinear)
        self.assertNotIsInstance(block.shared_expert_gate, qwen2_moe.MarlinLinear)

    def test_decoder_layer_uses_sparse_moe(self):
        install_fake_gptqmodel()
        layer = qwen2_moe.Qwen2MoeDecoderLayer(tiny_config(), layer_idx=0)

        self.assertIsInstance(layer.mlp, qwen2_moe.Qwen2MoeSparseMoeBlock)

    def test_tiny_model_forward_with_fake_attention(self):
        install_fake_gptqmodel()
        model = qwen2_moe.Qwen2MoeForCausalLM(tiny_config(num_hidden_layers=1))
        input_ids = torch.tensor([1, 2, 3], dtype=torch.int64)
        positions = torch.arange(input_ids.numel(), dtype=torch.int64)
        cu = torch.tensor([0, input_ids.numel()], dtype=torch.int32)
        set_varlen_context(cu, cu, input_ids.numel(), input_ids.numel(), torch.arange(input_ids.numel(), dtype=torch.int32))

        with torch.inference_mode():
            out = model(input_ids, positions)

        self.assertEqual(out.shape, (input_ids.numel(), 8))

    def test_model_uses_independent_quantized_projections(self):
        install_fake_gptqmodel()

        model = qwen2_moe.Qwen2MoeForCausalLM(tiny_config(num_hidden_layers=1))
        attn = model.model.layers[0].self_attn

        self.assertTrue(hasattr(attn, "q_proj"))
        self.assertTrue(hasattr(attn, "k_proj"))
        self.assertTrue(hasattr(attn, "v_proj"))
        self.assertFalse(hasattr(attn, "qkv_proj"))
        self.assertIsInstance(attn.q_proj, qwen2_moe.MarlinLinear)
        self.assertIsInstance(model.model.layers[0].mlp.experts[0].gate_proj, qwen2_moe.MarlinLinear)
        self.assertNotIsInstance(model.model.layers[0].mlp.gate, qwen2_moe.MarlinLinear)
        self.assertNotIsInstance(model.model.layers[0].mlp.shared_expert_gate, qwen2_moe.MarlinLinear)

    def test_gptq_loader_loads_quant_tensors_and_runs_post_init(self):
        from safetensors.torch import save_file
        from nanovllm.utils.loader import load_model

        install_fake_gptqmodel()
        model = qwen2_moe.Qwen2MoeForCausalLM(tiny_config(num_hidden_layers=1))
        qweight = torch.full_like(model.get_parameter("model.layers.0.self_attn.q_proj.qweight"), 7)
        qzeros = torch.full_like(model.get_parameter("model.layers.0.self_attn.q_proj.qzeros"), 3)
        scales = torch.full_like(model.get_parameter("model.layers.0.self_attn.q_proj.scales"), 0.5)
        g_idx = torch.arange(model.get_parameter("model.layers.0.self_attn.q_proj.g_idx").numel(), dtype=torch.int32)
        bias = torch.full_like(model.get_buffer("model.layers.0.self_attn.q_proj.bias"), 2.0)
        expert_qweight = torch.full_like(model.get_parameter("model.layers.0.mlp.experts.0.gate_proj.qweight"), 9)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_file(
                {
                    "model.layers.0.self_attn.q_proj.qweight": qweight,
                    "model.layers.0.self_attn.q_proj.qzeros": qzeros,
                    "model.layers.0.self_attn.q_proj.scales": scales,
                    "model.layers.0.self_attn.q_proj.g_idx": g_idx,
                    "model.layers.0.self_attn.q_proj.bias": bias,
                    "model.layers.0.mlp.experts.0.gate_proj.qweight": expert_qweight,
                },
                os.path.join(tmpdir, "model.safetensors"),
            )
            load_model(model, tmpdir)

        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.self_attn.q_proj.qweight"), qweight))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.self_attn.q_proj.qzeros"), qzeros))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.self_attn.q_proj.scales"), scales))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.self_attn.q_proj.g_idx"), g_idx))
        self.assertTrue(torch.equal(model.get_buffer("model.layers.0.self_attn.q_proj.bias"), bias))
        self.assertTrue(torch.equal(model.get_parameter("model.layers.0.mlp.experts.0.gate_proj.qweight"), expert_qweight))
        gptq_modules = [module for module in model.modules() if isinstance(module, qwen2_moe.MarlinLinear)]
        self.assertGreater(len(gptq_modules), 0)
        self.assertTrue(all(module.post_init_calls == 1 for module in gptq_modules))


if __name__ == "__main__":
    unittest.main()
