import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist

from nanovllm.layers.attention import Attention
from nanovllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nanovllm.layers.gptq import GPTQModelLinear, is_gptq_config
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import ColumnParallelLinear, QKVParallelLinear, ReplicatedLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope


class Qwen2MoeAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rope_theta: float = 1000000.0,
        qkv_bias: bool = True,
        quant_config=None,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.quantized = quant_config is not None

        if self.quantized:
            self.q_proj = GPTQModelLinear(quant_config, hidden_size, self.total_num_heads * self.head_dim, bias=qkv_bias)
            self.k_proj = GPTQModelLinear(quant_config, hidden_size, self.total_num_kv_heads * self.head_dim, bias=qkv_bias)
            self.v_proj = GPTQModelLinear(quant_config, hidden_size, self.total_num_kv_heads * self.head_dim, bias=qkv_bias)
            self.o_proj = GPTQModelLinear(quant_config, self.total_num_heads * self.head_dim, hidden_size, bias=True)
        else:
            self.qkv_proj = QKVParallelLinear(
                hidden_size,
                self.head_dim,
                self.total_num_heads,
                self.total_num_kv_heads,
                bias=qkv_bias,
            )
            self.o_proj = RowParallelLinear(
                self.total_num_heads * self.head_dim,
                hidden_size,
                bias=False,
            )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def pre_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.quantized:
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
        else:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v

    def post_attention(self, attn_output: torch.Tensor) -> torch.Tensor:
        return self.o_proj(attn_output.flatten(1, -1))

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(positions, hidden_states)
        o = self.attn(q, k, v)
        return self.post_attention(o)


class Qwen2MoeMLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config=None,
    ) -> None:
        super().__init__()
        if quant_config is not None:
            self.gate_proj = GPTQModelLinear(quant_config, hidden_size, intermediate_size, bias=True)
            self.up_proj = GPTQModelLinear(quant_config, hidden_size, intermediate_size, bias=True)
            self.down_proj = GPTQModelLinear(quant_config, intermediate_size, hidden_size, bias=True)
        else:
            self.gate_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
            self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=False)
            self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=False)
        assert hidden_act == "silu"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen2MoeSparseMoeBlock(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.norm_topk_prob = getattr(config, "norm_topk_prob", False)
        quant_config = config if is_gptq_config(config) else None
        self.gate = ReplicatedLinear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([
            Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.moe_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
            )
            for _ in range(config.num_experts)
        ])
        self.shared_expert = Qwen2MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.shared_expert_intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
        )
        self.shared_expert_gate = ReplicatedLinear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        router_logits = self.gate(hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (num_tokens, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        for expert_idx, expert_layer in enumerate(self.experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = hidden_states.index_select(0, top_x)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_output = self.shared_expert(hidden_states)
        shared_output = torch.sigmoid(self.shared_expert_gate(hidden_states)) * shared_output
        return final_hidden_states + shared_output


class Qwen2MoeDecoderLayer(nn.Module):

    def __init__(
        self,
        config,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2MoeAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=getattr(config, "rope_theta", 1000000.0),
            qkv_bias=getattr(config, "attention_bias", True),
            quant_config=config if is_gptq_config(config) else None,
        )
        mlp_only_layers = getattr(config, "mlp_only_layers", [])
        decoder_sparse_step = getattr(config, "decoder_sparse_step", 1)
        use_sparse_moe = (
            layer_idx not in mlp_only_layers
            and getattr(config, "num_experts", 0) > 0
            and (layer_idx + 1) % decoder_sparse_step == 0
        )
        if use_sparse_moe:
            self.mlp = Qwen2MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=config if is_gptq_config(config) else None,
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v, residual = self.pre_attention(positions, hidden_states, residual)
        attn_output = self.attention(q, k, v)
        return self.post_attention(attn_output, residual)

    def pre_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        q, k, v = self.self_attn.pre_attention(positions, hidden_states)
        return q, k, v, residual

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        return self.self_attn.attn(q, k, v)

    def post_attention(
        self,
        attn_output: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.self_attn.post_attention(attn_output)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2MoeModel(nn.Module):

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2MoeDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2MoeForCausalLM(nn.Module):
    qkv_packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(
        self,
        config,
    ) -> None:
        super().__init__()
        self.quantized = is_gptq_config(config)
        self.packed_modules_mapping = {} if self.quantized else self.qkv_packed_modules_mapping
        self.model = Qwen2MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
