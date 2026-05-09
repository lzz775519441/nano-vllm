import pickle
import warnings
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen2_moe import Qwen2MoeForCausalLM
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_decode_context, set_varlen_context, set_context_obj, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        if getattr(hf_config, "model_type", None) == "qwen2_moe":
            if config.tensor_parallel_size != 1:
                raise NotImplementedError("qwen2_moe currently supports tensor_parallel_size=1 only")
            if not config.enforce_eager:
                warnings.warn("qwen2_moe currently runs in eager mode only; forcing enforce_eager=True", RuntimeWarning)
                config.enforce_eager = True
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = self.load_model_class(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.graph_pool = None
        self.full_decode_graphs = {}
        self.piecewise_graphs = {}
        self.cudagraph_vars = {}
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    @staticmethod
    def load_model_class(hf_config):
        model_type = getattr(hf_config, "model_type", None)
        if model_type == "qwen2_moe":
            return Qwen2MoeForCausalLM(hf_config)
        if model_type == "qwen3":
            return Qwen3ForCausalLM(hf_config)
        raise NotImplementedError(f"Unsupported model_type: {model_type}")

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graph_pool, self.full_decode_graphs, self.piecewise_graphs
            del self.cudagraph_vars
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, False)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        num_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert num_blocks > 1
        config.num_kvcache_blocks = num_blocks - 1
        self.dummy_block_id = config.num_kvcache_blocks
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, num_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        pad_block = getattr(self, "dummy_block_id", 0)
        block_tables = [seq.block_table + [pad_block] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def get_slot_mapping(self, seq: Sequence, start: int, end: int) -> list[int]:
        slot_mapping = []
        start_block = start // self.block_size
        end_block = (end + self.block_size - 1) // self.block_size
        for i in range(start_block, end_block):
            slot_start = seq.block_table[i] * self.block_size
            if i == start_block:
                slot_start += start % self.block_size
            if i != end_block - 1:
                slot_end = seq.block_table[i] * self.block_size + self.block_size
            else:
                slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
            slot_mapping.extend(range(slot_start, slot_end))
        return slot_mapping

    def prepare_mixed(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        sample_indices = []
        seen_prefill = False
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            if seq.is_prefill:
                seen_prefill = True
                input_ids.extend(seq[start:end])
            else:
                assert not seen_prefill
                assert seqlen_q == 1
                input_ids.append(seq.last_token)
            positions.extend(range(start, end))
            if seq.needs_sampling:
                sample_indices.append(cu_seqlens_q[-1] + seqlen_q - 1)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if seq.block_table:
                slot_mapping.extend(self.get_slot_mapping(seq, start, end))
        if any(seq.block_table for seq in seqs):
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sample_indices = torch.tensor(sample_indices, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        set_varlen_context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, block_tables)
        return input_ids, positions, sample_indices

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        cache_seqlens = []
        for seq in seqs:
            assert not seq.is_prefill and seq.num_scheduled_tokens == 1
            start = seq.num_cached_tokens
            end = start + 1
            input_ids.append(seq.last_token)
            positions.append(start)
            slot_mapping.extend(self.get_slot_mapping(seq, start, end))
            cache_seqlens.append(start)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cache_seqlens = torch.tensor(cache_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        sample_indices = torch.arange(len(seqs), dtype=torch.int64, device="cuda")
        block_tables = self.prepare_block_tables(seqs)
        set_decode_context(slot_mapping, cache_seqlens, block_tables)
        return input_ids, positions, sample_indices

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs if seq.needs_sampling]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @staticmethod
    def _round_bucket(n: int) -> int:
        if n <= 8:
            return 1 << (n - 1).bit_length()
        return (n + 15) // 16 * 16

    def get_cudagraph_capture_sizes(self, limit: int) -> list[int]:
        if limit <= 0:
            return []
        sizes = [1, 2, 4, 8]
        size = 16
        while size <= limit:
            sizes.append(size)
            size += 16
        buckets = {self._round_bucket(size) for size in sizes if size <= limit}
        return sorted(buckets)

    def get_cudagraph_vars(self, token_bucket: int) -> dict[str, torch.Tensor]:
        if token_bucket in self.cudagraph_vars:
            return self.cudagraph_vars[token_bucket]
        config = self.config
        seq_max = max(config.max_num_seqs, token_bucket)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        graph_vars = dict(
            input_ids=torch.zeros(token_bucket, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(token_bucket, dtype=torch.int64, device="cuda"),
            slot_mapping=torch.full((token_bucket,), -1, dtype=torch.int32, device="cuda"),
            cache_seqlens=torch.zeros(token_bucket, dtype=torch.int32, device="cuda"),
            block_tables=torch.full((seq_max, max_num_blocks), self.dummy_block_id, dtype=torch.int32, device="cuda"),
        )
        self.cudagraph_vars[token_bucket] = graph_vars
        return graph_vars

    def prepare_cudagraph_inputs(self, graph_vars: dict[str, torch.Tensor], input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        actual_tokens = input_ids.size(0)
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["input_ids"][:actual_tokens].copy_(input_ids)
        graph_vars["positions"][:actual_tokens].copy_(positions)
        graph_vars["slot_mapping"][:actual_tokens].copy_(context.slot_mapping)

        if context.attn_mode == "decode":
            graph_vars["cache_seqlens"].zero_()
            graph_vars["block_tables"].fill_(self.dummy_block_id)
            graph_vars["cache_seqlens"][:actual_tokens].copy_(context.cache_seqlens)
            graph_vars["block_tables"][:actual_tokens, :context.block_tables.size(1)].copy_(context.block_tables)

    def capture_full_decode_graph(self, token_bucket: int, graph_vars: dict[str, torch.Tensor]):
        if token_bucket in self.full_decode_graphs:
            return self.full_decode_graphs[token_bucket]
        hf_config = self.config.hf_config
        output_buf = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        graph = torch.cuda.CUDAGraph()
        set_decode_context(
            graph_vars["slot_mapping"],
            graph_vars["cache_seqlens"],
            graph_vars["block_tables"][:token_bucket],
        )
        hidden_states = self.model(graph_vars["input_ids"], graph_vars["positions"])
        output_buf.copy_(hidden_states)
        with torch.cuda.graph(graph, self.graph_pool):
            hidden_states = self.model(graph_vars["input_ids"], graph_vars["positions"])
            output_buf.copy_(hidden_states)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.full_decode_graphs[token_bucket] = (graph, output_buf)
        return self.full_decode_graphs[token_bucket]

    def capture_first_piece_graph(self, token_bucket: int, graph_vars: dict[str, torch.Tensor]):
        graph_key = ("first_piece", token_bucket)
        if graph_key in self.piecewise_graphs:
            return self.piecewise_graphs[graph_key]
        hf_config = self.config.hf_config
        attn = self.model.model.layers[0].self_attn
        q_buf = torch.empty(token_bucket, attn.num_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        k_buf = torch.empty(token_bucket, attn.num_kv_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        v_buf = torch.empty(token_bucket, attn.num_kv_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        residual_buf = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        graph = torch.cuda.CUDAGraph()
        q, k, v, residual = self.model.model.first_piece(graph_vars["input_ids"], graph_vars["positions"])
        q_buf.copy_(q)
        k_buf.copy_(k)
        v_buf.copy_(v)
        residual_buf.copy_(residual)
        with torch.cuda.graph(graph, self.graph_pool):
            q, k, v, residual = self.model.model.first_piece(graph_vars["input_ids"], graph_vars["positions"])
            q_buf.copy_(q)
            k_buf.copy_(k)
            v_buf.copy_(v)
            residual_buf.copy_(residual)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.piecewise_graphs[graph_key] = (graph, (q_buf, k_buf, v_buf, residual_buf))
        return self.piecewise_graphs[graph_key]

    def capture_next_piece_graph(self, token_bucket: int, layer_idx: int, graph_vars: dict[str, torch.Tensor]):
        graph_key = ("next_piece", layer_idx, token_bucket)
        if graph_key in self.piecewise_graphs:
            return self.piecewise_graphs[graph_key]
        hf_config = self.config.hf_config
        attn = self.model.model.layers[layer_idx].self_attn
        attn_input = torch.empty(token_bucket, attn.num_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        residual_input = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        attn_input.zero_()
        residual_input.zero_()
        q_buf = torch.empty(token_bucket, attn.num_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        k_buf = torch.empty(token_bucket, attn.num_kv_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        v_buf = torch.empty(token_bucket, attn.num_kv_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        residual_buf = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        graph = torch.cuda.CUDAGraph()
        q, k, v, residual = self.model.model.next_piece(layer_idx, attn_input, residual_input, graph_vars["positions"])
        q_buf.copy_(q)
        k_buf.copy_(k)
        v_buf.copy_(v)
        residual_buf.copy_(residual)
        with torch.cuda.graph(graph, self.graph_pool):
            q, k, v, residual = self.model.model.next_piece(layer_idx, attn_input, residual_input, graph_vars["positions"])
            q_buf.copy_(q)
            k_buf.copy_(k)
            v_buf.copy_(v)
            residual_buf.copy_(residual)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.piecewise_graphs[graph_key] = (graph, attn_input, residual_input, (q_buf, k_buf, v_buf, residual_buf))
        return self.piecewise_graphs[graph_key]

    def capture_last_piece_graph(self, token_bucket: int, layer_idx: int, graph_vars: dict[str, torch.Tensor]):
        graph_key = ("last_piece", layer_idx, token_bucket)
        if graph_key in self.piecewise_graphs:
            return self.piecewise_graphs[graph_key]
        hf_config = self.config.hf_config
        attn = self.model.model.layers[layer_idx].self_attn
        attn_input = torch.empty(token_bucket, attn.num_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        residual_input = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        attn_input.zero_()
        residual_input.zero_()
        output_buf = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        graph = torch.cuda.CUDAGraph()
        hidden_states = self.model.model.final_piece(layer_idx, attn_input, residual_input)
        output_buf.copy_(hidden_states)
        with torch.cuda.graph(graph, self.graph_pool):
            hidden_states = self.model.model.final_piece(layer_idx, attn_input, residual_input)
            output_buf.copy_(hidden_states)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.piecewise_graphs[graph_key] = (graph, attn_input, residual_input, output_buf)
        return self.piecewise_graphs[graph_key]

    def capture_piecewise_graphs(self, token_bucket: int, graph_vars: dict[str, torch.Tensor]):
        self.capture_first_piece_graph(token_bucket, graph_vars)
        num_layers = len(self.model.model.layers)
        for layer_idx in range(num_layers - 1):
            self.capture_next_piece_graph(token_bucket, layer_idx, graph_vars)
        self.capture_last_piece_graph(token_bucket, num_layers - 1, graph_vars)

    def has_piecewise_graphs(self, token_bucket: int) -> bool:
        if not hasattr(self, 'piecewise_graphs'):
            return False
        num_layers = len(self.model.model.layers)
        if ("first_piece", token_bucket) not in self.piecewise_graphs:
            return False
        for layer_idx in range(num_layers - 1):
            if ("next_piece", layer_idx, token_bucket) not in self.piecewise_graphs:
                return False
        return ("last_piece", num_layers - 1, token_bucket) in self.piecewise_graphs

    def run_full_decode_graph(self, input_ids: torch.Tensor, positions: torch.Tensor):
        actual_context = get_context()
        token_bucket = self._round_bucket(input_ids.size(0))
        if token_bucket not in self.full_decode_graphs:
            return None
        graph_vars = self.cudagraph_vars[token_bucket]
        try:
            self.prepare_cudagraph_inputs(graph_vars, input_ids, positions)
            graph, hidden_states = self.full_decode_graphs[token_bucket]
            set_decode_context(
                graph_vars["slot_mapping"],
                graph_vars["cache_seqlens"],
                graph_vars["block_tables"][:token_bucket],
            )
            graph.replay()
        except Exception as exc:
            set_context_obj(actual_context)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                self.full_decode_graphs.pop(token_bucket, None)
                torch.cuda.empty_cache()
            warnings.warn(f"Full decode CUDA graph disabled for token_bucket={token_bucket}: {exc}", RuntimeWarning)
            return None
        return hidden_states

    def run_piecewise_graph(self, input_ids: torch.Tensor, positions: torch.Tensor):
        actual_context = get_context()
        actual_tokens = input_ids.size(0)
        token_bucket = self._round_bucket(actual_tokens)
        if not self.has_piecewise_graphs(token_bucket):
            return None
        graph_vars = self.cudagraph_vars[token_bucket]
        try:
            self.prepare_cudagraph_inputs(graph_vars, input_ids, positions)
            graph, outputs = self.piecewise_graphs[("first_piece", token_bucket)]
            graph.replay()
            q, k, v, residual = outputs
            num_layers = len(self.model.model.layers)
            for layer_idx, layer in enumerate(self.model.model.layers):
                attn_output = layer.attention(q[:actual_tokens], k[:actual_tokens], v[:actual_tokens])
                if layer_idx == num_layers - 1:
                    graph, attn_input, residual_input, hidden_states = self.piecewise_graphs[("last_piece", layer_idx, token_bucket)]
                    attn_input.zero_()
                    residual_input.zero_()
                    attn_input[:actual_tokens].copy_(attn_output)
                    residual_input[:actual_tokens].copy_(residual[:actual_tokens])
                    graph.replay()
                else:
                    graph, attn_input, residual_input, outputs = self.piecewise_graphs[("next_piece", layer_idx, token_bucket)]
                    attn_input.zero_()
                    residual_input.zero_()
                    attn_input[:actual_tokens].copy_(attn_output)
                    residual_input[:actual_tokens].copy_(residual[:actual_tokens])
                    graph.replay()
                    q, k, v, residual = outputs
        except Exception as exc:
            set_context_obj(actual_context)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                graph_keys = [graph_key for graph_key in self.piecewise_graphs if graph_key[-1] == token_bucket]
                for graph_key in graph_keys:
                    del self.piecewise_graphs[graph_key]
                self.full_decode_graphs.pop(token_bucket, None)
                self.cudagraph_vars.pop(token_bucket, None)
                torch.cuda.empty_cache()
            warnings.warn(f"Piecewise CUDA graph disabled for token_bucket={token_bucket}: {exc}", RuntimeWarning)
            return None
        return hidden_states

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, sample_indices: torch.Tensor, is_decode: bool):
        if self.enforce_eager:
            hidden_states = self.model(input_ids, positions)
        elif is_decode:
            hidden_states = self.run_full_decode_graph(input_ids, positions)
            if hidden_states is None:
                hidden_states = self.model(input_ids, positions)
        else:
            hidden_states = self.run_piecewise_graph(input_ids, positions)
            if hidden_states is None:
                hidden_states = self.model(input_ids, positions)

        if sample_indices.numel() == 0:
            return None
        return self.model.compute_logits(hidden_states.index_select(0, sample_indices))

    def run(self, seqs: list[Sequence], is_decode: bool) -> list[int]:
        if is_decode:
            input_ids, positions, sample_indices = self.prepare_decode(seqs)
        else:
            input_ids, positions, sample_indices = self.prepare_mixed(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, sample_indices, is_decode)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 and logits is not None else []
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        self.graph_pool = None
        self.full_decode_graphs = {}
        self.piecewise_graphs = {}
        self.cudagraph_vars = {}
        for token_bucket in self.get_cudagraph_capture_sizes(self.config.max_decode_cudagraph_tokens):
            graph_vars = self.get_cudagraph_vars(token_bucket)
            try:
                self.capture_full_decode_graph(token_bucket, graph_vars)
            except torch.cuda.OutOfMemoryError:
                self.full_decode_graphs.pop(token_bucket, None)
                self.cudagraph_vars.pop(token_bucket, None)
                torch.cuda.empty_cache()
                warnings.warn(f"Full decode CUDA graph capture stopped at token_bucket={token_bucket}: out of memory", RuntimeWarning)
                break

        for token_bucket in self.get_cudagraph_capture_sizes(self.config.max_piecewise_cudagraph_tokens):
            graph_vars = self.get_cudagraph_vars(token_bucket)
            try:
                self.capture_piecewise_graphs(token_bucket, graph_vars)
            except torch.cuda.OutOfMemoryError:
                graph_keys = [graph_key for graph_key in self.piecewise_graphs if graph_key[-1] == token_bucket]
                for graph_key in graph_keys:
                    del self.piecewise_graphs[graph_key]
                if token_bucket not in self.full_decode_graphs:
                    self.cudagraph_vars.pop(token_bucket, None)
                torch.cuda.empty_cache()
                warnings.warn(f"Piecewise CUDA graph capture stopped at token_bucket={token_bucket}: out of memory", RuntimeWarning)
                break
