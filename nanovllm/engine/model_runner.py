import gc
import pickle
import warnings
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.cudagraph import BatchDescriptor, CUDAGraphDispatcher, RuntimeMode
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_decode_context, set_varlen_context, set_context_obj, get_context, reset_context
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.dispatcher = CUDAGraphDispatcher("none" if config.enforce_eager else config.cudagraph_mode)
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
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

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graph_pool, self.full_decode_graphs, self.piecewise_graphs
            del self.piecewise_runtime_vars, self.disabled_graphs
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
        self.run(seqs)
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

    @staticmethod
    def _pad_lens(num_seqs: int, num_tokens: int, max_len: int) -> list[int]:
        if num_seqs == 0:
            assert num_tokens == 0
            return []
        base, extra = divmod(num_tokens, num_seqs)
        lens = [base + (i < extra) for i in range(num_seqs)]
        assert all(0 < x <= max_len for x in lens)
        return lens

    @staticmethod
    def _cu_seqlens(lens: list[int]) -> torch.Tensor:
        cu_seqlens = [0]
        for length in lens:
            cu_seqlens.append(cu_seqlens[-1] + length)
        return torch.tensor(cu_seqlens, dtype=torch.int32)

    def get_varlen_graph_key(self, input_ids: torch.Tensor):
        context = get_context()
        num_seqs = context.cu_seqlens_q.numel() - 1
        num_tokens = input_ids.size(0)
        seq_bucket = min(self.config.max_num_seqs, self._round_bucket(num_seqs))
        token_bucket = self._round_bucket(num_tokens)
        max_q_bucket = self._round_bucket(max(context.max_seqlen_q, 1))

        # Varlen attention expects cu_seqlens_q[-1] to match the static q size.
        # Extra q tokens therefore need their own padded sequence rows.
        pad_seqs = seq_bucket - num_seqs
        if pad_seqs == 0:
            token_bucket = num_tokens
        else:
            token_bucket = max(token_bucket, num_tokens + pad_seqs)
            pad_tokens = token_bucket - num_tokens
            if pad_tokens > pad_seqs * max_q_bucket:
                max_q_bucket = self._round_bucket((pad_tokens + pad_seqs - 1) // pad_seqs)

        max_k_bucket = self._round_bucket(max(context.max_seqlen_k, max_q_bucket, 1))
        return seq_bucket, token_bucket, max_q_bucket, max_k_bucket

    def get_decode_graph_key(self, input_ids: torch.Tensor):
        context = get_context()
        seq_bucket = min(self.config.max_num_seqs, self._round_bucket(input_ids.size(0)))
        block_table_bucket = self._round_bucket(context.block_tables.size(1))
        return seq_bucket, block_table_bucket

    def get_piecewise_runtime_vars(self, key: tuple[int, int, int, int]):
        if key in self.piecewise_runtime_vars:
            return self.piecewise_runtime_vars[key]
        config = self.config
        seq_bucket, token_bucket, max_q_bucket, max_k_bucket = key
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        graph_vars = dict(
            input_ids=torch.zeros(token_bucket, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(token_bucket, dtype=torch.int64, device="cuda"),
            slot_mapping=torch.full((token_bucket,), -1, dtype=torch.int32, device="cuda"),
            cu_seqlens_q=self._cu_seqlens(self._pad_lens(seq_bucket, token_bucket, max_q_bucket)).cuda(non_blocking=True),
            cu_seqlens_k=self._cu_seqlens(self._pad_lens(seq_bucket, token_bucket, max_q_bucket)).cuda(non_blocking=True),
            block_tables=torch.full((seq_bucket, max_num_blocks), self.dummy_block_id, dtype=torch.int32, device="cuda"),
            max_seqlen_q=max_q_bucket,
            max_seqlen_k=max_k_bucket,
        )
        self.piecewise_runtime_vars[key] = graph_vars
        return graph_vars

    def pad_piecewise_inputs(self, graph_vars: dict[str, torch.Tensor], input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        seq_bucket = graph_vars["block_tables"].size(0)
        token_bucket = graph_vars["input_ids"].size(0)
        actual_seqs = context.cu_seqlens_q.numel() - 1
        actual_tokens = input_ids.size(0)
        pad_seqs = seq_bucket - actual_seqs
        pad_tokens = token_bucket - actual_tokens
        pad_lens = self._pad_lens(pad_seqs, pad_tokens, graph_vars["max_seqlen_q"]) if pad_seqs else []

        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["block_tables"].fill_(self.dummy_block_id)
        graph_vars["input_ids"][:actual_tokens].copy_(input_ids)
        graph_vars["positions"][:actual_tokens].copy_(positions)
        graph_vars["slot_mapping"][:actual_tokens].copy_(context.slot_mapping)
        graph_vars["cu_seqlens_q"][:actual_seqs + 1].copy_(context.cu_seqlens_q)
        graph_vars["cu_seqlens_k"][:actual_seqs + 1].copy_(context.cu_seqlens_k)
        graph_vars["block_tables"][:actual_seqs, :context.block_tables.size(1)].copy_(context.block_tables)

        if pad_lens:
            pad_cu = self._cu_seqlens(pad_lens).cuda(non_blocking=True)[1:]
            graph_vars["cu_seqlens_q"][actual_seqs + 1:].copy_(actual_tokens + pad_cu)
            graph_vars["cu_seqlens_k"][actual_seqs + 1:].copy_(context.cu_seqlens_k[-1] + pad_cu)

        set_varlen_context(
            graph_vars["cu_seqlens_q"],
            graph_vars["cu_seqlens_k"],
            graph_vars["max_seqlen_q"],
            graph_vars["max_seqlen_k"],
            graph_vars["slot_mapping"],
            graph_vars["block_tables"],
        )

    def get_full_decode_vars(self, key: tuple[int, int]):
        if key in self.full_decode_graphs:
            return self.full_decode_graphs[key][1]
        config = self.config
        hf_config = config.hf_config
        seq_bucket, block_table_bucket = key
        graph_vars = dict(
            input_ids=torch.zeros(seq_bucket, dtype=torch.int64, device="cuda"),
            positions=torch.zeros(seq_bucket, dtype=torch.int64, device="cuda"),
            slot_mapping=torch.full((seq_bucket,), -1, dtype=torch.int32, device="cuda"),
            cache_seqlens=torch.zeros(seq_bucket, dtype=torch.int32, device="cuda"),
            block_tables=torch.full((seq_bucket, block_table_bucket), self.dummy_block_id, dtype=torch.int32, device="cuda"),
            outputs=torch.empty(seq_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda"),
        )
        return graph_vars

    def pad_full_decode_inputs(self, graph_vars: dict[str, torch.Tensor], input_ids: torch.Tensor, positions: torch.Tensor):
        context = get_context()
        bs = input_ids.size(0)
        graph_vars["input_ids"].zero_()
        graph_vars["positions"].zero_()
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["cache_seqlens"].zero_()
        graph_vars["block_tables"].fill_(self.dummy_block_id)
        graph_vars["input_ids"][:bs].copy_(input_ids)
        graph_vars["positions"][:bs].copy_(positions)
        graph_vars["slot_mapping"][:bs].copy_(context.slot_mapping)
        graph_vars["cache_seqlens"][:bs].copy_(context.cache_seqlens)
        graph_vars["block_tables"][:bs, :context.block_tables.size(1)].copy_(context.block_tables)

    def set_full_decode_context(self, graph_vars: dict[str, torch.Tensor]):
        set_decode_context(graph_vars["slot_mapping"], graph_vars["cache_seqlens"], graph_vars["block_tables"])

    def capture_full_decode_graph(self, key: tuple[int, int]):
        graph_vars = self.get_full_decode_vars(key)
        graph = torch.cuda.CUDAGraph()
        previous_context = get_context()
        self.set_full_decode_context(graph_vars)
        try:
            graph_vars["outputs"].copy_(self.model(graph_vars["input_ids"], graph_vars["positions"]))
            with torch.cuda.graph(graph, self.graph_pool):
                graph_vars["outputs"].copy_(self.model(graph_vars["input_ids"], graph_vars["positions"]))
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            torch.cuda.synchronize()
        finally:
            set_context_obj(previous_context)
        self.full_decode_graphs[key] = (graph, graph_vars)
        return graph, graph_vars

    def run_full_decode_graph(self, input_ids: torch.Tensor, positions: torch.Tensor):
        actual_context = get_context()
        key = self.get_decode_graph_key(input_ids)
        disabled_key = (RuntimeMode.FULL_DECODE, key)
        if disabled_key in self.disabled_graphs:
            return False, None
        try:
            graph, graph_vars = self.full_decode_graphs.get(key) or self.capture_full_decode_graph(key)
            self.pad_full_decode_inputs(graph_vars, input_ids, positions)
            self.set_full_decode_context(graph_vars)
            graph.replay()
        except Exception as exc:
            self.disabled_graphs.add(disabled_key)
            set_context_obj(actual_context)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
            warnings.warn(f"CUDA graph disabled for bucket {key}: {exc}", RuntimeWarning)
            return False, None
        return True, graph_vars["outputs"][:input_ids.size(0)]

    def empty_piece_buffers(self, token_bucket: int):
        hf_config = self.config.hf_config
        attn = self.model.model.layers[0].self_attn
        return (
            torch.empty(token_bucket, attn.num_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda"),
            torch.empty(token_bucket, attn.num_kv_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda"),
            torch.empty(token_bucket, attn.num_kv_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda"),
            torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda"),
        )

    def capture_piece0_graph(self, key: tuple[int, int, int, int], runtime_vars: dict[str, torch.Tensor]):
        graph_key = ("piece0", key)
        if graph_key in self.piecewise_graphs:
            return self.piecewise_graphs[graph_key]
        token_bucket = runtime_vars["input_ids"].size(0)
        q_buf, k_buf, v_buf, residual_buf = self.empty_piece_buffers(token_bucket)
        graph = torch.cuda.CUDAGraph()
        q, k, v, residual = self.model.model.first_piece(runtime_vars["input_ids"], runtime_vars["positions"])
        q_buf.copy_(q)
        k_buf.copy_(k)
        v_buf.copy_(v)
        residual_buf.copy_(residual)
        with torch.cuda.graph(graph, self.graph_pool):
            q, k, v, residual = self.model.model.first_piece(runtime_vars["input_ids"], runtime_vars["positions"])
            q_buf.copy_(q)
            k_buf.copy_(k)
            v_buf.copy_(v)
            residual_buf.copy_(residual)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.piecewise_graphs[graph_key] = (graph, (q_buf, k_buf, v_buf, residual_buf))
        return self.piecewise_graphs[graph_key]

    def capture_piece_next_graph(self, key: tuple[int, int, int, int], layer_idx: int, runtime_vars: dict[str, torch.Tensor]):
        graph_key = ("piece_next", layer_idx, key)
        if graph_key in self.piecewise_graphs:
            return self.piecewise_graphs[graph_key]
        token_bucket = runtime_vars["input_ids"].size(0)
        hf_config = self.config.hf_config
        attn = self.model.model.layers[layer_idx].self_attn
        attn_input = torch.empty(token_bucket, attn.num_heads, attn.head_dim, dtype=hf_config.dtype, device="cuda")
        residual_input = torch.empty(token_bucket, hf_config.hidden_size, dtype=hf_config.dtype, device="cuda")
        attn_input.zero_()
        residual_input.zero_()
        q_buf, k_buf, v_buf, residual_buf = self.empty_piece_buffers(token_bucket)
        graph = torch.cuda.CUDAGraph()
        q, k, v, residual = self.model.model.next_piece(layer_idx, attn_input, residual_input, runtime_vars["positions"])
        q_buf.copy_(q)
        k_buf.copy_(k)
        v_buf.copy_(v)
        residual_buf.copy_(residual)
        with torch.cuda.graph(graph, self.graph_pool):
            q, k, v, residual = self.model.model.next_piece(layer_idx, attn_input, residual_input, runtime_vars["positions"])
            q_buf.copy_(q)
            k_buf.copy_(k)
            v_buf.copy_(v)
            residual_buf.copy_(residual)
        if self.graph_pool is None:
            self.graph_pool = graph.pool()
        torch.cuda.synchronize()
        self.piecewise_graphs[graph_key] = (graph, attn_input, residual_input, (q_buf, k_buf, v_buf, residual_buf))
        return self.piecewise_graphs[graph_key]

    def capture_piece_last_graph(self, key: tuple[int, int, int, int], layer_idx: int, runtime_vars: dict[str, torch.Tensor]):
        graph_key = ("piece_last", layer_idx, key)
        if graph_key in self.piecewise_graphs:
            return self.piecewise_graphs[graph_key]
        token_bucket = runtime_vars["input_ids"].size(0)
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

    def ensure_piecewise_graphs(self, key: tuple[int, int, int, int], runtime_vars: dict[str, torch.Tensor]):
        self.capture_piece0_graph(key, runtime_vars)
        num_layers = len(self.model.model.layers)
        for layer_idx in range(num_layers - 1):
            self.capture_piece_next_graph(key, layer_idx, runtime_vars)
        self.capture_piece_last_graph(key, num_layers - 1, runtime_vars)

    def release_piecewise_bucket(self, key: tuple[int, int, int, int]):
        graph_keys = [graph_key for graph_key in self.piecewise_graphs if graph_key[-1] == key]
        for graph_key in graph_keys:
            del self.piecewise_graphs[graph_key]
        self.piecewise_runtime_vars.pop(key, None)
        gc.collect()
        torch.cuda.empty_cache()

    def piecewise_graph_allowed(self, key: tuple[int, int, int, int]):
        limit = self.config.max_piecewise_cudagraph_tokens
        return limit > 0 and key[1] <= limit

    def run_piecewise_graph(self, input_ids: torch.Tensor, positions: torch.Tensor):
        actual_context = get_context()
        key = self.get_varlen_graph_key(input_ids)
        disabled_key = (RuntimeMode.PIECEWISE, key)
        if disabled_key in self.disabled_graphs:
            return False, None
        if not self.piecewise_graph_allowed(key):
            self.disabled_graphs.add(disabled_key)
            warnings.warn(
                f"Piecewise CUDA graph skipped for bucket {key}: token bucket exceeds "
                f"max_piecewise_cudagraph_tokens={self.config.max_piecewise_cudagraph_tokens}",
                RuntimeWarning,
            )
            return False, None
        runtime_vars = self.get_piecewise_runtime_vars(key)
        try:
            self.ensure_piecewise_graphs(key, runtime_vars)
            self.pad_piecewise_inputs(runtime_vars, input_ids, positions)
            graph, outputs = self.capture_piece0_graph(key, runtime_vars)
            graph.replay()
            q, k, v, residual = outputs
            num_layers = len(self.model.model.layers)
            for layer_idx, layer in enumerate(self.model.model.layers):
                attn_output = layer.attention(q, k, v)
                if layer_idx == num_layers - 1:
                    graph, attn_input, residual_input, hidden_states = self.capture_piece_last_graph(key, layer_idx, runtime_vars)
                    attn_input.copy_(attn_output)
                    residual_input.copy_(residual)
                    graph.replay()
                else:
                    graph, attn_input, residual_input, outputs = self.capture_piece_next_graph(key, layer_idx, runtime_vars)
                    attn_input.copy_(attn_output)
                    residual_input.copy_(residual)
                    graph.replay()
                    q, k, v, residual = outputs
        except Exception as exc:
            self.disabled_graphs.add(disabled_key)
            set_context_obj(actual_context)
            if isinstance(exc, torch.cuda.OutOfMemoryError):
                runtime_vars = None
                self.release_piecewise_bucket(key)
                torch.cuda.empty_cache()
            warnings.warn(f"Piecewise CUDA graph disabled for bucket {key}: {exc}", RuntimeWarning)
            return False, None
        return True, hidden_states

    def run_piecewise_eager(self, input_ids: torch.Tensor, positions: torch.Tensor):
        q, k, v, residual = self.model.model.first_piece(input_ids, positions)
        num_layers = len(self.model.model.layers)
        for layer_idx, layer in enumerate(self.model.model.layers):
            attn_output = layer.attention(q, k, v)
            if layer_idx == num_layers - 1:
                return self.model.model.final_piece(layer_idx, attn_output, residual)
            q, k, v, residual = self.model.model.next_piece(layer_idx, attn_output, residual, positions)
        raise RuntimeError("Qwen3Model must contain at least one decoder layer")

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, sample_indices: torch.Tensor, runtime_mode: RuntimeMode):
        if runtime_mode == RuntimeMode.FULL_DECODE:
            used_graph, hidden_states = self.run_full_decode_graph(input_ids, positions)
            if not used_graph:
                hidden_states = self.model(input_ids, positions)
        elif runtime_mode == RuntimeMode.PIECEWISE and get_context().block_tables is not None:
            used_graph, hidden_states = self.run_piecewise_graph(input_ids, positions)
            if not used_graph:
                hidden_states = self.run_piecewise_eager(input_ids, positions)
        else:
            hidden_states = self.model(input_ids, positions)

        if sample_indices.numel() == 0:
            return None
        return self.model.compute_logits(hidden_states.index_select(0, sample_indices))

    def run(self, seqs: list[Sequence]) -> list[int]:
        descriptor = BatchDescriptor.from_sequences(seqs)
        runtime_mode = self.dispatcher.dispatch(descriptor)
        if descriptor.is_uniform_decode:
            input_ids, positions, sample_indices = self.prepare_decode(seqs)
        else:
            input_ids, positions, sample_indices = self.prepare_mixed(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, sample_indices, runtime_mode)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 and logits is not None else []
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        self.graph_pool = None
        self.full_decode_graphs = {}
        self.piecewise_graphs = {}
        self.piecewise_runtime_vars = {}
        self.disabled_graphs = set()
