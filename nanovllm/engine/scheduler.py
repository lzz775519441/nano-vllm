from collections import deque
from dataclasses import dataclass

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


@dataclass(slots=True)
class SchedulerOutput:
    seqs: list[Sequence]
    num_prefill_tokens: int
    num_decode_tokens: int

    @property
    def num_batched_tokens(self):
        return self.num_prefill_tokens + self.num_decode_tokens


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_mixed_prefill_tokens = getattr(config, "max_num_mixed_prefill_tokens", self.max_num_batched_tokens)
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> SchedulerOutput:
        scheduled_seqs = []
        scheduled_decodes = []
        num_batched_tokens = 0
        num_prefill_tokens = 0
        num_decode_tokens = 0

        # Decode first to keep already-running requests moving.
        while self.running and len(scheduled_seqs) < self.max_num_seqs and num_batched_tokens < self.max_num_batched_tokens:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
                scheduled_decodes.append(seq)
                num_batched_tokens += 1
                num_decode_tokens += 1
                continue
            break

        # Chunked prefill uses a smaller budget when mixed with decode, so
        # running requests are not hidden behind a very large prefill chunk.
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if num_decode_tokens > 0:
                mixed_prefill_remaining = self.max_num_mixed_prefill_tokens - num_prefill_tokens
                remaining = min(remaining, mixed_prefill_remaining)
            if remaining <= 0:
                break
            seq = self.waiting[0]
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                self.block_manager.allocate(seq, num_cached_blocks)

            num_tokens = seq.num_tokens - seq.num_cached_tokens
            num_tokens = min(num_tokens, remaining, self.block_manager.num_appendable_tokens(seq))
            if num_tokens <= 0:
                break

            seq.num_scheduled_tokens = num_tokens
            seq.is_prefill = True
            self.block_manager.may_append_tokens(seq, num_tokens)
            num_batched_tokens += seq.num_scheduled_tokens
            num_prefill_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)
            if seq.status == SequenceStatus.WAITING:
                break

        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_decodes))
        return SchedulerOutput(scheduled_seqs, num_prefill_tokens, num_decode_tokens)

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, scheduler_output: SchedulerOutput, token_ids: list[int]):
        token_iter = iter(token_ids)
        for seq in scheduler_output.seqs:
            needs_sampling = seq.needs_sampling
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if not needs_sampling:
                continue
            token_id = next(token_iter)
            seq.append_token(token_id)
            seq.is_prefill = False
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
