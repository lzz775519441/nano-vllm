import unittest
import pickle
from collections import deque
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus


class SchedulerTest(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def make_scheduler(self, max_tokens=4, max_seqs=8, num_blocks=16):
        config = SimpleNamespace(
            max_num_seqs=max_seqs,
            max_num_batched_tokens=max_tokens,
            eos=-1,
            kvcache_block_size=Sequence.block_size,
            num_kvcache_blocks=num_blocks,
        )
        return Scheduler(config)

    def make_running(self, scheduler: Scheduler, cached_tokens: list[int], pending_token: int):
        seq = Sequence(cached_tokens)
        scheduler.block_manager.may_append_tokens(seq, len(cached_tokens))
        seq.num_cached_tokens = len(cached_tokens)
        seq.append_token(pending_token)
        seq.status = SequenceStatus.RUNNING
        seq.is_prefill = False
        scheduler.running.append(seq)
        return seq

    def test_decode_is_scheduled_before_chunked_prefill(self):
        scheduler = self.make_scheduler(max_tokens=4)
        running = self.make_running(scheduler, [1, 2, 3], 4)
        waiting = Sequence([10, 11, 12, 13, 14, 15])
        scheduler.add(waiting)

        output = scheduler.schedule()

        self.assertEqual(output.seqs, [running, waiting])
        self.assertEqual(output.num_decode_tokens, 1)
        self.assertEqual(output.num_prefill_tokens, 3)
        self.assertEqual(running.num_scheduled_tokens, 1)
        self.assertEqual(waiting.num_scheduled_tokens, 3)
        self.assertEqual(waiting.status, SequenceStatus.WAITING)

        scheduler.postprocess(output, [99])

        self.assertEqual(running.completion_token_ids, [4, 99])
        self.assertEqual(waiting.completion_token_ids, [])
        self.assertEqual(waiting.num_cached_tokens, 3)

    def test_prefill_chunk_samples_only_after_prompt_is_complete(self):
        scheduler = self.make_scheduler(max_tokens=5)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7])
        scheduler.add(seq)

        first = scheduler.schedule()
        self.assertEqual(first.num_prefill_tokens, 5)
        self.assertFalse(seq.needs_sampling)
        scheduler.postprocess(first, [])
        self.assertEqual(seq.num_cached_tokens, 5)
        self.assertEqual(seq.completion_token_ids, [])
        self.assertEqual(scheduler.waiting, deque([seq]))

        second = scheduler.schedule()
        self.assertEqual(second.num_prefill_tokens, 2)
        self.assertTrue(seq.needs_sampling)
        scheduler.postprocess(second, [42])

        self.assertEqual(seq.status, SequenceStatus.RUNNING)
        self.assertEqual(seq.num_cached_tokens, 7)
        self.assertEqual(seq.completion_token_ids, [42])
        self.assertEqual(scheduler.running, deque([seq]))

    def test_mixed_final_prefill_and_decode_consume_tokens_in_scheduled_order(self):
        scheduler = self.make_scheduler(max_tokens=3)
        running = self.make_running(scheduler, [1, 2, 3], 4)
        waiting = Sequence([10, 11])
        scheduler.add(waiting)

        output = scheduler.schedule()
        self.assertEqual(output.seqs, [running, waiting])
        self.assertTrue(running.needs_sampling)
        self.assertTrue(waiting.needs_sampling)

        scheduler.postprocess(output, [50, 60])

        self.assertEqual(running.completion_token_ids, [4, 50])
        self.assertEqual(waiting.completion_token_ids, [60])

    def test_preempted_decode_returns_to_waiting_for_recompute(self):
        scheduler = self.make_scheduler(max_tokens=4, num_blocks=1)
        running = self.make_running(scheduler, [1, 2, 3, 4], 5)

        output = scheduler.schedule()

        self.assertEqual(output.seqs, [running])
        self.assertEqual(output.num_decode_tokens, 0)
        self.assertEqual(output.num_prefill_tokens, 4)
        self.assertEqual(running.status, SequenceStatus.WAITING)
        self.assertTrue(running.is_prefill)
        self.assertEqual(running.num_cached_tokens, 0)

    def test_decode_pickle_keeps_scheduled_kind_and_last_token(self):
        scheduler = self.make_scheduler()
        seq = self.make_running(scheduler, [1, 2, 3], 4)
        seq.num_scheduled_tokens = 1

        restored = pickle.loads(pickle.dumps(seq))

        self.assertFalse(restored.is_prefill)
        self.assertEqual(restored.num_scheduled_tokens, 1)
        self.assertEqual(restored.num_cached_tokens, 3)
        self.assertEqual(restored.last_token, 4)
        self.assertEqual(restored.token_ids, [])


if __name__ == "__main__":
    unittest.main()
