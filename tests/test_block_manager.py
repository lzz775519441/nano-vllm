import unittest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence


class BlockManagerTest(unittest.TestCase):
    def setUp(self):
        self.old_block_size = Sequence.block_size
        Sequence.block_size = 4

    def tearDown(self):
        Sequence.block_size = self.old_block_size

    def test_incremental_allocation_and_full_block_hashing(self):
        manager = BlockManager(num_blocks=3, block_size=Sequence.block_size)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8, 9])

        self.assertEqual(manager.can_allocate(seq), 0)
        manager.allocate(seq, 0)
        self.assertEqual(seq.block_table, [])

        manager.may_append_tokens(seq, 3)
        self.assertEqual(len(seq.block_table), 1)
        self.assertEqual(len(manager.free_block_ids), 2)

        seq.num_scheduled_tokens = 3
        manager.hash_blocks(seq)
        self.assertEqual(manager.hash_to_block_id, {})
        seq.num_cached_tokens += seq.num_scheduled_tokens

        manager.may_append_tokens(seq, 2)
        self.assertEqual(len(seq.block_table), 2)
        seq.num_scheduled_tokens = 2
        manager.hash_blocks(seq)

        first_block_hash = BlockManager.compute_hash([1, 2, 3, 4])
        self.assertEqual(manager.hash_to_block_id[first_block_hash], seq.block_table[0])

        manager.deallocate(seq)
        self.assertEqual(seq.block_table, [])
        self.assertEqual(seq.num_cached_tokens, 0)
        self.assertEqual(len(manager.free_block_ids), 3)

    def test_prefix_cache_reuses_complete_blocks_only(self):
        manager = BlockManager(num_blocks=4, block_size=Sequence.block_size)
        seq = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        manager.allocate(seq, 0)
        manager.may_append_tokens(seq, 8)
        seq.num_scheduled_tokens = 8
        manager.hash_blocks(seq)
        manager.deallocate(seq)

        cached = Sequence([1, 2, 3, 4, 5, 6, 7, 8])
        num_cached_blocks = manager.can_allocate(cached)
        manager.allocate(cached, num_cached_blocks)

        self.assertEqual(num_cached_blocks, 1)
        self.assertEqual(cached.num_cached_tokens, 4)
        self.assertEqual(len(cached.block_table), 1)


if __name__ == "__main__":
    unittest.main()
