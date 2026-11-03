"""Proteger la integridad de los recuerdos usados por DQN."""

import unittest

import numpy as np

from agents.replay import ReplayBuffer


class ReplayTests(unittest.TestCase):
    def test_full_memory_discards_only_oldest_experiences(self):
        memory = ReplayBuffer(3)
        for step in range(5):
            memory.add(np.full(4, step), step % 2, step, np.full(4, step + 1), False, False)
        batch = memory.sample(3)
        self.assertEqual(len(memory), 3)
        self.assertEqual(set(batch.rewards.tolist()), {2, 3, 4})
        np.testing.assert_array_equal(batch.next_observations, batch.observations + 1)

    def test_storage_and_sample_do_not_share_mutable_observations(self):
        memory = ReplayBuffer(2)
        observation = np.zeros(4, dtype=np.float32)
        following = np.ones(4, dtype=np.float32)
        memory.add(observation, 1, 1, following, False, True)
        observation[:] = 99
        following[:] = 99
        first = memory.sample(1)
        np.testing.assert_array_equal(first.observations, np.zeros((1, 4)))
        np.testing.assert_array_equal(first.next_observations, np.ones((1, 4)))
        first.observations[:] = -99
        np.testing.assert_array_equal(memory.sample(1).observations, np.zeros((1, 4)))
        self.assertEqual(len(memory), 1)

    def test_sampling_is_reproducible_and_preserves_flags(self):
        memories = [ReplayBuffer(5, seed=42), ReplayBuffer(5, seed=42)]
        for memory in memories:
            for step in range(5):
                memory.add(np.full(4, step), step % 2, step, np.full(4, step + 1),
                           step == 3, step == 4)
        first, second = (memory.sample(5) for memory in memories)
        np.testing.assert_array_equal(first.rewards, second.rewards)
        self.assertEqual(len(set(first.rewards.tolist())), 5)
        self.assertTrue(first.terminated[first.rewards == 3].item())
        self.assertFalse(first.truncated[first.rewards == 3].item())
        self.assertFalse(first.terminated[first.rewards == 4].item())
        self.assertTrue(first.truncated[first.rewards == 4].item())

    def test_cannot_sample_more_than_stored(self):
        memory = ReplayBuffer(2)
        with self.assertRaises(ValueError):
            memory.sample(1)
        memory.add(np.zeros(4), 0, 1, np.ones(4), False, False)
        for size in (0, 2):
            with self.assertRaises(ValueError):
                memory.sample(size)


if __name__ == "__main__":
    unittest.main()
