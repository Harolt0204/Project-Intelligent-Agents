"""Comprobar decisiones y objetivos de aprendizaje con casos conocidos."""

import unittest

import numpy as np

from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer
from experiments.entrenar_tabular import evaluate


class QLearningTests(unittest.TestCase):
    def test_numeric_update_changes_only_chosen_entry(self):
        agent = QLearningAgent((2,), alpha=0.1, gamma=0.9)
        agent.q[:] = [[5, 0], [10, 2]]
        self.assertAlmostEqual(agent.update((0,), 0, 1, (1,), False), 5.5)
        np.testing.assert_array_equal(agent.q, [[5.5, 0], [10, 2]])

    def test_failure_does_not_include_future_value(self):
        agent = QLearningAgent((2,), alpha=0.1, gamma=0.9)
        agent.q[:] = [[5, 0], [100, 100]]
        self.assertAlmostEqual(agent.update((0,), 0, 1, (1,), True), 4.6)

    def test_time_limit_keeps_future_value(self):
        # Un final por tiempo conserva terminated=False, aunque truncated=True.
        agent = QLearningAgent((2,), alpha=1.0, gamma=0.9)
        agent.q[1] = [10, 2]
        self.assertAlmostEqual(agent.update((0,), 0, 1, (1,), False), 10.0)

    def test_greedy_action_and_exploration(self):
        agent = QLearningAgent((1,), seed=42)
        agent.q[0] = [20, 8]
        self.assertEqual({agent.choose_action((0,), 0) for _ in range(100)}, {0})
        self.assertEqual({agent.choose_action((0,), 1) for _ in range(100)}, {0, 1})

    def test_reproducible_actions_and_unbiased_initial_ties(self):
        first = QLearningAgent((1,), seed=42)
        second = QLearningAgent((1,), seed=42)
        actions = [first.choose_action((0,), 0) for _ in range(100)]
        self.assertEqual(actions, [second.choose_action((0,), 0) for _ in range(100)])
        self.assertEqual(set(actions), {0, 1})

    def test_evaluation_preserves_training_table_and_rng(self):
        agent = QLearningAgent((6,) * 4, seed=42)
        original_q = agent.q.copy()
        original_rng = agent.rng.bit_generator.state
        first = evaluate(agent, CartPoleDiscretizer(6), seed=42, episodes=2)
        second = evaluate(agent, CartPoleDiscretizer(6), seed=42, episodes=2)
        np.testing.assert_array_equal(agent.q, original_q)
        self.assertEqual(agent.rng.bit_generator.state, original_rng)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
