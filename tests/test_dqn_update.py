"""Comprobar las referencias y qué red cambia durante el aprendizaje."""

import unittest

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay import ReplayBatch
from experiments.entrenar_dqn import evaluate_network


class DQNUpdateTests(unittest.TestCase):
    def make_batch(self):
        return ReplayBatch(np.zeros((2, 4), dtype=np.float32), np.array([0, 1]),
                           np.ones(2, dtype=np.float32), np.ones((2, 4), dtype=np.float32),
                           np.array([True, False]), np.array([False, True]))

    def test_terminal_and_time_limit_targets_have_known_values(self):
        agent = DQNAgent(gamma=0.9)
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            for parameter in agent.target.parameters():
                parameter.zero_()
            agent.target.layers[-1].bias.copy_(torch.tensor([10.0, 2.0]))
        result = agent.learn(self.make_batch())
        # Final real: solo 1. Corte temporal: 1 + 0.9*10 = 10.
        np.testing.assert_allclose(result["targets"], [1.0, 10.0])
        self.assertAlmostEqual(result["loss"], 5.0)

    def test_mse_changes_error_measure_without_changing_targets(self):
        agent = DQNAgent(gamma=0.9, loss="mse")
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            for parameter in agent.target.parameters():
                parameter.zero_()
            agent.target.layers[-1].bias.copy_(torch.tensor([10.0, 2.0]))
        result = agent.learn(self.make_batch())
        np.testing.assert_allclose(result["targets"], [1.0, 10.0])
        # Predicciones cero: promedio de errores cuadrados (1² + 10²)/2.
        self.assertAlmostEqual(result["loss"], 50.5)

    def test_learning_changes_only_online_then_sync_copies_values(self):
        agent = DQNAgent(seed=42)
        original_online = {name: value.clone() for name, value in agent.online.state_dict().items()}
        original_target = {name: value.clone() for name, value in agent.target.state_dict().items()}
        self.assertTrue(all(torch.equal(value, original_target[name])
                            for name, value in original_online.items()))
        agent.learn(self.make_batch())
        self.assertTrue(any(not torch.equal(value, original_online[name])
                            for name, value in agent.online.state_dict().items()))
        for name, value in agent.target.state_dict().items():
            torch.testing.assert_close(value, original_target[name], rtol=0, atol=0)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in agent.target.parameters()))
        agent.sync_target()
        for online, target in zip(agent.online.parameters(), agent.target.parameters()):
            torch.testing.assert_close(online, target, rtol=0, atol=0)
            self.assertNotEqual(online.data_ptr(), target.data_ptr())

    def test_greedy_policy_selects_highest_score(self):
        agent = DQNAgent()
        with torch.no_grad():
            for parameter in agent.online.parameters():
                parameter.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([-0.04, -0.17]))
        self.assertEqual(agent.choose_action(np.zeros(4), epsilon=0), 0)

    def test_evaluation_preserves_weights_modes_and_random_generators(self):
        agent = DQNAgent(seed=42)
        weights = {name: value.clone() for name, value in agent.online.state_dict().items()}
        rng_state = agent.rng.bit_generator.state
        torch_rng = torch.random.get_rng_state().clone()
        first = evaluate_network(agent.online, seed=42, episodes=2)
        second = evaluate_network(agent.online, seed=42, episodes=2)
        self.assertEqual(first, second)
        self.assertTrue(agent.online.training)
        self.assertEqual(agent.rng.bit_generator.state, rng_state)
        torch.testing.assert_close(torch.random.get_rng_state(), torch_rng, rtol=0, atol=0)
        for name, value in agent.online.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
