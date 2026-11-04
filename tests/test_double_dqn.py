"""Objetivos numéricos conocidos para distinguir DQN, Double DQN y la ablación."""

import unittest

import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.double_dqn import DoubleDQNAgent
from agents.dqn_without_target import DQNWithoutTargetAgent
from agents.replay import ReplayBatch


class DoubleDQNTests(unittest.TestCase):
    def prepared(self, cls):
        agent = cls(obs_dim=8, n_actions=4, gamma=0.9, seed=123)
        with torch.no_grad():
            for p in agent.online.parameters():
                p.zero_()
            agent.online.layers[-1].bias.copy_(torch.tensor([9., 1., -1., -2.]))
            if hasattr(agent, 'target'):
                for p in agent.target.parameters():
                    p.zero_()
                agent.target.layers[-1].bias.copy_(torch.tensor([2., 7., -1., -2.]))
        return agent

    def batch(self):
        return ReplayBatch(np.zeros((3, 8), np.float32), np.zeros(3, np.int64),
                           np.ones(3, np.float32), np.zeros((3, 8), np.float32),
                           np.array([False, True, False]), np.array([False, False, True]))

    def test_dqn_uses_target_max_while_double_uses_online_action(self):
        dqn = self.prepared(DQNAgent)
        double = self.prepared(DoubleDQNAgent)
        np.testing.assert_allclose(dqn.learn(self.batch())['targets'], [7.3, 1, 7.3], atol=1e-6)
        np.testing.assert_allclose(double.learn(self.batch())['targets'], [2.8, 1, 2.8], atol=1e-6)
        self.assertTrue(all(p.grad is None for p in double.target.parameters()))

    def test_ablation_has_no_separate_target_and_keeps_learning(self):
        agent = self.prepared(DQNWithoutTargetAgent)
        self.assertFalse(hasattr(agent, 'target'))
        old = agent.online.layers[-1].bias.detach().clone()
        result = agent.learn(self.batch())
        np.testing.assert_allclose(result['targets'], [9.1, 1, 9.1], atol=1e-6)
        agent.sync_target()
        self.assertTrue(all(p.requires_grad for p in agent.online.parameters()))
        self.assertFalse(torch.equal(old, agent.online.layers[-1].bias))


if __name__ == '__main__':
    unittest.main()
