"""Compatibilidad de modelos y configuración nueva con las ejecuciones originales."""

import unittest

import numpy as np
import torch

from agents.dqn import QNetwork
from envs.discretizers import CartPoleDiscretizer


class ConfigurationTests(unittest.TestCase):
    def test_different_resolution_per_variable_preserves_four_observations(self):
        grid = CartPoleDiscretizer(10, position_bins=2, velocity_bins=2)
        self.assertEqual(grid.shape, (2, 2, 10, 10))
        self.assertEqual(grid.encode([-1, 1, -999, 999]), (0, 1, 0, 9))
        self.assertEqual(int(np.prod(grid.shape)), 400)

    def test_nonuniform_edges_are_symmetric_ordered_and_finer_near_zero(self):
        grid = CartPoleDiscretizer(6, spacing_power=2)
        uniform = CartPoleDiscretizer(6)
        for edges, old in zip(grid.edges, uniform.edges):
            self.assertTrue(np.all(np.diff(edges) > 0))
            np.testing.assert_allclose(edges, -edges[::-1], atol=1e-14)
            self.assertLess(edges[3] - edges[2], old[3] - old[2])
        self.assertEqual(grid.encode([999, -999, 999, -999]), (5, 0, 5, 0))

    def test_saved_square_architecture_loads_with_two_widths(self):
        original = QNetwork(hidden_width=64)
        restored = QNetwork(hidden_width=[64, 64])
        restored.load_state_dict(original.state_dict())
        state = torch.tensor([0.02, -0.1, 0.03, 0.2])
        torch.testing.assert_close(original(state), restored(state), rtol=0, atol=0)

    def test_asymmetric_architecture_has_two_action_values(self):
        model = QNetwork(hidden_width=[120, 84])
        self.assertEqual(tuple(model(torch.zeros((5, 4))).shape), (5, 2))
        self.assertEqual(model.layers[2].in_features, 120)
        self.assertEqual(model.layers[2].out_features, 84)

    def test_custom_bounds_clip_and_resolve_near_zero(self):
        custom = CartPoleDiscretizer(10, 0.75, 0.8)
        self.assertEqual(custom.encode([0, -99, 0, 99])[1::2], (0, 9))
        np.testing.assert_allclose(custom.low, [-2.4, -0.75, -np.deg2rad(12), -0.8])
        self.assertNotEqual(custom.encode([0, 0.2, 0, 0]), custom.encode([0, 0.4, 0, 0]))

    def test_original_bounds_unchanged_by_default(self):
        np.testing.assert_allclose(CartPoleDiscretizer(10).low, [-2.4, -3, -np.deg2rad(12), -3.5])


if __name__ == '__main__':
    unittest.main()
