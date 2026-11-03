"""Verificar que la red pueda usarse para decisiones y aprendizaje posterior."""

import unittest

import torch

from agents.dqn import QNetwork


class QNetworkTests(unittest.TestCase):
    def test_single_and_batch_predictions_agree(self):
        torch.manual_seed(42)
        model = QNetwork()
        observation = torch.tensor([0.1, -0.2, 0.03, 0.4])
        with torch.inference_mode():
            single = model(observation)
            batch = model(torch.stack([observation, observation]))
        self.assertEqual(tuple(single.shape), (2,))
        self.assertEqual(tuple(batch.shape), (2, 2))
        torch.testing.assert_close(single, batch[0])
        torch.testing.assert_close(single, batch[1])

    def test_gradients_can_reach_all_layers(self):
        torch.manual_seed(42)
        model = QNetwork()
        predictions = model(torch.randn(8, 4))
        selected = predictions.gather(1, torch.tensor([[0], [1]] * 4)).squeeze(1)
        loss = torch.nn.functional.smooth_l1_loss(selected, torch.ones(8))
        loss.backward()
        for layer in (model.layers[0], model.layers[2], model.layers[4]):
            self.assertIsNotNone(layer.weight.grad)
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(layer.weight.grad.abs().sum().item(), 0)

    def test_inference_does_not_change_weights(self):
        model = QNetwork()
        original = {name: value.clone() for name, value in model.state_dict().items()}
        with torch.inference_mode():
            model(torch.zeros(4))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
