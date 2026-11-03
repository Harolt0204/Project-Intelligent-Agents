"""Auditoría numérica y de integración; no es un entrenamiento de CartPole."""

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay import ReplayBuffer


def real_transition_audit():
    memory = ReplayBuffer(128, seed=42)
    env = gym.make("CartPole-v1", max_episode_steps=17)
    expected = []
    ends = {"terminated": 0, "truncated": 0}
    try:
        env.action_space.seed(42)
        observation, _ = env.reset(seed=42)
        for _ in range(600):
            action = int(env.action_space.sample())
            following, reward, terminated, truncated, _ = env.step(action)
            memory.add(observation, action, reward, following, terminated, truncated)
            expected.append((observation.copy(), action, reward, following.copy(), terminated, truncated))
            observation = following
            ends["terminated"] += int(terminated)
            ends["truncated"] += int(truncated)
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
    lookup = {tuple(row[0]): row for row in expected[-128:]}
    assert len(lookup) == 128 and all(ends.values())
    batch = memory.sample(128)
    for i, observation in enumerate(batch.observations):
        original = lookup[tuple(observation)]
        assert batch.actions[i] == original[1] and batch.rewards[i] == original[2]
        np.testing.assert_array_equal(batch.next_observations[i], original[3])
        assert batch.terminated[i] == original[4] and batch.truncated[i] == original[5]

    # Comprobar explícitamente que la observación de reset no sustituye la final.
    env = gym.make("CartPole-v1", max_episode_steps=1)
    try:
        first, _ = env.reset(seed=42)
        last, reward, terminated, truncated, _ = env.step(0)
        one = ReplayBuffer(1)
        one.add(first, 0, reward, last, terminated, truncated)
        reset, _ = env.reset(seed=43)
        assert truncated and not terminated
        np.testing.assert_array_equal(one.sample(1).next_observations[0], last)
        assert not np.array_equal(last, reset)
    finally:
        env.close()
    return batch, {"collected": 600, "capacity": 128, "checked_retained_transitions": 128,
                   "episode_ends": ends, "final_observation_preserved_after_reset": True}


def numpy_weights(network):
    return {name: value.detach().numpy().astype(np.float64).copy()
            for name, value in network.state_dict().items()}


def numpy_forward(weights, observations):
    values = np.asarray(observations, dtype=np.float64)
    for index in (0, 2, 4):
        values = values @ weights[f"layers.{index}.weight"].T + weights[f"layers.{index}.bias"]
        if index != 4:
            values = np.maximum(values, 0)
    return values


def numerical_audit(batch):
    agent = DQNAgent(hidden_width=16, learning_rate=0.0005, seed=42)
    online = numpy_weights(agent.online)
    target = numpy_weights(agent.target)
    expected_targets = batch.rewards + 0.99 * (~batch.terminated) * numpy_forward(
        target, batch.next_observations).max(axis=1)

    def independent_loss(weights):
        scores = numpy_forward(weights, batch.observations)
        selected = scores[np.arange(len(batch.actions)), batch.actions]
        errors = np.abs(selected - expected_targets)
        return np.where(errors <= 1, 0.5 * errors ** 2, errors - 0.5).mean()

    expected_loss = float(independent_loss(online))
    delta = 1e-5
    online["layers.4.bias"][0] += delta
    plus = independent_loss(online)
    online["layers.4.bias"][0] -= 2 * delta
    minus = independent_loss(online)
    finite_difference = float((plus - minus) / (2 * delta))
    result = agent.learn(batch)
    actual_gradient = float(agent.online.layers[-1].bias.grad[0])
    grad_norm = float(torch.sqrt(sum(p.grad.square().sum() for p in agent.online.parameters())))
    assert grad_norm < 9.9  # En este caso la limitación de norma no debe activarse.
    np.testing.assert_allclose(result["targets"], expected_targets, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(result["loss"], expected_loss, atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual_gradient, finite_difference, atol=1e-5, rtol=1e-4)
    return {"numpy_target_max_abs_error": float(np.max(np.abs(np.asarray(result["targets"]) - expected_targets))),
            "torch_loss": result["loss"], "independent_numpy_loss": expected_loss,
            "finite_difference_gradient": finite_difference, "torch_gradient": actual_gradient,
            "gradient_norm": grad_norm}


def known_mdp_audit(seed):
    # Dos estados alternan siempre; acertar la acción da 1, fallar da 0.
    # Gamma=.9: Q óptimo correcto=10, incorrecto=9 en ambos estados.
    agent = DQNAgent(hidden_width=16, learning_rate=0.001, gamma=0.9, seed=seed)
    memory = ReplayBuffer(4, seed=seed + 30_000)
    states = np.array([[-1, 0, 0, 0], [1, 0, 0, 0]], dtype=np.float32)
    for state in (0, 1):
        for action in (0, 1):
            memory.add(states[state], action, float(action == state), states[1 - state], False, False)
    for update in range(1, 4001):
        agent.learn(memory.sample(4))
        if update % 100 == 0:
            agent.sync_target()
    with torch.inference_mode():
        predictions = agent.online(torch.as_tensor(states)).numpy()
    expected = np.array([[10, 9], [9, 10]], dtype=np.float64)
    assert predictions.argmax(axis=1).tolist() == [0, 1]
    max_error = float(np.abs(predictions - expected).max())
    assert max_error < 0.5, f"El problema de control no se resolvió: error={max_error}"
    return {"seed": seed, "updates": 4000, "gamma": 0.9, "target_sync_every_updates": 100,
            "known_optimal_q": expected.tolist(), "learned_q": predictions.tolist(),
            "max_abs_error": max_error, "greedy_actions": predictions.argmax(axis=1).tolist()}


def main():
    torch.set_num_threads(1)
    batch, transitions = real_transition_audit()
    numeric = numerical_audit(batch)
    print("Flujo de experiencias y comprobación numérica independientes: correctos.", flush=True)
    controls = []
    for seed in (42, 43, 44):
        result = known_mdp_audit(seed)
        controls.append(result)
        print(f"Control con solución conocida, semilla {seed}: error máximo={result['max_abs_error']:.4f}", flush=True)
    record = {"purpose": "Auditoría de integración, no resultado experimental de CartPole",
              "torch": torch.__version__, "transitions": transitions, "numeric": numeric,
              "known_mdp_controls": controls,
              "scope": "No se encontró error en estos controles; no demuestra ausencia de todos los errores ni estabilidad en CartPole"}
    folder = Path("results/parte_13_auditoria")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "audit.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
