"""Parte 9: una actualización real, manteniendo fija la red objetivo."""

import json
from pathlib import Path

import gymnasium as gym
import torch

from agents.dqn import DQNAgent
from agents.replay import ReplayBuffer


def snapshot(network):
    return {name: value.clone() for name, value in network.state_dict().items()}


def equals(network, values):
    return all(torch.equal(value, values[name]) for name, value in network.state_dict().items())


def main():
    torch.set_num_threads(1)
    agent = DQNAgent(seed=42)
    memory = ReplayBuffer(capacity=100, seed=42)
    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=42)
        env.action_space.seed(42)
        for _ in range(64):
            action = int(env.action_space.sample())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            memory.add(observation, action, reward, next_observation, terminated, truncated)
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()

    initial_online = snapshot(agent.online)
    initial_target = snapshot(agent.target)
    assert equals(agent.online, initial_target)
    batch = memory.sample(8)
    result = agent.learn(batch)
    online_changed = not equals(agent.online, initial_online)
    target_unchanged = equals(agent.target, initial_target)
    assert online_changed and target_unchanged
    assert all(parameter.grad is None for parameter in agent.target.parameters())

    with torch.no_grad():
        predictions_after = agent.online(torch.as_tensor(batch.observations)).gather(
            1, torch.as_tensor(batch.actions)[:, None]).squeeze(1)
        loss_after = torch.nn.functional.smooth_l1_loss(
            predictions_after, torch.tensor(result["targets"])).item()
    agent.sync_target()
    synced = equals(agent.target, snapshot(agent.online))
    assert synced
    print("Guardamos 64 experiencias reales y tomamos un lote de 8.")
    print("Al inicio: ambas redes tienen los mismos valores, en memorias separadas.")
    print(f"Después de aprender una vez: online cambió={online_changed}; target se mantuvo={target_unchanged}.")
    print(f"Error del lote antes: {result['loss']:.6f}")
    print(f"Error del mismo lote después (contra las mismas referencias): {loss_after:.6f}")
    print(f"Después de copiar online hacia target: coinciden={synced}.")
    print("Un ajuste no demuestra que el agente ya sepa equilibrar el poste.")

    output = Path("results/parte_09_red_objetivo")
    output.mkdir(parents=True, exist_ok=True)
    record = {"purpose": "Demostración de una sola actualización, no entrenamiento completo",
              "seed": 42, "torch": torch.__version__, "collected_steps": 64,
              "batch_size": 8, "updates": 1, "learning_rate": 0.001, "gamma": 0.99,
              **result, "loss_after_same_targets": loss_after,
              "online_changed": online_changed, "target_unchanged_during_learning": target_unchanged,
              "networks_equal_after_sync": synced}
    (output / "demostracion.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
