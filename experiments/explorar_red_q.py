"""Parte 7: calcular dos valores Q con una red sin entrenar."""

import json
from pathlib import Path

import gymnasium as gym
import torch

from agents.dqn import QNetwork


def main():
    seed = 42
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    network = QNetwork(obs_dim=4, n_actions=2, hidden_width=64)
    network.eval()
    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=seed)
    finally:
        env.close()

    # No se discretiza: los cuatro números entran directamente como float32.
    inputs = torch.as_tensor(observation, dtype=torch.float32)
    with torch.inference_mode():
        q_values = network(inputs)
    action = int(torch.argmax(q_values).item())
    parameters = sum(parameter.numel() for parameter in network.parameters())
    print(f"PyTorch: {torch.__version__} | dispositivo: CPU")
    print("Arquitectura: 4 entradas -> 64 -> 64 -> 2 salidas")
    print(f"Parámetros ajustables: {parameters}")
    print(f"Observación real: {observation}")
    print(f"Q(izquierda): {q_values[0].item():.6f}")
    print(f"Q(derecha):   {q_values[1].item():.6f}")
    print(f"Acción con mayor puntuación: {action} ({'izquierda' if action == 0 else 'derecha'})")
    print("La red NO está entrenada: sus puntuaciones aún no son estimaciones útiles.")
    print("Son valores Q, no porcentajes; pueden ser negativos y no tienen que sumar 1.")
    print("Todavía faltan replay buffer, red objetivo y actualización de los pesos.")

    output = Path("results/parte_07_red_q")
    output.mkdir(parents=True, exist_ok=True)
    result = {"purpose": "Demostración de una red Q sin entrenar", "trained": False,
              "seed": seed, "torch": torch.__version__, "device": "cpu",
              "architecture": [4, 64, 64, 2], "parameters": parameters,
              "observation": observation.tolist(), "q_values": q_values.tolist(),
              "greedy_action": action}
    (output / "demostracion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
