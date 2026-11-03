"""Parte 8: guardar y recuperar experiencias reales, todavía sin entrenar."""

import json
from pathlib import Path

import gymnasium as gym

from agents.replay import ReplayBuffer


def main():
    # Capacidad pequeña para mostrar qué ocurre cuando la memoria se llena.
    memory = ReplayBuffer(capacity=8, seed=42)
    env = gym.make("CartPole-v1")
    experiences = []
    try:
        env.action_space.seed(42)
        observation, _ = env.reset(seed=42)
        for step in range(1, 13):
            action = int(env.action_space.sample())
            next_observation, reward, terminated, truncated, _ = env.step(action)
            memory.add(observation, action, reward, next_observation, terminated, truncated)
            experiences.append({"step": step, "observation": observation.tolist(),
                                "action": action, "reward": reward,
                                "next_observation": next_observation.tolist(),
                                "terminated": terminated, "truncated": truncated})
            observation = next_observation
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()

    batch = memory.sample(batch_size=4)
    print("Interactuamos 12 pasos con CartPole, usando acciones aleatorias.")
    print(f"Capacidad: {memory.capacity}. Experiencias conservadas: {len(memory)}.")
    print("Quedan los pasos 5 al 12: los cuatro más antiguos fueron reemplazados.")
    print("\nSacamos 4 recuerdos al azar, sin repetir dentro de este lote:")
    sampled = []
    for index in range(4):
        record = {"observation": batch.observations[index].tolist(),
                  "action": int(batch.actions[index]), "reward": float(batch.rewards[index]),
                  "next_observation": batch.next_observations[index].tolist(),
                  "terminated": bool(batch.terminated[index]),
                  "truncated": bool(batch.truncated[index])}
        sampled.append(record)
        print(f"Recuerdo {index + 1}: estado={record['observation']}")
        print(f"  Acción: {record['action']}; recompensa: {record['reward']}")
        print(f"  Estado siguiente: {record['next_observation']}")
    print("\nLas experiencias elegidas siguen en la memoria; muestrear no las borra.")
    print("No se actualizaron pesos de una red ni se entrenó un agente.")

    output = Path("results/parte_08_memoria")
    output.mkdir(parents=True, exist_ok=True)
    result = {"purpose": "Demostración del replay buffer, sin entrenamiento",
              "seed": 42, "capacity": 8, "collected_steps": 12, "stored": len(memory),
              "batch_size": 4, "all_collected": experiences, "sampled": sampled}
    (output / "demostracion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
