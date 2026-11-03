"""Ver los agentes ya entrenados en una ventana, sin volver a aprender."""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def load_policy(kind, seed):
    if kind == "random":
        rng = np.random.default_rng(seed + 20_000)
        return lambda observation: int(rng.integers(2))

    if kind == "tabular":
        from agents.tabular import QLearningAgent
        from envs.discretizers import CartPoleDiscretizer

        folder = PROJECT / f"results/parte_06_bins/bins_10/seed_{seed}"
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        discretizer = CartPoleDiscretizer(config["bins"])
        if not (np.allclose(discretizer.low, config["low"]) and np.allclose(discretizer.high, config["high"])):
            raise ValueError("Los rangos no coinciden con el entrenamiento guardado")
        agent = QLearningAgent(discretizer.shape, seed=seed + 20_000)
        table = np.load(folder / "q_table.npy", allow_pickle=False)
        if table.shape != agent.q.shape or not np.all(np.isfinite(table)):
            raise ValueError("Tabla incompatible o con valores no finitos")
        agent.q[:] = table
        return lambda observation: agent.choose_action(discretizer.encode(observation), epsilon=0)

    import torch
    from agents.dqn import QNetwork

    torch.set_num_threads(1)
    folder = PROJECT / f"results/parte_15_mse_completo/seed_{seed}"
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    network = QNetwork(hidden_width=config["architecture"][1]).eval()
    network.load_state_dict(torch.load(folder / "online_final.pt", map_location="cpu", weights_only=True))
    rng = np.random.default_rng(seed + 20_000)

    def choose(observation):
        with torch.inference_mode():
            scores = network(torch.as_tensor(observation, dtype=torch.float32)).numpy()
        return int(rng.choice(np.flatnonzero(scores == scores.max())))

    return choose


def show(kind, seed=44, episodes=3, headless=False):
    labels = {"random": "Acciones al azar", "tabular": "Q-learning, 10 intervalos", "dqn": "DQN con MSE"}
    print(f"\n{labels[kind]}" + (f" | ejecución entrenada: {seed}" if kind != "random" else ""), flush=True)
    if kind != "random":
        print("Cargando el modelo guardado. Durante esta muestra no aprende.", flush=True)
    if kind == "dqn" and seed == 44:
        print("Esta es la ejecución exitosa; el promedio de las tres fue 234,1 puntos.")
    if kind == "tabular":
        print("El promedio de las tres ejecuciones de Q-learning fue 359,7 puntos.")
    policy = load_policy(kind, seed)
    env = gym.make("CartPole-v1", render_mode=None if headless else "human")
    returns = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 10_000 + episode)
            total = 0.0
            while True:
                action = policy(observation)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
            print(f"Intento {episode + 1}/{episodes}: {total:.0f} puntos", flush=True)
    finally:
        env.close()
    print("Más puntos significa más pasos manteniendo el poste equilibrado.")
    print("Una demostración individual no sustituye los promedios del informe.\n")
    return returns


def print_summary():
    folder = PROJECT / "results/parte_15_mse_completo"
    rows = json.loads((folder / "comparacion/summary.json").read_text(encoding="utf-8"))
    print("\nRESULTADOS FINALES DEL TIER 1")
    print("Cada promedio incluye tres entrenamientos y 100 intentos por modelo.\n")
    for row in rows:
        print(f"{row['method']}: {row['final_mean_100_episodes']:.1f} puntos de media")
    print("\nDQN MSE: una ejecución obtuvo 500; las otras dos quedaron cerca de 101.")
    print("El mejor intento de una demostración no es el promedio del experimento.")
    input("\nPulsa Enter para volver al menú...")


def menu():
    while True:
        print("\nMOSTRAR TIER 1 — CARTPOLE")
        print("1. Ver acciones al azar")
        print("2. Ver Q-learning entrenado")
        print("3. Ver DQN entrenado: ejecución exitosa")
        print("4. Ver DQN: otra ejecución, para mostrar su variación")
        print("5. Abrir la gráfica de resultados")
        print("6. Ver el resumen de resultados")
        print("0. Salir")
        option = input("Elige una opción: ").strip()
        if option == "0":
            return
        try:
            if option in ("1", "2", "3", "4"):
                kind = {"1": "random", "2": "tabular", "3": "dqn", "4": "dqn"}[option]
                show(kind, seed=42 if option == "4" else 44)
            elif option == "5":
                file = PROJECT / "results/parte_15_mse_completo/comparacion/tier1_comparacion.png"
                os.startfile(str(file))
            elif option == "6":
                print_summary()
            else:
                print("Escribe un número de la lista.")
        except KeyboardInterrupt:
            print("\nDemostración interrumpida.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("random", "tabular", "dqn"))
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=44)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--headless", action="store_true", help="Comprobar sin abrir ventana")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes debe ser positivo")
    if args.agent is None:
        menu()
    else:
        show(args.agent, args.seed, args.episodes, args.headless)


if __name__ == "__main__":
    main()
