"""Ver los agentes ya entrenados en una ventana, sin volver a aprender."""

import argparse
import json
import os
from pathlib import Path

import gymnasium as gym
import numpy as np


PROJECT = Path(__file__).resolve().parents[1]


def model_folder(kind, seed, original=False):
    pointer = PROJECT / "results/tier1_actual.json"
    data = json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() and not original else {
        "dqn_root": "results/parte_15_mse_completo", "tabular_root": "results/parte_06_bins"}
    root = PROJECT / (data["tabular_root"] if kind == "tabular" else data["dqn_root"])
    if kind == "tabular":
        root = root / f"bins_{data.get('tabular_bins', 10)}"
    return root / f"seed_{seed}"


def load_policy(kind, seed, original=False):
    if kind == "random":
        rng = np.random.default_rng(seed + 20_000)
        return lambda observation: int(rng.integers(2))

    if kind == "tabular":
        from agents.tabular import QLearningAgent
        from envs.discretizers import CartPoleDiscretizer

        folder = model_folder(kind, seed, original)
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        discretizer = CartPoleDiscretizer(config["bins"], -config["low"][1], -config["low"][3],
                                         config.get("spacing_power", 1.0),
                                         position_bins=config.get("bins_per_dimension", [None, None])[0],
                                         velocity_bins=config.get("bins_per_dimension", [None, None])[1])
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
    folder = model_folder(kind, seed, original)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    network = QNetwork(hidden_width=config["architecture"][1:-1]).eval()
    network.load_state_dict(torch.load(folder / "online_final.pt", map_location="cpu", weights_only=True))
    rng = np.random.default_rng(seed + 20_000)

    def choose(observation):
        with torch.inference_mode():
            scores = network(torch.as_tensor(observation, dtype=torch.float32)).numpy()
        return int(rng.choice(np.flatnonzero(scores == scores.max())))

    return choose


def show(kind, seed=42, episodes=3, headless=False, original=False):
    labels = {"random": "Acciones al azar", "tabular": "Q-learning entrenado", "dqn": "DQN con MSE"}
    print(f"\n{labels[kind]}" + (f" | ejecución entrenada: {seed}" if kind != "random" else ""), flush=True)
    if kind != "random":
        print("Cargando el modelo guardado. Durante esta muestra no aprende.", flush=True)
    if kind != "random":
        folder = model_folder(kind, seed, original)
        mean = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["after"]
        means = [json.loads((folder.parent / f"seed_{s}/summary.json").read_text(encoding="utf-8"))["after"] for s in (42, 43, 44)]
        print(f"Modelo: {folder.relative_to(PROJECT)}")
        if kind == "tabular":
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            print(f"Intervalos por observación: {config.get('bins_per_dimension', [config['bins']] * 4)}")
        print(f"Su media de 100 episodios: {mean:.2f}. Media entre tres semillas: {np.mean(means):.2f}.")
    policy = load_policy(kind, seed, original)
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
    data = json.loads((PROJECT / "results/tier1_actual.json").read_text(encoding="utf-8"))
    rows = json.loads((PROJECT / data["comparison"] / "summary.json").read_text(encoding="utf-8"))
    print("\nRESULTADOS FINALES DEL TIER 1")
    print("Cada promedio incluye tres entrenamientos y 100 intentos por modelo.\n")
    for row in rows:
        print(f"{row['method']}: {row['final_mean_100_episodes']:.1f} puntos de media")
    for row in rows:
        print(f"{row['method']}: {row['final_successful_seeds']}/3 semillas alcanzaron 475 en evaluación final.")
    print("El mejor intento de una demostración no es el promedio del experimento.")
    input("\nPulsa Enter para volver al menú...")


def menu():
    while True:
        print("\nMOSTRAR TIER 1 — CARTPOLE")
        print("1. Ver acciones al azar")
        print("2. Ver Q-learning entrenado")
        print("3. Ver DQN actual entrenado")
        print("4. Ver DQN anterior, para comparar antes y después")
        print("5. Abrir la gráfica de resultados")
        print("6. Ver el resumen de resultados")
        print("0. Salir")
        option = input("Elige una opción: ").strip()
        if option == "0":
            return
        try:
            if option in ("1", "2", "3", "4"):
                kind = {"1": "random", "2": "tabular", "3": "dqn", "4": "dqn"}[option]
                show(kind, seed=42, original=option == "4")
            elif option == "5":
                data = json.loads((PROJECT / "results/tier1_actual.json").read_text(encoding="utf-8"))
                file = PROJECT / data["comparison"] / "tier1_comparacion.png"
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
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), default=42)
    parser.add_argument("--original", action="store_true", help="Mostrar modelos previos a la mejora")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--headless", action="store_true", help="Comprobar sin abrir ventana")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("episodes debe ser positivo")
    if args.agent is None:
        menu()
    else:
        show(args.agent, args.seed, args.episodes, args.headless, args.original)


if __name__ == "__main__":
    main()
