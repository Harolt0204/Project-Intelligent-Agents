"""Evaluación adicional en estados iniciales no usados para ajustar Tier 1."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from agents.dqn import QNetwork
from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer
from experiments.entrenar_dqn import evaluate_network
from experiments.entrenar_tabular import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/parte_15_mse_completo/evaluacion_nueva"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output
    if (output / "summary.json").exists():
        raise SystemExit("La evaluación nueva ya está guardada; conservarla sin seleccionar otra muestra.")
    output.mkdir(parents=True, exist_ok=True)
    protocol = {"purpose": "Evaluación adicional al cerrar Tier 1, sin ajustes posteriores de estos modelos",
                "training_seeds": [42, 43, 44], "episodes_per_model": 100,
                "evaluation_function_seed": "1000000 + 1000 * training_seed",
                "environment_reset_seeds": "1010000 + 1000 * training_seed + episode_index",
                "evaluation_action_seed": "1020000 + 1000 * training_seed",
                "training": False, "model_selection": "Final de 200000 pasos",
                "parameter_selection_history": "La configuración MSE fue elegida tras experimentos exploratorios anteriores"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    groups = [(f"Q-learning ({bins} intervalos)", Path(f"results/parte_06_bins/bins_{bins}"), bins)
              for bins in (6, 10, 20)] + [
                  ("DQN (Smooth L1 inicial)", Path("results/parte_10_dqn"), None),
                  ("DQN (MSE)", Path("results/parte_15_mse_completo"), None)]
    records, summaries = [], []
    for label, root, bins in groups:
        means = []
        for seed in (42, 43, 44):
            folder = root / f"seed_{seed}"
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            assert config["training_steps"] == 200_000
            evaluation_seed = 1_000_000 + 1000 * seed
            if bins is None:
                model = QNetwork(hidden_width=config["architecture"][1:-1]).eval()
                model.load_state_dict(torch.load(folder / "online_final.pt", map_location="cpu", weights_only=True))
                returns = evaluate_network(model, evaluation_seed, 100)
            else:
                discretizer = CartPoleDiscretizer(bins)
                assert np.allclose(discretizer.low, config["low"]) and np.allclose(discretizer.high, config["high"])
                agent = QLearningAgent(discretizer.shape)
                agent.q[:] = np.load(folder / "q_table.npy", allow_pickle=False)
                returns = evaluate(agent, discretizer, evaluation_seed, 100)
            average = float(np.mean(returns))
            means.append(average)
            records.append({"method": label, "training_seed": seed,
                            "source_run": str(folder), "evaluation_function_seed": evaluation_seed,
                            "episodes": 100, "mean_return": average, "returns": returns})
            print(f"Evaluación nueva: {label}, semilla entrenada {seed}: {average:.2f}", flush=True)
        summaries.append({"method": label, "mean_return": float(np.mean(means)),
                          "std_across_training_seed_means": float(np.std(means)),
                          "means_by_training_seed": dict(zip(["42", "43", "44"], means)),
                          "models_meeting_475": sum(value >= 475 for value in means)})
    (output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
