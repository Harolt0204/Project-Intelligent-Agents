"""Contrastar las predicciones Q finales con un límite teórico y revisar pérdidas."""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import QNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/parte_10_dqn"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    for folder in sorted(args.root.glob("seed_*")):
        config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        model = QNetwork(hidden_width=config["architecture"][1:-1]).eval()
        model.load_state_dict(torch.load(folder / "online_final.pt", map_location="cpu", weights_only=True))
        diagnostic_seed = config["seed"] + 90_000
        env = gym.make("CartPole-v1")
        states = []
        try:
            env.action_space.seed(diagnostic_seed)
            observation, _ = env.reset(seed=diagnostic_seed)
            for _ in range(1000):
                states.append(observation.copy())
                observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    observation, _ = env.reset()
        finally:
            env.close()
        with torch.inference_mode():
            inputs = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            values = model(inputs).numpy()
            first_hidden = model.layers[1](model.layers[0](inputs))
            second_hidden = model.layers[3](model.layers[2](first_hidden))
        # No confundir el retorno descontado Q con la suma sin descuento del episodio.
        upper_bound = 1 / (1 - config["gamma"])
        with (folder / "loss.csv").open(encoding="utf-8", newline="") as file:
            losses = [float(row["loss"]) for row in csv.DictReader(file)]
        record = {"seed": config["seed"], "diagnostic_env_seed": diagnostic_seed,
                  "observations": len(states), "policy_used_for_diagnostic_states": "random",
                  "q_min": float(values.min()), "q_max": float(values.max()),
                  "theoretical_upper_bound": upper_bound,
                  "fraction_entries_above_bound": float((values > upper_bound).mean()),
                  "fraction_entries_below_zero": float((values < 0).mean()),
                  "q_std_across_diagnostic_states_by_action": values.std(axis=0).tolist(),
                  "greedy_action_counts": np.bincount(values.argmax(axis=1), minlength=2).tolist(),
                  "first_hidden_units_zero_on_all_diagnostic_states": int((first_hidden == 0).all(dim=0).sum()),
                  "second_hidden_units_zero_on_all_diagnostic_states": int((second_hidden == 0).all(dim=0).sum()),
                  "mean_loss_first_1000_updates": float(np.mean(losses[:1000])),
                  "mean_loss_last_1000_updates": float(np.mean(losses[-1000:])),
                  "reason": "Reward <=1, gamma<1: discounted return <=1/(1-gamma); all rewards are nonnegative",
                  "limit": "Diagnóstico del modelo final en 1000 estados de una política aleatoria; no demuestra la causa única de un fallo"}
        (folder / "q_diagnostic.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
