"""Parte 4: entrenamiento piloto de Q-learning con evaluación independiente."""

import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np

from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer


def evaluate(agent, discretizer, seed, episodes):
    """Evaluar sin aprender ni consumir el generador aleatorio de entrenamiento."""
    evaluator = QLearningAgent(discretizer.shape, seed=seed + 20_000)
    evaluator.q[:] = agent.q
    env = gym.make("CartPole-v1")
    returns = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 10_000 + episode)
            episode_return = 0.0
            while True:
                state = discretizer.encode(observation)
                action = evaluator.choose_action(state, epsilon=0.0)
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return returns


def train(args, seed):
    folder = args.output / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    discretizer = CartPoleDiscretizer(args.bins)
    agent = QLearningAgent(discretizer.shape, alpha=0.1, gamma=0.99, seed=seed)
    config = {
        "purpose": getattr(args, "purpose", "Piloto didáctico; comparación formal de granularidades pendiente"),
        "environment": "CartPole-v1", "seed": seed, "bins": args.bins,
        "training_steps": args.steps, "alpha": agent.alpha, "gamma": agent.gamma,
        "epsilon_start": 1.0, "epsilon_end": 0.05, "decay_fraction": 0.8,
        "eval_every": args.eval_every, "periodic_eval_episodes": 10,
        "initial_and_final_eval_episodes": 100, "success_threshold": 475,
        "low": discretizer.low.tolist(), "high": discretizer.high.tolist(),
        "eval_reset_seeds": "seed + 10000 + episode; mismas condiciones por checkpoint",
        "eval_action_rng_seed": seed + 20_000,
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": os.cpu_count(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("gymnasium", "numpy")},
    }
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    eval_rows = []
    started = time.perf_counter()
    training_seconds = 0.0

    def checkpoint(step, episodes):
        returns = evaluate(agent, discretizer, seed, episodes)
        eval_rows.append({"training_steps": step, "episodes": episodes,
                          "mean_return": float(np.mean(returns)),
                          "std_return": float(np.std(returns)),
                          "training_seconds": training_seconds,
                          "total_seconds": time.perf_counter() - started,
                          "returns": returns})
        print(f"Semilla {seed} | pasos {step:>6} | evaluación ({episodes} intentos): "
              f"{np.mean(returns):.1f}", flush=True)

    checkpoint(0, 100)
    env = gym.make("CartPole-v1")
    coverage_outside = np.zeros(4, dtype=np.int64)
    visited = np.zeros(discretizer.shape, dtype=bool)
    episode_return = 0.0
    episode_steps = 0
    episode = 0
    try:
        observation, _ = env.reset(seed=seed)
        with (folder / "training.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["training_steps", "episode",
                                    "episode_steps", "return", "epsilon",
                                    "terminated", "truncated", "budget_cut"])
            writer.writeheader()
            timer = time.perf_counter()
            for step in range(1, args.steps + 1):
                coverage_outside += (observation < discretizer.low) | (observation > discretizer.high)
                state = discretizer.encode(observation)
                visited[state] = True
                # Decaimiento por pasos, independiente de la duración del episodio.
                fraction = min((step - 1) / (0.8 * args.steps), 1.0)
                epsilon = 1.0 + fraction * (0.05 - 1.0)
                action = agent.choose_action(state, epsilon)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state = discretizer.encode(next_observation)
                agent.update(state, action, reward, next_state, terminated)
                episode_return += reward
                episode_steps += 1
                observation = next_observation
                if terminated or truncated or step == args.steps:
                    episode += 1
                    writer.writerow({"training_steps": step, "episode": episode,
                                     "episode_steps": episode_steps, "return": episode_return,
                                     "epsilon": epsilon, "terminated": terminated,
                                     "truncated": truncated,
                                     "budget_cut": not (terminated or truncated)})
                    if terminated or truncated:
                        observation, _ = env.reset()
                    episode_return = 0.0
                    episode_steps = 0
                if step % args.eval_every == 0 or step == args.steps:
                    training_seconds += time.perf_counter() - timer
                    checkpoint(step, 100 if step == args.steps else 10)
                    timer = time.perf_counter()
    finally:
        env.close()

    np.save(folder / "q_table.npy", agent.q)
    with (folder / "evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({**row, "returns": json.dumps(row["returns"])})
    summary = {"seed": seed, "before": eval_rows[0]["mean_return"],
               "after": eval_rows[-1]["mean_return"],
               "final_threshold_met": eval_rows[-1]["mean_return"] >= 475,
               "training_seconds": training_seconds,
               "total_seconds": time.perf_counter() - started,
               "visited_states": int(visited.sum()), "possible_states": args.bins ** 4,
               "training_input_observations": args.steps,
               "outside_range_counts_by_dimension": coverage_outside.tolist()}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("results/parte_04_tabular"))
    args = parser.parse_args()
    if args.steps < 1 or args.bins < 2 or args.eval_every < 1:
        parser.error("steps y eval-every deben ser positivos; bins debe ser >= 2")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Las semillas deben ser diferentes y no negativas")
    if any((args.output / f"seed_{seed}").exists() for seed in args.seeds):
        parser.error("Ya existen resultados con esas semillas; utiliza otro --output")
    summaries = [train(args, seed) for seed in args.seeds]
    before = [row["before"] for row in summaries]
    after = [row["after"] for row in summaries]
    aggregate = {"runs": summaries, "mean_before": float(np.mean(before)),
                 "mean_after": float(np.mean(after)),
                 "std_across_seed_means": float(np.std(after)),
                 "note": "Piloto; no sustituye la comparación formal de Tier 1"}
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nPromedio entre semillas: antes={np.mean(before):.1f}; después={np.mean(after):.1f}")
    print(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()
