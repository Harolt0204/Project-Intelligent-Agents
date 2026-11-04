"""Entrenar Q-learning propio en CartPole con discretización configurable."""

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
    discretizer = CartPoleDiscretizer(args.bins, getattr(args, "velocity_limit", 3.0),
                                     getattr(args, "angular_velocity_limit", 3.5),
                                     getattr(args, "spacing_power", 1.0),
                                     getattr(args, "position_bins", None),
                                     getattr(args, "velocity_bins", None))
    agent = QLearningAgent(discretizer.shape, alpha=getattr(args, "alpha", 0.1),
                          gamma=getattr(args, "gamma", 0.99), seed=seed,
                          initial_q=getattr(args, "initial_q", 0.0))
    epsilon_end = getattr(args, "epsilon_end", 0.05)
    decay_steps = getattr(args, "epsilon_decay_steps", None) or 0.8 * args.steps
    alpha_start = agent.alpha
    alpha_end = getattr(args, "alpha_end", None)
    alpha_end = alpha_start if alpha_end is None else alpha_end
    alpha_decay_steps = getattr(args, "alpha_decay_steps", None) or args.steps
    config = {
        "purpose": getattr(args, "purpose", "Q-learning tabular propio; consultar protocolo y presupuesto registrados"),
        "environment": "CartPole-v1", "seed": seed, "bins": args.bins,
        "training_steps": args.steps, "alpha": agent.alpha, "gamma": agent.gamma,
        "initial_q": getattr(args, "initial_q", 0.0),
        "alpha_end": alpha_end, "alpha_decay_steps": alpha_decay_steps,
        "alpha_schedule": "Lineal por pasos; constante si alpha_end=alpha",
        "epsilon_start": 1.0, "epsilon_end": epsilon_end, "decay_fraction": decay_steps / args.steps,
        "epsilon_decay_steps": decay_steps,
        "eval_every": args.eval_every, "periodic_eval_episodes": 10,
        "initial_and_final_eval_episodes": 100, "success_threshold": 475,
        "threshold_confirmation": "100 episodios cuando el control de 10 obtiene >=475",
        "low": discretizer.low.tolist(), "high": discretizer.high.tolist(),
        "spacing_power": discretizer.spacing_power,
        "bins_per_dimension": list(discretizer.shape),
        "edges": [edge.tolist() for edge in discretizer.edges],
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
    confirmations = []

    def checkpoint(step, episodes):
        returns = evaluate(agent, discretizer, seed, episodes)
        if episodes == 100 or np.mean(returns) >= 475:
            confirmed = returns if episodes == 100 else evaluate(agent, discretizer, seed, 100)
            confirmations.append({"training_steps": step, "mean_return": float(np.mean(confirmed)),
                                  "threshold_met": bool(np.mean(confirmed) >= 475), "returns": confirmed})
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
                                    "terminated", "truncated", "budget_cut", "alpha"])
            writer.writeheader()
            timer = time.perf_counter()
            for step in range(1, args.steps + 1):
                coverage_outside += (observation < discretizer.low) | (observation > discretizer.high)
                state = discretizer.encode(observation)
                visited[state] = True
                # Decaimiento por pasos, independiente de la duración del episodio.
                fraction = min((step - 1) / decay_steps, 1.0)
                epsilon = 1.0 + fraction * (epsilon_end - 1.0)
                alpha_fraction = min((step - 1) / max(alpha_decay_steps - 1, 1), 1.0)
                agent.alpha = alpha_start + alpha_fraction * (alpha_end - alpha_start)
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
                                     "budget_cut": not (terminated or truncated), "alpha": agent.alpha})
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
    (folder / "threshold_checks.json").write_text(json.dumps(confirmations, indent=2), encoding="utf-8")
    reached = [row["training_steps"] for row in confirmations if row["threshold_met"]]
    with (folder / "evaluation.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(eval_rows[0]))
        writer.writeheader()
        for row in eval_rows:
            writer.writerow({**row, "returns": json.dumps(row["returns"])})
    summary = {"seed": seed, "before": eval_rows[0]["mean_return"],
               "after": eval_rows[-1]["mean_return"],
               "final_threshold_met": eval_rows[-1]["mean_return"] >= 475,
               "first_confirmed_threshold_step": min(reached) if reached else None,
               "training_seconds": training_seconds,
               "total_seconds": time.perf_counter() - started,
               "visited_states": int(visited.sum()), "possible_states": int(np.prod(discretizer.shape)),
               "training_input_observations": args.steps,
               "outside_range_counts_by_dimension": coverage_outside.tolist()}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30_000)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--position-bins", type=int, help="Si se omite, usa bins")
    parser.add_argument("--velocity-bins", type=int, help="Si se omite, usa bins")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--eval-every", type=int, default=5_000)
    parser.add_argument("--output", type=Path, default=Path("results/parte_04_tabular"))
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--alpha-end", type=float, help="Si se omite, alpha permanece constante")
    parser.add_argument("--alpha-decay-steps", type=int, help="Por defecto usa todo el presupuesto")
    parser.add_argument("--initial-q", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--velocity-limit", type=float, default=3.0)
    parser.add_argument("--angular-velocity-limit", type=float, default=3.5)
    parser.add_argument("--spacing-power", type=float, default=1.0,
                        help="1: intervalos uniformes; >1: más detalle cerca del equilibrio")
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int)
    args = parser.parse_args()
    if any(n is not None and n < 2 for n in (args.position_bins, args.velocity_bins)):
        parser.error("position-bins y velocity-bins deben ser >=2")
    if not np.isfinite(args.spacing_power) or args.spacing_power <= 0:
        parser.error("spacing-power debe ser finito y positivo")
    if args.alpha_end is not None and not 0 < args.alpha_end <= 1:
        parser.error("alpha-end debe estar en (0, 1]")
    if args.alpha_decay_steps is not None and args.alpha_decay_steps < 1:
        parser.error("alpha-decay-steps debe ser positivo")
    if not np.isfinite(args.initial_q):
        parser.error("initial-q debe ser finito")
    if not 0 < args.alpha <= 1 or not 0 <= args.gamma < 1 or not 0 <= args.epsilon_end <= 1:
        parser.error("Revisar alpha, gamma (<1) y epsilon-end")
    if args.epsilon_decay_steps is not None and args.epsilon_decay_steps < 1:
        parser.error("epsilon-decay-steps debe ser positivo")
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
                 "note": "Estadísticas por configuración; la comparación entre métodos se genera por separado"}
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\nPromedio entre semillas: antes={np.mean(before):.1f}; después={np.mean(after):.1f}")
    print(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()
