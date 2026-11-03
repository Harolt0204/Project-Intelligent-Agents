"""Entrenar DQN propio en CartPole, con presupuesto comparable a Q-learning."""

import argparse
from copy import deepcopy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import DQNAgent
from agents.replay import ReplayBuffer


def evaluate_network(network, seed, episodes):
    """Evaluar una copia sin modificar pesos ni generadores de entrenamiento."""
    evaluator = deepcopy(network).eval()
    rng = np.random.default_rng(seed + 20_000)
    env = gym.make("CartPole-v1")
    returns = []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + 10_000 + episode)
            episode_return = 0.0
            while True:
                with torch.inference_mode():
                    values = evaluator(torch.as_tensor(observation, dtype=torch.float32)).numpy()
                action = int(rng.choice(np.flatnonzero(values == values.max())))
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += reward
                if terminated or truncated:
                    break
            returns.append(episode_return)
    finally:
        env.close()
    return returns


def collect_probe_states(seed, count=256):
    """Estados para diagnóstico, nunca añadidos a la memoria de entrenamiento."""
    env = gym.make("CartPole-v1")
    states = []
    try:
        env.action_space.seed(seed + 90_000)
        observation, _ = env.reset(seed=seed + 90_000)
        for _ in range(count):
            states.append(observation.copy())
            observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                observation, _ = env.reset()
    finally:
        env.close()
    return torch.as_tensor(np.asarray(states), dtype=torch.float32)


def train(args, seed):
    folder = args.output / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    loss_name = getattr(args, "loss", "smooth_l1")
    agent = DQNAgent(seed=seed, learning_rate=args.learning_rate, gamma=0.99, loss=loss_name)
    memory = ReplayBuffer(50_000, seed=seed + 30_000)
    decay_steps = args.epsilon_decay_steps if args.epsilon_decay_steps is not None else 0.8 * args.steps
    probe_states = collect_probe_states(seed)
    config = {"environment": "CartPole-v1", "algorithm": "vanilla DQN propio",
              "seed": seed, "training_steps": args.steps, "architecture": [4, 64, 64, 2],
              "learning_rate": args.learning_rate, "gamma": 0.99, "buffer_size": 50_000,
              "batch_size": 64, "learning_starts": 1000, "train_every": 4,
              "target_sync_every_environment_steps": args.target_sync_every, "optimizer": "Adam",
              "loss": "MSE" if loss_name == "mse" else "Smooth L1", "gradient_norm_limit": 10,
              "epsilon_start": 1, "epsilon_end": 0.05, "decay_fraction": decay_steps / args.steps,
              "epsilon_decay_steps": decay_steps,
              "q_probe": {"states": 256, "policy": "random", "seed": seed + 90_000,
                          "used_for_training": False},
              "eval_every": 10_000, "periodic_eval_episodes": 10,
              "initial_final_eval_episodes": 100, "success_threshold": 475,
              "threshold_confirmation": "100 episodios cuando el control de 10 obtiene >=475",
              "eval_reset_seeds": "seed + 10000 + episode",
              "eval_action_rng_seed": seed + 20_000, "replay_seed": seed + 30_000,
              "python": platform.python_version(), "platform": platform.platform(),
              "processor": platform.processor(), "logical_cpus": os.cpu_count(),
              "device": "cpu", "torch_threads": 1, "deterministic_algorithms": True,
              "packages": {name: importlib.metadata.version(name)
                           for name in ("gymnasium", "numpy", "torch")}}
    (folder / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    training_seconds = 0.0
    updates = 0
    syncs = 0
    eval_rows = []
    confirmations = []

    def checkpoint(step, episodes):
        returns = evaluate_network(agent.online, seed, episodes)
        mean = float(np.mean(returns))
        with torch.inference_mode():
            probe_values = agent.online(probe_states)
        eval_rows.append({"training_steps": step, "episodes": episodes,
                          "mean_return": mean, "std_return": float(np.std(returns)),
                          "training_seconds": training_seconds,
                          "total_seconds": time.perf_counter() - started,
                          "returns": json.dumps(returns),
                          "q_probe_min": float(probe_values.min()),
                          "q_probe_max": float(probe_values.max()),
                          "q_probe_fraction_above_100": float((probe_values > 100).float().mean())})
        if episodes == 100:
            confirmations.append({"training_steps": step, "mean_return": mean,
                                  "threshold_met": mean >= 475, "returns": returns})
        elif mean >= 475:
            confirmation = evaluate_network(agent.online, seed, 100)
            confirmations.append({"training_steps": step, "mean_return": float(np.mean(confirmation)),
                                  "threshold_met": bool(np.mean(confirmation) >= 475),
                                  "returns": confirmation})
        with (folder / "evaluation.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(eval_rows[0]))
            writer.writeheader()
            writer.writerows(eval_rows)
        (folder / "threshold_checks.json").write_text(json.dumps(confirmations, indent=2), encoding="utf-8")
        print(f"DQN target={args.target_sync_every} semilla {seed} | pasos {step:>6} | "
              f"evaluación ({episodes}): {mean:.1f} | Q máximo: {float(probe_values.max()):.1f}", flush=True)

    checkpoint(0, 100)
    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=seed)
        episode = 0
        episode_steps = 0
        episode_return = 0.0
        with ((folder / "training.csv").open("w", newline="", encoding="utf-8") as train_file,
              (folder / "loss.csv").open("w", newline="", encoding="utf-8") as loss_file):
            writer = csv.DictWriter(train_file, fieldnames=["training_steps", "episode", "episode_steps",
                                    "return", "epsilon", "terminated", "truncated", "budget_cut"])
            writer.writeheader()
            loss_writer = csv.writer(loss_file)
            loss_writer.writerow(["training_steps", "update", "loss"])
            timer = time.perf_counter()
            for step in range(1, args.steps + 1):
                fraction = min((step - 1) / decay_steps, 1.0)
                epsilon = 1 + fraction * (0.05 - 1)
                action = agent.choose_action(observation, epsilon)
                following, reward, terminated, truncated, _ = env.step(action)
                memory.add(observation, action, reward, following, terminated, truncated)
                if step >= 1000 and step % 4 == 0:
                    metrics = agent.learn(memory.sample(64))
                    updates += 1
                    loss_writer.writerow([step, updates, metrics["loss"]])
                if step % args.target_sync_every == 0:
                    agent.sync_target()
                    syncs += 1
                observation = following
                episode_steps += 1
                episode_return += reward
                if terminated or truncated or step == args.steps:
                    episode += 1
                    writer.writerow({"training_steps": step, "episode": episode,
                                     "episode_steps": episode_steps, "return": episode_return,
                                     "epsilon": epsilon, "terminated": terminated, "truncated": truncated,
                                     "budget_cut": not (terminated or truncated)})
                    if terminated or truncated:
                        observation, _ = env.reset()
                    episode_return = 0.0
                    episode_steps = 0
                if step % 10_000 == 0 or step == args.steps:
                    training_seconds += time.perf_counter() - timer
                    train_file.flush()
                    loss_file.flush()
                    checkpoint(step, 100 if step == args.steps else 10)
                    timer = time.perf_counter()
    finally:
        env.close()
    torch.save(agent.online.state_dict(), folder / "online_final.pt")
    torch.save(agent.target.state_dict(), folder / "target_final.pt")
    reached = [row["training_steps"] for row in confirmations if row["threshold_met"]]
    summary = {"seed": seed, "before": eval_rows[0]["mean_return"], "after": eval_rows[-1]["mean_return"],
               "final_threshold_met": eval_rows[-1]["mean_return"] >= 475,
               "first_confirmed_threshold_step": min(reached) if reached else None,
               "threshold_note": "Primer control observado; no es el instante exacto de aprendizaje",
               "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
               "updates": updates, "target_syncs": syncs, "stored_experiences": len(memory)}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/parte_10_dqn"))
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--loss", choices=("smooth_l1", "mse"), default="smooth_l1")
    parser.add_argument("--target-sync-every", type=int, default=500)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None,
                        help="Si se omite, usa el 80%% del presupuesto de pasos")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    args = parser.parse_args()
    if args.steps < 1000 or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Usar al menos 1000 pasos y semillas diferentes no negativas")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate debe ser un número finito positivo")
    if args.target_sync_every < 1 or (args.epsilon_decay_steps is not None and args.epsilon_decay_steps < 1):
        parser.error("Los intervalos de copia y decaimiento deben ser positivos")
    if any((args.output / f"seed_{seed}").exists() for seed in args.seeds):
        parser.error("Ya existen resultados; utiliza otro --output")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {"seeds": args.seeds, "training_steps_per_run": args.steps,
                "learning_rate": args.learning_rate,
                "loss": args.loss,
                "target_sync_every": args.target_sync_every,
                "epsilon_decay_steps": args.epsilon_decay_steps if args.epsilon_decay_steps is not None else 0.8 * args.steps,
                "parameter_selection": "Tasa fijada antes de esta ejecución; conservar cada experimento en una carpeta distinta",
                "model_selection": "Se entrega el modelo final, sin elegir el mejor checkpoint",
                "comparison": "Comparar solo ejecuciones con igual presupuesto y calendario de exploración",
                "order": "Secuencial en CPU, sin ejecutar otros entrenamientos a la vez"}
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summaries = [train(args, seed) for seed in args.seeds]
    aggregate = {"runs": summaries, "mean_before": float(np.mean([r["before"] for r in summaries])),
                 "mean_after": float(np.mean([r["after"] for r in summaries])),
                 "std_across_seed_means": float(np.std([r["after"] for r in summaries]))}
    (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"DQN final: {aggregate['mean_after']:.2f} ± {aggregate['std_across_seed_means']:.2f}", flush=True)


if __name__ == "__main__":
    main()
