"""Entrenamiento propio compartido por Tier 2 y Tier 3 en LunarLander-v3."""

from copy import deepcopy
import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import DQNAgent, QNetwork
from agents.double_dqn import DoubleDQNAgent
from agents.dqn_without_target import DQNWithoutTargetAgent
from agents.replay import ReplayBuffer

ROOT = Path(__file__).resolve().parents[1]
CLASSES = {"dqn": DQNAgent, "double": DoubleDQNAgent, "no_target": DQNWithoutTargetAgent}


def configuration():
    return json.loads((ROOT / "configs/tier23_lunarlander.json").read_text(encoding="utf-8"))


def plan(config):
    """Cinco DQN únicos, una ablación y Double DQN; baseline reutilizado."""
    groups = {"base": {"variant": "dqn", "parameters": dict(config["base"]), "role": "baseline"}}
    for key, values in config["studies"].items():
        for value in values:
            if value == config["base"][key]:
                continue
            name = f"{key}_{value:g}".replace(".", "p")
            groups[name] = {"variant": "dqn", "parameters": {**config["base"], key: value}, "role": key}
    groups["no_target"] = {"variant": "no_target", "parameters": dict(config["base"]), "role": "ablation"}
    groups["double"] = {"variant": "double", "parameters": dict(config["base"]), "role": "tier3"}
    return groups


def fingerprint(config, job, seed):
    return hashlib.sha256(json.dumps({"config": config, "job": job, "seed": seed}, sort_keys=True).encode()).hexdigest()


def make_env(config, render_mode=None):
    return gym.make(config["environment"], render_mode=render_mode, **config["environment_kwargs"])


def evaluate(network, config, seed, episodes, evaluation_base=None):
    """Otro entorno y RNG; no consumir estados aleatorios del entrenamiento."""
    evaluator = deepcopy(network).eval()
    base = config["evaluation_base"] if evaluation_base is None else evaluation_base
    rng = np.random.default_rng(base + 500_000 + 1000 * seed)
    env = make_env(config)
    returns = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=base + 1000 * seed + episode)
            total = 0.0
            while True:
                with torch.inference_mode():
                    scores = evaluator(torch.as_tensor(obs, dtype=torch.float32)).numpy()
                action = int(rng.choice(np.flatnonzero(scores == scores.max())))
                obs, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    break
            returns.append(float(total))
    finally:
        env.close()
    return returns


def probe_states(config, seed, count=256):
    env = make_env(config)
    observations = []
    try:
        env.action_space.seed(3_000_000 + seed)
        obs, _ = env.reset(seed=3_000_000 + seed)
        for _ in range(count):
            observations.append(obs.copy())
            obs, _, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
    return torch.tensor(np.asarray(observations), dtype=torch.float32)


def train(config, job, seed, folder):
    """Mismo bucle para todos los tratamientos; solo cambian parámetros o agente."""
    folder = Path(folder)
    p = job["parameters"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder.mkdir(parents=True, exist_ok=False)
    env = make_env(config)
    obs_dim, actions = int(env.observation_space.shape[0]), int(env.action_space.n)
    agent = CLASSES[job["variant"]](obs_dim=obs_dim, n_actions=actions,
            hidden_width=p["hidden_widths"], learning_rate=p["learning_rate"], gamma=p["gamma"],
            seed=seed, loss=p["loss"])
    replay = ReplayBuffer(p["buffer_size"], observation_shape=(obs_dim,), seed=4_000_000 + seed)
    probes = probe_states(config, seed)
    source_paths = ["agents/dqn.py", "agents/double_dqn.py", "agents/dqn_without_target.py", "agents/replay.py", "experiments/lunarlander.py"]
    run_config = {"input_hash": fingerprint(config, job, seed), "protocol": config, "job": job, "seed": seed,
        "architecture": [obs_dim, *p["hidden_widths"], actions], "environment_max_steps": env.spec.max_episode_steps,
        "environment_declared_threshold": env.spec.reward_threshold,
        "python": platform.python_version(), "platform": platform.platform(), "processor": platform.processor(),
        "logical_cpus": os.cpu_count(), "device": "cpu", "torch_threads": 1,
        "packages": {name: metadata.version(name) for name in ("gymnasium", "numpy", "torch", "Box2D")},
        "source_hashes": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths},
        "evaluation_reset_seed": "evaluation_base + 1000 * training_seed + episode",
        "evaluation_action_seed": "evaluation_base + 500000 + 1000 * training_seed",
        "replay_seed": 4_000_000 + seed,
        "target_network": job["variant"] != "no_target"}
    (folder / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    eval_rows, confirmations = [], []
    training_seconds = 0.0
    updates = syncs = 0
    started = time.perf_counter()

    def checkpoint(step, episodes):
        returns = evaluate(agent.online, config, seed, episodes)
        mean = float(np.mean(returns))
        with torch.inference_mode():
            online_q = agent.online(probes)
            if hasattr(agent, "target"):
                target_q = agent.target(probes)
                choices = online_q.argmax(1, keepdim=True)
                gap = float((target_q.max(1).values - target_q.gather(1, choices).squeeze(1)).mean())
            else:
                gap = None
        eval_rows.append({"training_steps": step, "episodes": episodes, "mean_return": mean,
            "std_return": float(np.std(returns)), "returns": json.dumps(returns),
            "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
            "q_probe_min": float(online_q.min()), "q_probe_max": float(online_q.max()),
            "target_selection_gap": gap})
        if episodes == config["final_eval_episodes"] or mean >= config["threshold"]:
            confirmed = returns if episodes == config["final_eval_episodes"] else evaluate(agent.online, config, seed, config["final_eval_episodes"])
            confirmations.append({"training_steps": step, "mean_return": float(np.mean(confirmed)),
                "threshold_met": bool(np.mean(confirmed) >= config["threshold"]), "returns": confirmed})
        with (folder / "evaluation.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(eval_rows[0]))
            writer.writeheader()
            writer.writerows(eval_rows)
        (folder / "threshold_checks.json").write_text(json.dumps(confirmations, indent=2), encoding="utf-8")
        print(f"{folder.parent.name} | semilla {seed} | paso {step:>6} | evaluación {episodes}: {mean:.2f}", flush=True)

    step = 0
    try:
        checkpoint(0, config["final_eval_episodes"])
        obs, _ = env.reset(seed=seed)
        episode = episode_steps = 0
        total = 0.0
        with ((folder / "training.csv").open("w", newline="", encoding="utf-8") as training_file,
              (folder / "loss.csv").open("w", newline="", encoding="utf-8") as loss_file):
            writer = csv.DictWriter(training_file, fieldnames=["training_steps", "episode", "episode_steps", "return",
                "epsilon", "terminated", "truncated", "budget_cut"])
            writer.writeheader()
            loss_writer = csv.writer(loss_file)
            loss_writer.writerow(["training_steps", "updates", "mean_loss", "prediction_mean", "target_mean"])
            timer = time.perf_counter()
            for step in range(1, config["steps"] + 1):
                fraction = min((step - 1) / p["epsilon_decay_steps"], 1)
                epsilon = 1 + fraction * (p["epsilon_end"] - 1)
                action = agent.choose_action(obs, epsilon)
                following, reward, terminated, truncated, _ = env.step(action)
                # Conservar el estado terminal real; reset ocurre después de almacenarlo.
                replay.add(obs, action, reward, following, terminated, truncated)
                if step >= p["learning_starts"] and len(replay) >= p["batch_size"] and step % p["train_every"] == 0:
                    losses = []
                    for _ in range(p["gradient_steps"]):
                        result = agent.learn(replay.sample(p["batch_size"]))
                        if not np.isfinite(result["loss"]):
                            raise FloatingPointError("Pérdida no finita; se conserva el diagnóstico, sin inventar una evaluación")
                        losses.append(result["loss"])
                        updates += 1
                    loss_writer.writerow([step, updates, float(np.mean(losses)),
                        float(np.mean(result["predictions_before"])), float(np.mean(result["targets"]))])
                if job["variant"] != "no_target" and step % p["target_sync_every"] == 0:
                    agent.sync_target()
                    syncs += 1
                obs = following
                total += reward
                episode_steps += 1
                if terminated or truncated or step == config["steps"]:
                    episode += 1
                    writer.writerow({"training_steps": step, "episode": episode, "episode_steps": episode_steps,
                        "return": total, "epsilon": epsilon, "terminated": terminated, "truncated": truncated,
                        "budget_cut": not (terminated or truncated)})
                    if terminated or truncated:
                        obs, _ = env.reset()
                    total = 0.0
                    episode_steps = 0
                if step % config["eval_every"] == 0 or step == config["steps"]:
                    training_seconds += time.perf_counter() - timer
                    training_file.flush()
                    loss_file.flush()
                    checkpoint(step, config["final_eval_episodes"] if step == config["steps"] else config["periodic_eval_episodes"])
                    timer = time.perf_counter()
        torch.save(agent.online.state_dict(), folder / "online_final.pt")
        if hasattr(agent, "target"):
            torch.save(agent.target.state_dict(), folder / "target_final.pt")
        reached = [row["training_steps"] for row in confirmations if row["threshold_met"]]
        summary = {"seed": seed, "input_hash": run_config["input_hash"], "before": eval_rows[0]["mean_return"],
            "after": eval_rows[-1]["mean_return"], "final_threshold_met": eval_rows[-1]["mean_return"] >= config["threshold"],
            "first_confirmed_threshold_step": min(reached) if reached else None,
            "training_seconds": training_seconds, "total_seconds": time.perf_counter() - started,
            "updates": updates, "target_syncs": syncs, "stored_experiences": len(replay), "complete": True}
        (folder / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    except Exception as error:
        (folder / "failure.json").write_text(json.dumps({"training_steps": step, "error": repr(error), "complete": False}, indent=2), encoding="utf-8")
        raise
    finally:
        env.close()


def load_model(folder):
    folder = Path(folder)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    architecture = config["architecture"]
    torch.set_num_threads(1)
    model = QNetwork(architecture[0], architecture[-1], architecture[1:-1]).eval()
    model.load_state_dict(torch.load(folder / "online_final.pt", map_location="cpu", weights_only=True))
    return model, config
