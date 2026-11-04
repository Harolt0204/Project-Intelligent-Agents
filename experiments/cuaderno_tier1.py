"""Herramientas compartidas por el notebook, la reproducción y la demostración.

Los algoritmos viven en agents/ y envs/: el notebook usa esas implementaciones.
"""

import argparse
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import torch

from agents.dqn import QNetwork
from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer

PROJECT = Path(__file__).resolve().parents[1]
CURRENT = PROJECT / "results/tier1_actual.json"


def parametros():
    """Devolver una copia nueva de la configuración compartida y editable."""
    return json.loads((PROJECT / "configs/tier1_mejorado.json").read_text(encoding="utf-8"))


def actual():
    return json.loads(CURRENT.read_text(encoding="utf-8"))


def codigo(objeto):
    """Mostrar el código real, sin mantener una segunda copia en el notebook."""
    from IPython.display import Code, display
    display(Code(inspect.getsource(objeto), language="python"))


def cargar_modelo(folder):
    """Reconstruir el agente según sus parámetros guardados, sin aprender."""
    folder = Path(folder)
    if not folder.is_absolute():
        folder = PROJECT / folder
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    if (folder / "online_final.pt").exists():
        torch.set_num_threads(1)
        model = QNetwork(hidden_width=config["architecture"][1:-1]).eval()
        model.load_state_dict(torch.load(folder / "online_final.pt", weights_only=True, map_location="cpu"))
        return model, None, config
    discretizer = CartPoleDiscretizer(config["bins"], -config["low"][1], -config["low"][3])
    if not np.allclose(discretizer.low, config["low"]) or not np.allclose(discretizer.high, config["high"]):
        raise ValueError("El discretizador no coincide con el modelo guardado")
    model = QLearningAgent(discretizer.shape, seed=config["seed"] + 20_000)
    table = np.load(folder / "q_table.npy", allow_pickle=False)
    if table.shape != model.q.shape or not np.isfinite(table).all():
        raise ValueError("La tabla guardada es incompatible")
    model.q[:] = table
    return model, discretizer, config


def evaluar_guardado(folder, evaluation_seed=None, episodes=100):
    from experiments.entrenar_dqn import evaluate_network
    from experiments.entrenar_tabular import evaluate
    model, discretizer, config = cargar_modelo(folder)
    seed = config["seed"] if evaluation_seed is None else evaluation_seed
    if discretizer is None:
        return evaluate_network(model, seed, episodes)
    return evaluate(model, discretizer, seed, episodes)


def demostrar(folder, episodes=1, inline=True, headless=False):
    """Mismo modelo y semillas de evaluación; animación dentro del notebook."""
    if episodes < 1:
        raise ValueError("episodes debe ser positivo")
    model, discretizer, config = cargar_modelo(folder)
    rng = np.random.default_rng(config["seed"] + 20_000)
    env = gym.make("CartPole-v1", render_mode=None if headless else ("rgb_array" if inline else "human"))
    frames, returns = [], []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=config["seed"] + 10_000 + episode)
            total = 0
            while True:
                if inline and not headless:
                    frames.append(env.render())
                if discretizer is None:
                    with torch.inference_mode():
                        values = model(torch.as_tensor(observation, dtype=torch.float32)).numpy()
                    action = int(rng.choice(np.flatnonzero(values == values.max())))
                else:
                    action = model.choose_action(discretizer.encode(observation), 0)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                if terminated or truncated:
                    if inline and not headless:
                        frames.append(env.render())
                    break
            returns.append(total)
    finally:
        env.close()
    print(f"Semilla {config['seed']}: {returns}. Esta muestra no sustituye la evaluación de 100 episodios.")
    if frames:
        from IPython.display import HTML, display
        from matplotlib import animation
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=(6, 4))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        image = ax.imshow(frames[0])
        ax.axis("off")
        def update(index):
            image.set_data(frames[index])
            return (image,)
        anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=20, blit=True)
        html = anim.to_jshtml(embed_frames=True, default_mode="once")
        fig.clear()
        display(HTML(html))
    return returns


def resumen():
    """Tabla HTML legible en VS Code, sin requerir pandas."""
    from html import escape
    from IPython.display import HTML, display
    data = actual()
    rows = json.loads((PROJECT / data["comparison"] / "summary.json").read_text(encoding="utf-8"))
    body = "".join(
        f"<tr><td>{escape(row['method'])}</td><td>{row['final_mean_100_episodes']:.2f}</td>"
        f"<td>{row['std_across_seed_means']:.2f}</td><td>{row['final_successful_seeds']}/3</td>"
        f"<td>{row['mean_training_seconds']:.1f} s</td></tr>" for row in rows)
    display(HTML("<table><tr><th>Método</th><th>Media final</th><th>DE entre semillas</th>"
                 "<th>Semillas ≥475</th><th>Tiempo de entrenamiento</th></tr>" + body + "</table>"))
    return rows


def entrenar(config, output, metodo="todos"):
    """Entrenar desde parámetros del notebook; nunca reemplazar resultados."""
    from experiments.entrenar_dqn import train as train_dqn
    from experiments.entrenar_tabular import train as train_tabular
    output = Path(output)
    if not output.is_absolute():
        output = PROJECT / output
    if output.exists():
        raise FileExistsError(f"Conservamos resultados previos. Escoge otra carpeta: {output}")
    if metodo not in ("todos", "dqn", "tabular"):
        raise ValueError("metodo debe ser todos, dqn o tabular")
    if (config["steps"] < 1000 or config["eval_every"] < 1 or not config["seeds"] or not config["bins"]
            or min(config["seeds"]) < 0 or len(set(config["seeds"])) != len(config["seeds"])):
        raise ValueError("Revisar pasos, intervalo y semillas independientes")
    dqn, tabular = config["dqn"], config["tabular"]
    numeric = [dqn["learning_rate"], dqn["gamma"], dqn["epsilon_end"], tabular["alpha"],
               tabular["gamma"], tabular["epsilon_end"], tabular["velocity_limit"], tabular["angular_velocity_limit"]]
    if (not np.isfinite(numeric).all() or min(*dqn["hidden_widths"], dqn["batch_size"], dqn["buffer_size"], dqn["train_every"],
            dqn["target_sync_every"], dqn["learning_starts"], dqn["epsilon_decay_steps"]) < 1
            or dqn["batch_size"] > dqn["buffer_size"] or dqn["learning_rate"] <= 0
            or dqn["loss"] not in ("mse", "smooth_l1") or len(dqn["hidden_widths"]) != 2
            or not 0 <= dqn["gamma"] < 1 or not 0 <= dqn["epsilon_end"] <= 1
            or min(config["bins"]) < 2 or not 0 < tabular["alpha"] <= 1
            or not 0 <= tabular["gamma"] < 1 or not 0 <= tabular["epsilon_end"] <= 1
            or min(tabular["epsilon_decay_steps"], tabular["velocity_limit"], tabular["angular_velocity_limit"]) <= 0):
        raise ValueError("Parámetros de entrenamiento fuera de rango")
    output.mkdir(parents=True)
    (output / "protocol.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    groups = []
    if metodo in ("todos", "tabular"):
        groups += [(f"bins_{bins}", train_tabular, {**config["tabular"], "bins": bins}) for bins in config["bins"]]
    if metodo in ("todos", "dqn"):
        groups.append(("dqn", train_dqn, config["dqn"]))
    result = {}
    for label, trainer, parameters in groups:
        args = SimpleNamespace(**parameters, steps=config["steps"], eval_every=config["eval_every"],
                               output=output / label, purpose="Comparación Tier 1 con parámetros documentados")
        runs = [trainer(args, seed) for seed in config["seeds"]]
        final = [run["after"] for run in runs]
        aggregate = {"runs": runs, "mean_after": float(np.mean(final)),
                     "std_across_seed_means": float(np.std(final))}
        (args.output / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        result[label] = aggregate
    return result


def main():
    parser = argparse.ArgumentParser(description="Reproducir el entrenamiento utilizado por el notebook")
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/tier1_mejorado.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metodo", choices=("todos", "dqn", "tabular"), default="todos")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    entrenar(config, args.output, args.metodo)


if __name__ == "__main__":
    main()
