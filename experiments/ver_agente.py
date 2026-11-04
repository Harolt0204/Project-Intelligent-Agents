"""Comparar visualmente una tabla vacía y la tabla entrenada, sin aprendizaje."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer


def rollout(agent, discretizer, env_seed):
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    episode_return = 0.0
    try:
        observation, _ = env.reset(seed=env_seed)
        frames.append(Image.fromarray(env.render()).resize((450, 300)))
        while True:
            action = agent.choose_action(discretizer.encode(observation), epsilon=0)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            frames.append(Image.fromarray(env.render()).resize((450, 300)))
            if terminated or truncated:
                break
        return frames, {"return": episode_return, "steps": len(frames) - 1,
                        "terminated": terminated, "truncated": truncated}
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/parte_04_tabular/seed_42"))
    parser.add_argument("--env-seed", type=int, help="Por defecto: primer episodio de evaluación")
    parser.add_argument("--output", type=Path, default=Path("results/parte_05_visual"))
    args = parser.parse_args()
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    discretizer = CartPoleDiscretizer(config["bins"], -config["low"][1], -config["low"][3],
                                     config.get("spacing_power", 1.0),
                                     position_bins=config.get("bins_per_dimension", [None, None])[0],
                                     velocity_bins=config.get("bins_per_dimension", [None, None])[1])
    if not (np.allclose(discretizer.low, config["low"]) and
            np.allclose(discretizer.high, config["high"])):
        raise ValueError("Los rangos actuales no coinciden con los del entrenamiento")
    env_seed = config["seed"] + 10_000 if args.env_seed is None else args.env_seed
    action_seed = config["seed"] + 20_000
    before_agent = QLearningAgent(discretizer.shape, seed=action_seed,
                                  initial_q=config.get("initial_q", 0.0))
    after_agent = QLearningAgent(discretizer.shape, seed=action_seed)
    table = np.load(args.run / "q_table.npy", allow_pickle=False)
    if table.shape != after_agent.q.shape or not np.all(np.isfinite(table)):
        raise ValueError("Tabla Q incompatible o con valores no finitos")
    after_agent.q[:] = table

    before_frames, before = rollout(before_agent, discretizer, env_seed)
    after_frames, after = rollout(after_agent, discretizer, env_seed)
    # Evaluar no debe modificar los valores aprendidos.
    np.testing.assert_array_equal(after_agent.q, table)
    frames = []
    for index in range(max(len(before_frames), len(after_frames))):
        canvas = Image.new("RGB", (900, 370), "#edf2f7")
        draw = ImageDraw.Draw(canvas)
        for x, title, images, result in (
            (0, "ANTES: sin entrenamiento", before_frames, before),
            (450, "DESPUES: tabla entrenada", after_frames, after),
        ):
            current = min(index, len(images) - 1)
            canvas.paste(images[current], (x, 45))
            finished = index >= len(images) - 1
            status = "FINALIZADO" if finished else "en marcha"
            draw.text((x + 15, 8), title, fill="#172b4d")
            draw.text((x + 15, 25), f"Pasos: {current} | {status}", fill="#172b4d")
        draw.line((450, 0, 450, 345), fill="#94a3b8", width=2)
        draw.text((15, 352), "Mismo inicio | Sin actualizar tablas | Un intento ilustrativo, no un promedio", fill="#172b4d")
        frames.append(canvas)

    args.output.mkdir(parents=True, exist_ok=True)
    durations = [20] * len(frames)  # CartPole simula pasos de 0.02 segundos.
    durations[-1] = 1200
    frames[0].save(args.output / "comparacion.gif", save_all=True,
                   append_images=frames[1:], duration=durations, loop=0)
    frames[min(30, len(frames) - 1)].save(args.output / "vista_previa.png")
    result = {"source_run": str(args.run), "env_seed": env_seed,
              "action_seed": action_seed, "epsilon": 0, "learning": False,
              "selection": "Primer episodio de evaluación; no seleccionado por rendimiento",
              "before": before, "after": after,
              "note": "Una demostración visual no sustituye la evaluación de 100 episodios"}
    (args.output / "demostracion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Animación guardada en {args.output / 'comparacion.gif'}")


if __name__ == "__main__":
    main()
