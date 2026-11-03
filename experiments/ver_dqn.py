"""Guardar una animación del modelo DQN final, sin volver a entrenarlo."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw
import torch

from agents.dqn import QNetwork


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/parte_10_dqn/seed_42"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    config = json.loads((args.run / "config.json").read_text(encoding="utf-8"))
    network = QNetwork(hidden_width=config["architecture"][1]).eval()
    network.load_state_dict(torch.load(args.run / "online_final.pt", map_location="cpu", weights_only=True))
    rng = np.random.default_rng(config["seed"] + 20_000)
    env_seed = config["seed"] + 10_000
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    frames = []
    steps = 0
    episode_return = 0.0

    def frame():
        canvas = Image.new("RGB", (600, 440), "#edf2f7")
        canvas.paste(Image.fromarray(env.render()), (0, 40))
        ImageDraw.Draw(canvas).text((12, 12), f"DQN entrenado | pasos: {steps} | sin aprendizaje", fill="#172b4d")
        frames.append(canvas)

    try:
        observation, _ = env.reset(seed=env_seed)
        frame()
        while True:
            with torch.inference_mode():
                values = network(torch.as_tensor(observation, dtype=torch.float32)).numpy()
            action = int(rng.choice(np.flatnonzero(values == values.max())))
            observation, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            episode_return += reward
            frame()
            if terminated or truncated:
                break
    finally:
        env.close()
    output = args.run / "visual"
    output.mkdir(parents=True, exist_ok=True)
    durations = [20] * len(frames)
    durations[-1] = 1200
    frames[0].save(output / "dqn_final.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    frames[min(100, len(frames) - 1)].save(output / "vista_previa.png")
    result = {"selection": "Primera semilla y primer episodio de evaluación; sin buscar el mejor intento",
              "env_seed": env_seed, "return": episode_return, "steps": steps,
              "terminated": terminated, "truncated": truncated, "model": "online_final.pt"}
    (output / "demostracion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
