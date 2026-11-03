"""Parte 2: observar cómo varios estados continuos comparten una casilla."""

import argparse

import gymnasium as gym
import numpy as np

from envs.discretizers import CartPoleDiscretizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=args.seed)
    finally:
        env.close()

    print("Estado real inicial:", observation)
    print("Índices numerados desde 0. Todavía no hay aprendizaje.\n")
    for bins in (6, 10, 20):
        discretizer = CartPoleDiscretizer(bins)
        state = discretizer.encode(observation)
        states = bins ** 4
        # Dos acciones por estado; float64 utiliza 8 bytes por valor Q.
        q_table = np.zeros(discretizer.shape + (2,), dtype=np.float64)
        print(f"{bins} bins/variable: estado={state}, estados posibles={states:,}, "
              f"valores Q={q_table.size:,}, memoria={q_table.nbytes / 1024**2:.3f} MiB")
        print(f"  Valores iniciales [izquierda, derecha]: {q_table[state]}")

    discretizer = CartPoleDiscretizer(6)
    print("\nEjemplo con 6 bins:")
    for values in ([0.01, 0.02, 0.03, 0.04], [0.02, 0.03, 0.04, 0.05],
                   [0.01, 0.02, 0.18, 0.04]):
        print(f"  {values} -> {discretizer.encode(values)}")
    print("Los dos primeros estados comparten casilla; el tercero cambia el ángulo.")
    print("Agrupar permite reutilizar experiencia, pero pierde detalle.")


if __name__ == "__main__":
    main()
