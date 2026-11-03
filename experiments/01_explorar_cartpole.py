"""Primer ejercicio: interactuar con CartPole sin entrenar un agente."""

import argparse

import gymnasium as gym


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true", help="Mostrar la simulación")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes debe ser al menos 1")

    # Sin --render funciona sin abrir ventanas y se ejecuta más rápido.
    env = gym.make("CartPole-v1", render_mode="human" if args.render else None)
    try:
        # El generador de acciones tiene su propia semilla.
        env.action_space.seed(args.seed)
        print(f"Gymnasium: {gym.__version__}")
        print(f"Observaciones: {env.observation_space}")
        print(f"Acciones: {env.action_space} (0: izquierda, 1: derecha)")
        print(f"Límite de pasos por episodio: {env.spec.max_episode_steps}")
        print(f"Umbral registrado por Gymnasium: {env.spec.reward_threshold}")
        print("Criterio del proyecto: retorno medio >= 475 en 100 episodios.")
        print("Este ejercicio es una demostración, no una evaluación del proyecto.\n")

        for episode in range(args.episodes):
            # Sembrar el entorno una vez; los resets posteriores continúan su RNG.
            observation, info = env.reset(seed=args.seed if episode == 0 else None)
            if episode == 0:
                print("Estado inicial [posición, velocidad, ángulo, velocidad angular]:")
                print(observation)

            episode_return = 0.0
            steps = 0
            while True:
                # Elegir al azar: no hay tabla Q, red neuronal ni aprendizaje aún.
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)
                episode_return += reward
                steps += 1
                if episode == 0 and steps <= 3:
                    print(f"Paso {steps}: acción={action}, recompensa={reward}, "
                          f"nuevo estado={observation}")

                # Ambos casos finalizan el episodio; su significado es diferente.
                if terminated or truncated:
                    break

            print(f"Episodio {episode + 1}: pasos={steps}, retorno={episode_return:.0f}, "
                  f"terminated={terminated}, truncated={truncated}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
