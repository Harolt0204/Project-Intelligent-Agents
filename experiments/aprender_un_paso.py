"""Parte 3: ver un ajuste de Q-learning y un episodio real con aprendizaje."""

import gymnasium as gym

from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer


def main():
    discretizer = CartPoleDiscretizer(6)

    print("EJEMPLO INVENTADO: reproducir el cálculo explicado en el chat")
    agent = QLearningAgent(discretizer.shape, alpha=0.1, gamma=0.9)
    state = (3, 3, 3, 3)
    next_state = (3, 3, 4, 3)
    agent.q[state] = [5.0, 0.0]
    agent.q[next_state] = [10.0, 2.0]
    print(f"Antes: izquierda={agent.q[state][0]}, derecha={agent.q[state][1]}")
    value = agent.update(state, action=0, reward=1.0,
                         next_state=next_state, terminated=False)
    print("Objetivo = 1 + 0.9 * 10 = 10")
    print(f"Después: izquierda={value}, derecha={agent.q[state][1]}")

    print("\nEPISODIO REAL: otra tabla, empezando completamente en cero")
    # Esta tabla es independiente: no hereda los números inventados del ejemplo.
    agent = QLearningAgent(discretizer.shape, alpha=0.1, gamma=0.9, seed=42)
    env = gym.make("CartPole-v1")
    try:
        observation, _ = env.reset(seed=42)
        state = discretizer.encode(observation)
        steps = 0
        episode_return = 0.0
        while True:
            # Epsilon fijo solo para esta demostración. Al entrenar lo reduciremos.
            action = agent.choose_action(state, epsilon=0.5)
            observation, reward, terminated, truncated, _ = env.step(action)
            next_state = discretizer.encode(observation)
            old_value = float(agent.q[state][action])
            new_value = agent.update(state, action, reward, next_state, terminated)
            steps += 1
            episode_return += reward
            if steps <= 3 or terminated or truncated:
                print(f"Paso {steps}: casilla={state}, acción={action}, "
                      f"recompensa={reward}, Q antes={old_value:.4f}, "
                      f"Q después={new_value:.4f}")
            state = next_state
            if terminated or truncated:
                break
        print(f"Retorno del episodio: {episode_return:.0f}")
        print("La tabla se actualizó; un episodio no demuestra mejora de rendimiento.")
        print("Demostración didáctica: no es un resultado experimental del artículo.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
