"""Tabla Q, elección epsilon-greedy y actualización de Q-learning."""

import numpy as np


class QLearningAgent:
    def __init__(self, state_shape, n_actions=2, alpha=0.1, gamma=0.99, seed=42):
        if not 0 < alpha <= 1:
            raise ValueError("alpha debe estar en (0, 1]")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma debe estar en [0, 1]")
        self.alpha = alpha  # Qué fracción del error incorporamos en cada ajuste.
        self.gamma = gamma  # Cuánto valoramos las recompensas futuras.
        self.n_actions = n_actions
        self.q = np.zeros(tuple(state_shape) + (n_actions,), dtype=np.float64)
        self.rng = np.random.default_rng(seed)

    def choose_action(self, state, epsilon):
        """Explorar con probabilidad epsilon; en otro caso usar el mayor Q."""
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon debe estar en [0, 1]")
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))

        values = self.q[state]
        # Empates al azar para no favorecer siempre la acción 0 al empezar.
        best_actions = np.flatnonzero(values == np.max(values))
        return int(self.rng.choice(best_actions))

    def update(self, state, action, reward, next_state, terminated):
        """Actualizar solo Q(state, action) y devolver el nuevo valor.

        Si termina la tarea, no hay recompensa futura. Un corte por tiempo
        (truncated) reinicia el episodio, pero permite estimar valor futuro.
        El llamador debe pasar `terminated`, no `terminated or truncated`.
        """
        old_value = self.q[state + (action,)]
        future_value = 0.0 if terminated else float(np.max(self.q[next_state]))
        target = reward + self.gamma * future_value
        new_value = old_value + self.alpha * (target - old_value)
        self.q[state + (action,)] = new_value
        return float(new_value)
