"""Memoria circular de experiencias para el futuro agente DQN."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray


class ReplayBuffer:
    """Guardar transiciones y muestrearlas uniformemente sin reemplazo.

    Al llenarse, cada experiencia nueva sustituye la más antigua. Las dos
    observaciones se copian para evitar cambios al reutilizar arrays del entorno.
    terminated y truncated se conservan separados para calcular objetivos luego.
    """

    def __init__(self, capacity, observation_shape=(4,), seed=42):
        if not isinstance(capacity, int) or isinstance(capacity, bool) or capacity < 1:
            raise ValueError("capacity debe ser un entero positivo")
        self.capacity = capacity
        self.observation_shape = tuple(observation_shape)
        self.observations = np.empty((capacity,) + self.observation_shape, dtype=np.float32)
        self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)
        self._position = 0
        self._size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self._size

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        observation = np.asarray(observation, dtype=np.float32)
        next_observation = np.asarray(next_observation, dtype=np.float32)
        if (observation.shape != self.observation_shape or
                next_observation.shape != self.observation_shape):
            raise ValueError("Las observaciones no coinciden con la forma de la memoria")
        index = self._position
        self.observations[index] = observation
        self.actions[index] = action
        self.rewards[index] = reward
        self.next_observations[index] = next_observation
        self.terminated[index] = terminated
        self.truncated[index] = truncated
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size):
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool) or
                not 1 <= batch_size <= self._size):
            raise ValueError("batch_size debe estar entre 1 y la cantidad guardada")
        indices = self.rng.choice(self._size, size=batch_size, replace=False)
        # El indexado con un array produce copias: el lote no modifica la memoria.
        return ReplayBatch(
            self.observations[indices], self.actions[indices], self.rewards[indices],
            self.next_observations[indices], self.terminated[indices], self.truncated[indices],
        )
