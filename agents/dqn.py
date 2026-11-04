"""Red Q y agente DQN propio con red objetivo y actualización de un lote."""

from copy import deepcopy

import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    """De observaciones continuas a un valor Q por acción discreta.

    Acepta un estado (obs_dim,) o un lote (batch_size, obs_dim).
    Devuelve (n_actions,) o (batch_size, n_actions), respectivamente.
    La salida es lineal: los valores Q no son probabilidades.
    """

    def __init__(self, obs_dim=4, n_actions=2, hidden_width=64):
        super().__init__()
        # Un entero mantiene dos capas iguales; una pareja permite, por ejemplo, 120/84.
        widths = (hidden_width, hidden_width) if isinstance(hidden_width, int) else tuple(hidden_width)
        if len(widths) != 2 or min(obs_dim, n_actions, *widths) < 1:
            raise ValueError("Las dimensiones deben ser positivas")
        self.layers = nn.Sequential(
            nn.Linear(obs_dim, widths[0]),
            nn.ReLU(),
            nn.Linear(widths[0], widths[1]),
            nn.ReLU(),
            nn.Linear(widths[1], n_actions),
        )

    def forward(self, observations):
        return self.layers(observations)


class DQNAgent:
    """La red online aprende; target conserva una referencia entre copias.

    El bucle de entrenamiento decidirá cuándo llamar a sync_target(), medido
    en pasos del entorno. learn() no sincroniza la referencia automáticamente.
    """

    def __init__(self, obs_dim=4, n_actions=2, hidden_width=64,
                 learning_rate=1e-3, gamma=0.99, seed=42, loss="smooth_l1"):
        if learning_rate <= 0 or not 0 <= gamma <= 1:
            raise ValueError("learning_rate debe ser positiva y gamma estar en [0, 1]")
        if loss not in ("smooth_l1", "mse"):
            raise ValueError("loss debe ser smooth_l1 o mse")
        torch.manual_seed(seed)
        self.online = QNetwork(obs_dim, n_actions, hidden_width)
        self.target = deepcopy(self.online)
        self.target.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.n_actions = n_actions
        self.loss_name = loss
        self.rng = np.random.default_rng(seed)

    def choose_action(self, observation, epsilon):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon debe estar en [0, 1]")
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.n_actions))
        with torch.no_grad():
            values = self.online(torch.as_tensor(observation, dtype=torch.float32)).numpy()
        best_actions = np.flatnonzero(values == values.max())
        return int(self.rng.choice(best_actions))

    def learn(self, batch):
        observations = torch.as_tensor(batch.observations, dtype=torch.float32)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32)
        next_observations = torch.as_tensor(batch.next_observations, dtype=torch.float32)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool)

        # De las dos salidas solo comparamos la correspondiente a la acción tomada.
        predictions = self.online(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            next_values = self.target(next_observations).max(dim=1).values
            # Un corte por tiempo no elimina el valor futuro: solo terminated.
            targets = rewards + self.gamma * (~terminated).float() * next_values

        if self.loss_name == "mse":
            loss = nn.functional.mse_loss(predictions, targets)
        else:
            loss = nn.functional.smooth_l1_loss(predictions, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return {"loss": float(loss.detach()),
                "predictions_before": predictions.detach().tolist(),
                "targets": targets.tolist()}

    def sync_target(self):
        """Actualizar la referencia mediante una copia completa de la red online."""
        self.target.load_state_dict(self.online.state_dict())
