"""Tier 3: separar elección y valoración de la siguiente acción."""

from agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    """La red online elige la acción; target estima su valor.

    El resto del agente y el entrenamiento es compartido con DQN estándar.
    learn() llama este cálculo dentro de torch.no_grad().
    """

    def bootstrap_values(self, next_observations):
        actions = self.online(next_observations).argmax(dim=1, keepdim=True)
        return self.target(next_observations).gather(1, actions).squeeze(1)
