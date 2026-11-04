"""Ablación de Tier 2: replay permanece, se elimina la red objetivo independiente."""

from agents.dqn import DQNAgent


class DQNWithoutTargetAgent(DQNAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.target

    def bootstrap_values(self, next_observations):
        return self.online(next_observations).max(dim=1).values

    def sync_target(self):
        """No hay una red objetivo que actualizar en esta ablación."""
