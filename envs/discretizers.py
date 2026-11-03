"""Agrupar observaciones continuas de CartPole en casillas de una tabla."""

import numpy as np


class CartPoleDiscretizer:
    """Mismo número de intervalos uniformes para cada una de las 4 variables.

    Los límites de posición y ángulo corresponden a la zona previa al fallo.
    Las velocidades no tienen límites finitos en el espacio declarado: usamos
    rangos iniciales de diseño que habrá que revisar con datos de entrenamiento.
    Los valores fuera del rango se asignan a las casillas de los extremos.
    """

    def __init__(self, bins=6):
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("bins debe ser un entero mayor o igual que 2")
        self.bins = bins
        # Orden: posición (m), velocidad (m/s), ángulo (rad), vel. angular (rad/s).
        self.low = np.array([-2.4, -3.0, -np.deg2rad(12), -3.5])
        self.high = -self.low
        # n intervalos requieren n-1 fronteras interiores.
        self.edges = tuple(
            np.linspace(low, high, bins + 1)[1:-1]
            for low, high in zip(self.low, self.high)
        )
        self.shape = (bins,) * 4

    def encode(self, observation):
        """Devolver 4 índices entre 0 y bins-1; frontera exacta va a la derecha."""
        values = np.asarray(observation, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("Se requieren 4 observaciones finitas")
        return tuple(
            int(np.digitize(value, edges))
            for value, edges in zip(values, self.edges)
        )
