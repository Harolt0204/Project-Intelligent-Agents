"""Agrupar observaciones continuas de CartPole en casillas de una tabla."""

import numpy as np


class CartPoleDiscretizer:
    """Discretizar las cuatro variables, con resolución configurable por dimensión.

    spacing_power=1 conserva intervalos uniformes. Un valor mayor concentra
    fronteras cerca del equilibrio (cero), manteniendo el número de casillas.
    position_bins y velocity_bins permiten tablas más pequeñas sin eliminar variables.

    Los límites de posición y ángulo corresponden a la zona previa al fallo.
    Las velocidades no tienen límites finitos en el espacio declarado: usamos
    rangos iniciales de diseño que habrá que revisar con datos de entrenamiento.
    Los valores fuera del rango se asignan a las casillas de los extremos.
    """

    def __init__(self, bins=6, velocity_limit=3.0, angular_velocity_limit=3.5,
                 spacing_power=1.0, position_bins=None, velocity_bins=None):
        if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
            raise ValueError("bins debe ser un entero mayor o igual que 2")
        self.bins = bins
        counts = (bins if position_bins is None else position_bins,
                  bins if velocity_bins is None else velocity_bins, bins, bins)
        if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in counts):
            raise ValueError("Cada dimensión requiere al menos dos intervalos enteros")
        if not np.isfinite(spacing_power) or spacing_power <= 0:
            raise ValueError("spacing_power debe ser finito y positivo")
        self.spacing_power = spacing_power
        if not np.isfinite([velocity_limit, angular_velocity_limit]).all() or min(velocity_limit, angular_velocity_limit) <= 0:
            raise ValueError("Los límites de velocidad deben ser finitos y positivos")
        # Orden: posición (m), velocidad (m/s), ángulo (rad), vel. angular (rad/s).
        self.low = np.array([-2.4, -velocity_limit, -np.deg2rad(12), -angular_velocity_limit])
        self.high = -self.low
        # n intervalos requieren n-1 fronteras interiores.
        self.edges = tuple(
            np.linspace(low, high, count + 1)[1:-1]
            for count, low, high in zip(counts, self.low, self.high)
        )
        if spacing_power != 1.0:
            # Concentrar fronteras cerca de cero sin cambiar observaciones ni acciones.
            units = [np.linspace(-1.0, 1.0, count + 1)[1:-1] for count in counts]
            self.edges = tuple(high * np.sign(unit) * np.abs(unit) ** spacing_power
                               for high, unit in zip(self.high, units))
        self.shape = counts

    def encode(self, observation):
        """Devolver un índice por variable; frontera exacta va a la derecha."""
        values = np.asarray(observation, dtype=float)
        if values.shape != (4,) or not np.all(np.isfinite(values)):
            raise ValueError("Se requieren 4 observaciones finitas")
        return tuple(
            int(np.digitize(value, edges))
            for value, edges in zip(values, self.edges)
        )


def discretization_label(config):
    """Nombre de figura que distingue resolución uniforme y por dimensión."""
    counts = config.get("bins_per_dimension", [config["bins"]] * 4)
    detail = str(counts[0]) if len(set(counts)) == 1 else "/".join(map(str, counts))
    return f"Q-learning ({detail} intervalos)"
