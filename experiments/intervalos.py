"""IC t del 95% para las tres réplicas independientes de este proyecto.

La unidad experimental es una semilla de entrenamiento, no un episodio.
Son intervalos puntuales, bajo el modelo de medias de semillas normales e
independientes; con n=3 ese supuesto no puede comprobarse de forma convincente.
"""

import numpy as np

T_975_DF2 = 4.302652729696142


def mean_ci95(values, axis=0):
    """Media y extremos sin recorte. Un punto incompleto queda sin estimación."""
    values = np.asarray(values, dtype=float)
    if values.shape[axis] != 3:
        raise ValueError('Este protocolo requiere exactamente tres semillas independientes.')
    complete = np.isfinite(values).all(axis=axis)
    mean = np.mean(values, axis=axis)
    half = T_975_DF2 * np.std(values, axis=axis, ddof=1) / np.sqrt(3)
    return tuple(np.where(complete, v, np.nan) for v in (mean, mean-half, mean+half))


def ci_record(values):
    mean, low, high = mean_ci95(values)
    return {'n_training_seeds': 3, 'mean': float(mean), 'ci95_low': float(low),
            'ci95_high': float(high), 'ci_method': 'Student t, df=2, ddof=1',
            'unit': 'training seed mean', 'scope': 'pointwise, conditional on selected configuration'}
