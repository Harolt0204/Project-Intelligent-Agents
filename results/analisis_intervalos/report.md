# Intervalos calculados desde resultados guardados

| Tier | Método | Muestra | Media | IC 95% |
|---|---|---|---:|---:|
| 1 | Q-learning (2/2/6/6 intervalos) | final | 495.73 | [477.36, 514.10] |
| 1 | Q-learning (2/2/10/10 intervalos) | final | 163.84 | [32.43, 295.25] |
| 1 | Q-learning (2/2/20/20 intervalos) | final | 170.21 | [94.55, 245.88] |
| 1 | DQN ajustado | final | 498.60 | [492.56, 504.63] |
| 2 | DQN LunarLander | test | 272.52 | [259.64, 285.39] |
| 2 | DQN LunarLander | confirmation | 273.69 | [261.28, 286.10] |
| 3 | Double DQN LunarLander | test | 277.37 | [263.11, 291.63] |
| 3 | Double DQN LunarLander | confirmation | 278.07 | [269.06, 287.08] |

## Diferencias emparejadas: Double menos DQN

Cada diferencia usa la misma semilla y muestra de evaluación. No se combinan las dos muestras.

| Muestra | Diferencia media | IC 95% |
|---|---:|---:|
| test_mean | +4.85 | [2.14, 7.56] |
| confirmation_mean | +4.38 | [-6.51, 15.27] |

Con tres semillas estos IC dependen fuertemente del supuesto de normalidad. La comparación es exploratoria y posterior al desarrollo; no constituye una prueba confirmatoria universal.
En la confirmación el IC de la diferencia incluye cero. Los seis modelos superan individualmente 200 en ambas muestras; ese hecho descriptivo no depende de que el IC entre algoritmos excluya cero.
Método y límites: report/intervalos_confianza.md. No se ejecutó ningún entrenamiento para producir estas figuras.
