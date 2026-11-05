# Mejora de Tier 2 y Tier 3

## Protocolo y alcance

Presupuesto completo: 1000000 pasos por modelo; semillas [42, 43, 44]. Ambos algoritmos usan los mismos parámetros.
Parámetros: {'hidden_widths': [128, 128], 'learning_rate': 0.00063, 'loss': 'smooth_l1', 'gamma': 0.99, 'buffer_size': 100000, 'batch_size': 128, 'learning_starts': 5000, 'train_every': 4, 'gradient_steps': 1, 'target_sync_every': 1000, 'epsilon_decay_steps': 200000, 'epsilon_end': 0.05}. Calendario de tasa: {'decay_start': 300000, 'decay_end': 800000, 'final': 5e-05}.
Se guardan los pesos del final del entrenamiento y, por separado, el checkpoint con mayor media de validación confirmada en 100 episodios.
Validación: base 30000000; controles de 10 cada 50000 pasos. Se amplían a 100 cuando alcanzan 200 y siempre al final.
Prueba y confirmación: bases 40000000 y 50000000, 100 episodios por modelo en cada una. No intervienen en la selección.
Los modelos seleccionados se cargan del disco y reproducen exactamente sus 100 retornos de validación antes de probarlos.
Las curvas describen la red durante el entrenamiento; en el último control usan los primeros diez episodios. Las tablas de prueba corresponden al checkpoint seleccionado, que puede ser anterior al último paso.
El paso seleccionado no es un ahorro de entrenamiento: cada modelo consume todo el presupuesto antes de probarse.
Los estudios originales de 500000 pasos y sus resultados negativos se conservan. Esta ronda cambia conjuntamente el presupuesto y varios ajustes; no permite atribuir la mejora a uno solo.
No se cambian recompensas, límite de 1000 pasos del entorno ni semillas de entrenamiento.

Grupos completos: 2/2. Objetivo de ambas pruebas, tres semillas por método: True.

| Método | Semilla | Primer umbral validado | Paso seleccionado | Validación 100 | Prueba 100 | Confirmación 100 |
|---|---:|---:|---:|---:|---:|---:|
| base | 42 | 350000 | 500000 | 270.64 | 272.70 | 269.08 |
| base | 43 | 250000 | 850000 | 278.32 | 277.61 | 279.00 |
| base | 44 | 450000 | 850000 | 276.82 | 267.25 | 273.00 |
| double | 42 | 450000 | 950000 | 276.93 | 278.71 | 274.18 |
| double | 43 | 200000 | 400000 | 281.67 | 282.33 | 278.67 |
| double | 44 | 300000 | 800000 | 281.72 | 271.08 | 281.35 |

| Método | Media prueba ± DE | Media confirmación ± DE | Semillas ≥200 en ambas |
|---|---:|---:|---:|
| base | 272.52 ± 4.23 | 273.69 ± 4.08 | 3/3 |
| double | 277.37 ± 4.69 | 278.07 ± 2.96 | 3/3 |

Double menos DQN en prueba: +4.85; diferencias por semilla +6.00, +4.72, +3.83.
La diferencia describe estas tres repeticiones; no demuestra una superioridad universal de un método.
Alcanzar 200 en las tres semillas no garantiza éxito para cualquier inicialización futura. La DE es entre medias de semillas (ddof=0).

## Tiempo y reproducción

Tiempos por modelo en summary.json: entrenamiento excluye validación/prueba; total las incluye. Las semillas comparten CPU, un hilo PyTorch por proceso. No es un benchmark aislado.

| Método | Entrenamiento medio (s) | Total medio (s) |
|---|---:|---:|
| base | 1898.6 | 2185.2 |
| double | 2002.5 | 2309.9 |

Equipo: AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD; 8 procesadores lógicos, CPU, un hilo por proceso. Windows-11-10.0.26200-SP0.
Python 3.12.0; paquetes {'torch': '2.15.0+cpu', 'numpy': '2.5.4', 'gymnasium': '1.4.0', 'Box2D': '2.3.10'}.

Reproducir esta ronda: python -m experiments.mejorar_tier23 --output results/reproduccion_mejora
Regenerar figuras sin entrenar: python -m experiments.resumir_mejora --root results/tier23_mejora_v1
Los modelos incluyen versiones de paquetes, hardware y huellas de código en config.json.
Referencia de parámetros: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
