# Tier 3 — LunarLander-v3

## Protocolo

Entorno LunarLander-v3, acciones discretas, parámetros {'continuous': False, 'enable_wind': False, 'gravity': -10.0}.
Presupuesto 500000 pasos, semillas [42, 43, 44], evaluación greedy cada 50000 pasos.
Base: red 8 → 128 → 128 → 4, Adam lr=0.0005, pérdida smooth_l1, gamma=0.99.
Replay 100000, lote 64, ajustes desde 5000 pasos: 1 ajuste cada 8 pasos; copia target cada 2000.
Exploración lineal de 1 a 0.05 en 150000 pasos, después constante. Evaluación sin exploración.
Controles de 10 episodios; inicio/final de 100 y confirmación de 100 cuando un control alcanza 200.
Se entregan modelos finales. No se elige el mejor checkpoint ni se descarta una semilla débil.
Las curvas usan los primeros 10 episodios al inicio/final; las tablas finales usan los 100.
La banda es una DE entre las medias de las semillas (ddof=0), no un IC del 95%.
DQN base se reutiliza en los dos estudios, la ablación y Tier 3. No son entrenamientos adicionales independientes.
Todos los agentes son propios; la referencia SB3 Zoo solo orientó parámetros. No se importó un agente entrenado.
Errata del protocolo congelado: el texto descriptivo selection menciona 200000 por arrastre de un piloto. El campo steps, los bucles y los CSV usan 500000; no se cambia el protocolo histórico.

## Estado y resultados completos

Grupos completos: 2/2. No se informan ejecuciones parciales como resultados finales.

| Grupo | Media final ± DE | Retornos por semilla | Semillas ≥200 | Entrenamiento medio | Total medio con evaluación |
|---|---:|---|---:|---:|---:|
| base | 194.77 ± 34.50 | 166.40, 174.58, 243.33 | 1/3 | 1027.4 s | 1209.0 s |
| double | 122.85 ± 91.26 | 227.71, 5.26, 135.60 | 1/3 | 910.4 s | 1091.6 s |

La columna entrenamiento excluye evaluación; total la incluye. El total va desde la evaluación inicial hasta guardar los modelos, excluyendo arranque de Python y preparación inicial.
Los tratamientos se ejecutan en secuencia; sus semillas pueden correr en procesos paralelos, un hilo PyTorch por proceso.
Consultar execution.json: la primera semilla base comenzó con ejecución secuencial. Los tiempos no se interpretan como comparaciones aisladas de hardware.
Durante el trabajo se realizan verificaciones y mediciones de coste; estos tiempos de reloj describen esta máquina y no son benchmarks aislados.

## Tier 3 — Double DQN

Double DQN menos DQN base: -71.91 puntos de media final.
Diferencias emparejadas por semilla [42, 43, 44]: +61.31, -169.33, -107.73; mejora en 1/3.
La red online elige argmax; target valora esa acción. En DQN estándar target selecciona y valora.
Se mantienen parámetros, inicializaciones por semilla, presupuesto y evaluación del baseline.
Double DQN no aumenta la media final observada. Separar selección y valoración no garantiza una política mejor con este presupuesto.
La propuesta busca reducir sesgo de sobreestimación; no garantiza mejorar cada entorno o cada semilla.

## Diagnóstico y límites

LunarLander tiene recompensas distintas de CartPole. No se usa el límite Q≤100 del análisis anterior.
Los CSV registran pérdidas, extremos Q en 256 estados y una brecha de selección entre online y target.
Esa brecha no es el error Q frente al valor verdadero ni demuestra sobreestimación por sí sola.
Las semillas se usan durante el desarrollo; tres repeticiones son el mínimo, no una garantía universal.
La comparación causal de hiperparámetros depende de mantener el resto del protocolo; revisar config.json por modelo.
Tier 3 requiere comparar base y Double DQN con tres semillas y parámetros iguales.
La brecha entre redes se registra después de las copias programadas y puede ser cero por esa coincidencia temporal.

## Pasos hasta el umbral

- base: 42: 400000, 43: not reached, 44: 450000
- double: 42: 500000, 43: not reached, 44: not reached

## Estabilidad y pérdidas medidas

diagnostics.json resume la pérdida del último bloque y los valores Q de diagnóstico por modelo.
La pérdida mide ajuste al objetivo de bootstrap, no calidad de aterrizaje. Una pérdida menor no implica mayor retorno.
El mejor control de 100 solo se usa para describir la trayectoria, nunca para elegir los pesos entregados. No todos los controles intermedios se ampliaron a 100.
1 ejecuciones confirmaron ≥200 en un control anterior y terminaron por debajo de 200.
- base, semilla 42: mejor confirmación observada 240.06 a 400000 pasos; final 166.40.
La regresión es compatible con el aprendizaje sobre objetivos cambiantes y trayectorias distintas de replay/exploración; los registros por sí solos no aíslan una causa única.

Valores por semilla, en orden [42, 43, 44]. Pérdida: mediana del último bloque de entrenamiento; Q: máximo final en el banco de 256 estados aleatorios de cada semilla.
Ese banco no estima el valor verdadero de la política ni permite demostrar sobreestimación.

| Grupo | Medianas de pérdida | Máximos Q en el banco |
|---|---|---|
| base | 0.774, 0.274, 0.546 | 112.12, 113.70, 118.32 |
| double | 0.716, 0.116, 0.460 | 103.69, 109.54, 115.98 |

## Equipo y reproducción

Python 3.12.0; paquetes {'gymnasium': '1.4.0', 'numpy': '2.5.4', 'torch': '2.15.0+cpu', 'Box2D': '2.3.10'}; Windows-11-10.0.26200-SP0.
CPU AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD, 8 procesadores lógicos; un hilo, sin GPU.
Parámetros: configs/tier23_lunarlander.json. Notebooks: Tier2_LunarLander.ipynb y Tier3_DoubleDQN.ipynb.

## Evaluación adicional con nuevos inicios

Las condiciones iniciales se fijaron en el protocolo antes de entrenar. No se utilizan para reajustar parámetros ni escoger una semilla.
Base y Double pertenecen a la comparación predefinida. También se evalúa el DQN de mayor media de desarrollo de Tier 2: la elección de esa variante es posterior a desarrollo y está registrada en new_protocol.json.
Son nuevas condiciones iniciales de la misma distribución, no nuevos entrenamientos.

| Grupo | Media nueva ± DE | Semillas ≥200 |
|---|---:|---:|
| base | 203.84 ± 32.62 | 1/3 |
| double | 124.13 ± 96.54 | 1/3 |

## Finalización de los episodios de desarrollo

Resting indica que el cuerpo quedó en reposo; body_contact indica contacto del cuerpo; time_limit es corte por tiempo.
Estos conteos describen comportamiento. El criterio del proyecto sigue siendo la recompensa media, no solo el contacto con el suelo.

La clasificación prioriza corte por tiempo, contacto del cuerpo, fuera del área y reposo; algunos eventos físicos pueden coincidir.

| Grupo | Cuerpo/contacto | Fuera del área | Reposo | Límite de tiempo | Otros |
|---|---:|---:|---:|---:|---:|
| base | 2 | 0 | 187 | 111 | 0 |
| double | 12 | 0 | 118 | 170 | 0 |

Para los grupos cuya media final queda por debajo de 200, los finales observados ayudan a concretar el fallo:
- base: reposo del cuerpo en 187/300 episodios, la finalización más frecuente. Debe leerse junto con el retorno: reposar no basta para obtener ≥200, y alcanzar el límite no confirma un aterrizaje.
- double: agotamiento del límite de tiempo en 170/300 episodios, la finalización más frecuente. Debe leerse junto con el retorno: reposar no basta para obtener ≥200, y alcanzar el límite no confirma un aterrizaje.
Son síntomas observados de las políticas finales; no identifican por sí solos una causa única del proceso de aprendizaje.

## Referencias

- Gymnasium LunarLander: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- Referencia de parámetros: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
- Double DQN: van Hasselt, Guez y Silver, https://arxiv.org/abs/1509.06461
