# Tier 3 — LunarLander-v3

## Protocolo

Entorno LunarLander-v3, acciones discretas, parámetros {'continuous': False, 'enable_wind': False, 'gravity': -10.0}.
Presupuesto 500000 pasos, semillas [42, 43, 44], evaluación greedy cada 50000 pasos.
Controles de 10 episodios; inicio/final de 100 y confirmación de 100 cuando un control alcanza 200.
Se entregan modelos finales. No se elige el mejor checkpoint ni se descarta una semilla débil.
Las curvas usan los primeros 10 episodios al inicio/final; las tablas finales usan los 100.
La banda es una DE entre las medias de las semillas (ddof=0), no un IC del 95%.
DQN base se reutiliza en los dos estudios, la ablación y Tier 3. No son entrenamientos adicionales independientes.
Todos los agentes son propios; la referencia SB3 Zoo solo orientó parámetros. No se importó un agente entrenado.

## Estado y resultados completos

Grupos completos: 0/2. No se informan ejecuciones parciales como resultados finales.

| Grupo | Media final ± DE | Retornos por semilla | Semillas ≥200 | Entrenamiento medio |
|---|---:|---|---:|---:|

Tiempos excluyen evaluación, dependen de esta máquina y no son benchmarks universales.
Se ejecuta un entrenamiento a la vez en CPU con un hilo PyTorch.

## Hiperparámetros


## Diagnóstico y límites

LunarLander tiene recompensas distintas de CartPole. No se usa el límite Q≤100 del análisis anterior.
Los CSV registran pérdidas, extremos Q en 256 estados y una brecha de selección entre online y target.
Esa brecha no es el error Q frente al valor verdadero ni demuestra sobreestimación por sí sola.
Las semillas se usan durante el desarrollo; tres repeticiones son el mínimo, no una garantía universal.
La comparación causal de hiperparámetros depende de mantener el resto del protocolo; revisar config.json por modelo.
El estado completo exige ambos estudios (dos parámetros, tres valores cada uno), la ablación y Double DQN.

## Pasos hasta el umbral


## Referencias

- Gymnasium LunarLander: https://gymnasium.farama.org/environments/box2d/lunar_lander/
- Referencia de parámetros: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml
- Double DQN: van Hasselt, Guez y Silver, https://arxiv.org/abs/1509.06461
