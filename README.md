# Project 01 - Aprendizaje por refuerzo

Implementaciones propias de **Q-learning, DQN y Double DQN** en Python/PyTorch,
para CartPole-v1 y LunarLander-v3 de Gymnasium. Proyecto de Agentes Inteligentes.
[Enunciado y requisitos](Project_01_Reinforcement_Learning_DQN.md).

El repositorio incluye código, configuraciones, registros por semilla, modelos,
figuras, GIF y análisis teórico. El artículo final y la defensa oral son
entregables separados que todavía están en preparación.

## Estructura

| Ruta | Contenido |
|---|---|
| `agents/` | Q-learning tabular, DQN, Double DQN y replay buffer |
| `envs/` | Discretización de CartPole |
| `configs/` | Parámetros y protocolos de los experimentos |
| `experiments/` | Entrenamiento, evaluación, análisis y reproducción |
| `tests/` | Pruebas de componentes y protocolos |
| `results/` | Registros, modelos, métricas, figuras y GIF |
| `report/` | Análisis teórico, discusión y registro de uso de IA |
| `output/pdf/` | Análisis teórico exportado a PDF |

## Instalación

Entorno verificado: **Windows, Python 3.12 de 64 bits, CPU**, Gymnasium 1.4.0,
Box2D 2.3.10 y PyTorch 2.15.0+cpu. No se requiere GPU para los experimentos
incluidos. Cada computadora debe crear su propio entorno virtual.

```powershell
git clone https://github.com/Harolt0204/Project-Intelligent-Agents.git
cd Project-Intelligent-Agents
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-notebook.txt
```

`requirements-notebook.txt` incluye `requirements.txt` y añade soporte Jupyter.
Para utilizar solamente los módulos de entrenamiento y análisis, puede
instalarse `requirements.txt`. Los comandos siguientes se ejecutan desde la
raíz del repositorio; no es necesario activar el entorno virtual.
La instalación en otros sistemas no se validó; en Linux la ruta equivalente
es `.venv/bin/python`, y en macOS debe adaptarse la dependencia de PyTorch.

## Notebooks

| Cuaderno | Contenido |
|---|---|
| [Tier1_Explicado.ipynb](Tier1_Explicado.ipynb) | CartPole: Q-learning, discretizaciones y comparación con DQN |
| [Tier2_LunarLander.ipynb](Tier2_LunarLander.ipynb) | DQN, estudios de hiperparámetros y ablación |
| [Tier3_DoubleDQN.ipynb](Tier3_DoubleDQN.ipynb) | Double DQN y comparación emparejada con DQN |
| [Analisis_Teorico_e_Intervalos.ipynb](Analisis_Teorico_e_Intervalos.ipynb) | Análisis obligatorio y figuras actualizadas con IC del 95% |

En VS Code, utilizar las extensiones Python y Jupyter y seleccionar el kernel
`.venv\Scripts\python.exe`. Las implementaciones se importan de los módulos del
repositorio. Los entrenamientos formales están desactivados inicialmente.
Los resultados incluidos pueden consultarse sin volver a entrenar.

El cuaderno de análisis reúne las figuras con intervalos actuales. Las salidas
anteriores guardadas en los cuadernos de cada Tier pueden conservar las bandas
originales de desviación estándar, también admitidas por la guía.

## Resultados y protocolo

Cada configuración usa las semillas de entrenamiento **42, 43 y 44**. Las
puntuaciones finales son medias de **100 episodios greedy por modelo**, separadas
del entrenamiento. La DE siguiente describe la variación entre las tres medias
de semillas (ddof=0); no es un intervalo de confianza.

| Entorno y método | Retorno medio ± DE | Objetivo | Semillas que lo alcanzan |
|---|---:|---:|---:|
| CartPole: Q-learning 2/2/6/6 | 495.73 ± 6.04 | ≥475 | 3/3 |
| CartPole: DQN | 498.60 ± 1.98 | ≥475 | 3/3 |
| LunarLander: DQN | 272.52 ± 4.23 | ≥200 | 3/3 |
| LunarLander: Double DQN | 277.37 ± 4.69 | Comparación con DQN | 3/3 también superan 200 |

Las evaluaciones adicionales con otros 100 episodios por modelo confirmaron el
cumplimiento en las tres semillas de cada método. No son nuevas semillas de
entrenamiento. El éxito observado no garantiza éxito en cualquier inicialización.

- **Tier 1:** 200000 pasos por modelo, tanto tabular como DQN. Se comparan tres
  granularidades angulares: 2/2/6/6, 2/2/10/10 y 2/2/20/20. Se entregan modelos
  finales. [Resultados y límites](report/qlearning_objetivo.md).
- **Tier 2 inicial:** siete grupos únicos de tres semillas y 500000 pasos por
  modelo. Tasa de aprendizaje: 0.0001, 0.0005 y 0.001; frecuencia de target: 500,
  2000 y 8000. Se cambia un parámetro a la vez; la base se comparte entre estudios.
  La ablación elimina la red objetivo. [Estudios y discusión](results/tier23_lunarlander/report_tier2.md).
- **Tier 2 y Tier 3 mejorados:** DQN y Double DQN entrenan desde cero 1000000 de
  pasos por modelo, con los mismos parámetros y presupuesto. Se elige el mejor
  checkpoint por validación confirmada de 100 episodios; sus pesos se fijan antes
  de dos muestras de prueba separadas. Los pesos de la última actualización
  también se conservan. [Protocolo](report/plan_mejora_lunar.md) y
  [resultados](results/tier23_mejora_v1/report.md).

Los barridos, la ablación y los resultados negativos se conservan como evidencia.
La mejora conjunta cambia varios factores y no permite atribuir su efecto a uno
solo. Double DQN obtuvo una media algo mayor; la muestra de confirmación no
sustenta una afirmación de superioridad general.

Las curvas actuales muestran **IC t del 95% puntuales** entre tres medias de
semillas, con dos grados de libertad. No son bandas simultáneas. La normalidad
aproximada es un supuesto difícil de comprobar con tres réplicas.
[Método estadístico y limitaciones](report/intervalos_confianza.md).

## Reproducir las figuras desde los registros

Estos comandos no entrenan agentes. Cada uno reconstruye las figuras indicadas
usando los archivos incluidos en `results/`.

```powershell
# Tier 1: tres granularidades y DQN, entrenamiento y evaluación
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --dqn-root results/parte_16_estabilidad/dqn_referencia --tabular-root results/parte_18_qlearning/resolucion_suave --dqn-label "DQN ajustado" --output results/parte_18_qlearning/comparacion

# Estudios iniciales: learning_rate, target_sync_every, ablation y double_dqn
.\.venv\Scripts\python.exe -m experiments.analizar_tier23

# Mejora: learning, tests, dqn_learning y dqn_tests
.\.venv\Scripts\python.exe -m experiments.resumir_mejora

# IC de medias finales y diferencias emparejadas
.\.venv\Scripts\python.exe -m experiments.resumir_intervalos
```

Las salidas se guardan en las carpetas `comparacion/` y `figures/` de sus
experimentos, y en `results/analisis_intervalos/`. Se generan PNG y PDF.
[Etapas exploratorias y comandos históricos](report/historial_experimentos.md).

## Reproducir entrenamientos

Usar carpetas nuevas para conservar los registros entregados. Estos comandos
sí realizan entrenamientos completos y pueden tardar varias horas.

```powershell
# Tier 1: tres granularidades tabulares y DQN, tres semillas por configuración
.\.venv\Scripts\python.exe -m experiments.cuaderno_tier1 --output results/reproduccion_tier1

# Comparación de esa nueva ejecución
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --tabular-root results/reproduccion_tier1 --dqn-root results/reproduccion_tier1/dqn --dqn-label "DQN ajustado"

# Tier 2/3 iniciales: todos los estudios y la ablación, 500000 pasos por modelo
.\.venv\Scripts\python.exe -m experiments.paralelo_tier23 --output results/reproduccion_estudios

# Tier 2/3 mejorados: dos métodos, tres semillas, 1000000 pasos por modelo
.\.venv\Scripts\python.exe -m experiments.mejorar_tier23 --output results/reproduccion_mejora
```

Para repetir un estudio inicial por separado, añadir `--groups` al comando
`experiments.paralelo_tier23`:

| Estudio | Grupos |
|---|---|
| Tasa de aprendizaje | `base learning_rate_0p0001 learning_rate_0p001` |
| Frecuencia de target | `base target_sync_every_500 target_sync_every_8000` |
| Ablación | `base no_target` |
| Double DQN | `base double` |

Configuraciones: [Tier 1](configs/tier1_mejorado.json),
[Tier 2/3 iniciales](configs/tier23_lunarlander.json) y
[Tier 2/3 mejorados](configs/tier23_mejora_v1.json).
Cada ejecución registra sus versiones, semillas, tiempos y hardware. Las semillas
pueden compartir CPU, por lo que los tiempos no constituyen benchmarks aislados.
No se garantiza identidad numérica entre plataformas diferentes.

## Modelos y evidencia visual

Las rutas vigentes se indican en `results/tier1_actual.json` y
`results/tier23_actual.json`. Q-learning guarda tablas `.npy`; las redes se
incluyen como `.pt`. En LunarLander, `online_selected.pt` contiene el modelo elegido
por validación y `online_final.pt` los pesos del último paso.

| Agente | GIF incluido |
|---|---|
| Q-learning, CartPole | [CartPole tabular](results/parte_18_qlearning/resolucion_suave/bins_6/seed_42/visual/comparacion.gif) |
| DQN, CartPole | [CartPole DQN](results/parte_16_estabilidad/dqn_referencia/seed_42/visual/dqn_final.gif) |
| DQN, LunarLander | [LunarLander DQN](results/tier23_mejora_v1/base/seed_42/visual/lander.gif) |
| Double DQN, LunarLander | [LunarLander Double DQN](results/tier23_mejora_v1/double/seed_42/visual/lander.gif) |

Ejecutar un modelo guardado en una ventana, sin aprender:

```powershell
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent tabular --episodes 1
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent dqn --episodes 1
.\.venv\Scripts\python.exe -m experiments.mostrar_lunarlander --group base --seed 42
.\.venv\Scripts\python.exe -m experiments.mostrar_lunarlander --group double --seed 42
```

Una demostración individual no sustituye la evaluación de 100 episodios.

## Verificación

```powershell
# Dependencias y reproducción de un episodio de cada método de Tier 1
.\.venv\Scripts\python.exe -m experiments.verificar_instalacion

# Pruebas de componentes y protocolos
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

# Fuentes, modelos y datos de la mejora de LunarLander; requiere historial Git
.\.venv\Scripts\python.exe -m experiments.verificar_mejora

# Intervalos y ejecución del cuaderno de análisis
.\.venv\Scripts\python.exe -m experiments.verificar_analisis
```

Validación registrada: 31 pruebas unitarias; auditoría de los seis modelos
mejorados; reproducción de retornos guardados y comprobación de notebooks y GIF.
Evidencia: [auditoría de modelos](results/tier23_mejora_v1/audit.json) y
[auditoría de análisis](results/analisis_intervalos/audit.json).

## Análisis teórico y transparencia

El componente sin implementación justifica cuantitativamente las limitaciones
de RL tabular/DQN para BipedalWalker, Hopper, HalfCheetah, Ant y píxeles de Atari,
y la pertinencia de PPO, SAC y TD3. Incluye referencias primarias verificadas.
[Texto del análisis](report/analisis_teorico.md) y
[PDF de dos páginas](output/pdf/analisis_teorico.pdf).

El PDF se incluye ya generado. Para exportarlo de nuevo se necesita la dependencia
opcional `reportlab` y ejecutar `python -m experiments.exportar_analisis_pdf`.
El código de los agentes es propio; las referencias externas no se entregan como
agentes prefabricados. El apoyo de IA y sus verificaciones se documentan en
[la declaración y registro de uso de IA](report/uso_ia.md).
