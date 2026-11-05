"""Generar el cuaderno educativo a partir del código compartido del proyecto."""

import json
from pathlib import Path


def main():
    import nbformat as nbf
    root = Path(__file__).resolve().parents[1]
    cells = []
    def md(text):
        cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text):
        cells.append(nbf.v4.new_code_cell(text.strip()))

    md("""
# Tier 1 explicado: aprender a equilibrar CartPole

**Integrantes:** Harolt y compañero · Agentes Inteligentes · Project 01

Este cuaderno es la entrada principal para **entender el código, cambiar parámetros,
consultar resultados y ver el carrito**. Los algoritmos son nuestros y viven en
`agents/`, `envs/` y `experiments/`. Aquí se muestra su código real; no hay una segunda
implementación que pueda quedar desactualizada.

**En VS Code:** abrir la carpeta completa del proyecto → abrir este archivo →
**Seleccionar kernel → Entornos de Python → `.venv\\Scripts\\python.exe`**.
Si falta soporte, instalar las extensiones Python y Jupyter de Microsoft y ejecutar
en la terminal `.\\.venv\\Scripts\\python.exe -m pip install -r requirements-notebook.txt`.

Ejecutar las celdas **en orden** con Shift+Enter. “Ejecutar todo” carga ejemplos y
resultados guardados: el entrenamiento largo y la animación se activan expresamente.

## Recorrido
1. Preparación y problema · 2. Parámetros · 3. Q-learning · 4. DQN y memoria
5. Entrenamiento y evaluación · 6. Resultados · 7. Carrito · 8. Reproducción y defensa.
""")
    code("""
from pathlib import Path
import sys
import json
import copy
import csv
import numpy as np
import torch
import gymnasium as gym
from IPython.display import display, Markdown, Image

# Funciona abriendo el notebook desde la raíz o desde una subcarpeta del proyecto.
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'agents/dqn.py').is_file()), None)
if ROOT is None:
    raise RuntimeError('Abre en VS Code la carpeta completa del proyecto.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.cuaderno_tier1 import (
    parametros, actual, codigo, resumen, entrenar, demostrar, evaluar_guardado,
)
from agents.tabular import QLearningAgent
from envs.discretizers import CartPoleDiscretizer
from agents.dqn import QNetwork, DQNAgent
from agents.replay import ReplayBuffer
from experiments.entrenar_tabular import train as entrenar_tabular, evaluate as evaluar_tabular
from experiments.entrenar_dqn import train as entrenar_dqn, evaluate_network

torch.set_num_threads(1)
torch.manual_seed(123)  # Solo ejemplos didácticos; cada entrenamiento fija su propia semilla.
print('Python:', sys.executable)
print('Gymnasium:', gym.__version__, '| PyTorch:', torch.__version__)
print('Proyecto:', ROOT)
""")
    md("""
## 1. ¿Qué aprende el agente?

El carrito puede empujar a la **izquierda (0)** o **derecha (1)**. La observación tiene
cuatro números: **posición, velocidad, ángulo de la vara y velocidad angular**.
Recibe **+1 por paso**; el episodio termina si cae la vara o el carrito sale del rango.
A los 500 pasos se corta el episodio por tiempo.

El **objetivo del enunciado** es retorno medio ≥475 en 100 episodios. No equivale a
ver un único episodio de 500. Comprobamos también la especificación instalada:
el umbral declarado por la librería puede diferir del exigido por el docente.
""")
    code("""
env = gym.make('CartPole-v1')
try:
    observacion, info = env.reset(seed=123)
    print('Observación inicial:', observacion)
    print('Acciones:', env.action_space, '| Observaciones:', env.observation_space)
    print('Límite:', env.spec.max_episode_steps, '| Umbral declarado:', env.spec.reward_threshold)
    siguiente, recompensa, terminated, truncated, info = env.step(0)
    print('Después de mover a la izquierda:', siguiente)
    print('Recompensa:', recompensa, '| terminated:', terminated, '| truncated:', truncated)
finally:
    env.close()
""")
    md("""
## 2. Parámetros que controlamos

La siguiente celda carga un diccionario **editable** de `configs/tier1_mejorado.json`.
Cambiarlo sí modifica los entrenamientos que lancemos más abajo; **no modifica modelos
ya guardados**. Sus parámetros originales se consultan en cada `config.json`.

| Parámetro | Significado |
|---|---|
| `steps` | Interacciones de entrenamiento por agente: mismo presupuesto para ambos métodos. |
| `seeds` | Repeticiones independientes; mínimo 3, 5 recomendadas para Tier 1. |
| `bins` | Tres granularidades: 6, 10 y 20. Se aplican al ángulo y su velocidad; por defecto también a las otras variables. |
| `position_bins` / `velocity_bins` | Opcional: menos intervalos para posición y velocidad del carrito, manteniendo las cuatro observaciones. |
| `eval_every` | Pasos entre evaluaciones de 10 episodios sin exploración. |
| `alpha` / `learning_rate` | Tamaño de los ajustes de la tabla / red neuronal. |
| `initial_q` | Estimación inicial en todas las casillas de la tabla; 0 o un valor optimista. |
| `alpha_end` | Tamaño final de los ajustes tabulares; si coincide con alpha, permanece constante. |
| `spacing_power` | 1 mantiene intervalos uniformes; >1 concentra detalle cerca del equilibrio. |
| `gamma` | Peso de recompensas futuras. |
| `epsilon_decay_steps` / `epsilon_end` | Duración del descenso de exploración / exploración final. |
| `velocity_limit` / `angular_velocity_limit` | Rangos de velocidad cubiertos por los intervalos tabulares. |
| `hidden_widths` | Número de neuronas de cada capa oculta. |
| `buffer_size` / `batch_size` | Experiencias conservadas / experiencias por ajuste. |
| `learning_starts` / `train_every` | Cuándo empezar a ajustar / pasos entre ajustes. |
| `target_sync_every` | Pasos entre copias de la red principal a la red objetivo. |
| `loss` | Cómo medimos el error: `mse` o `smooth_l1`. |

Las opciones de DQN y tabular pueden diferir: la comparación exige igual presupuesto,
entorno y protocolo de evaluación, no algoritmos con idénticos parámetros internos.
""")
    code("""
CONFIG = parametros()
# Ejemplos de edición (descomentar para un experimento nuevo):
# CONFIG['dqn']['learning_rate'] = 0.0001
# CONFIG['tabular']['alpha'] = 0.05
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
""")
    md("""
## 3. Q-learning: aprender con una tabla

### 3.1. Convertir observaciones en casillas
Las observaciones son continuas; una tabla necesita índices enteros. `encode` asigna
un intervalo a cada variable. Si todas usan 6 intervalos, hay 6⁴=1296 estados;
con 20 para todas, 160000. Podemos dedicar solo 2 intervalos a posición y velocidad:
con 20 en cada variable angular, quedan 2×2×20×20=1600 estados. Mantenemos las cuatro
observaciones, pero cambiamos cuánto detalle se conserva en cada una.
Los valores fuera de rango van a una casilla extrema; no se eliminan.

Los rangos de posición y ángulo conservan los límites físicos de fallo. Afinamos los
rangos de velocidad para aprovechar la resolución cerca de las situaciones habituales.
También podemos concentrar las fronteras cerca de cero con `spacing_power`: la tabla
sigue teniendo el mismo número de casillas, pero distingue mejor diferencias pequeñas
cerca del equilibrio. Los intervalos exteriores se hacen más amplios.
""")
    code("""
codigo(CartPoleDiscretizer)
discretizador = CartPoleDiscretizer(
    bins=CONFIG['bins'][0],
    velocity_limit=CONFIG['tabular']['velocity_limit'],
    angular_velocity_limit=CONFIG['tabular']['angular_velocity_limit'],
    spacing_power=CONFIG['tabular'].get('spacing_power', 1.0),
    position_bins=CONFIG['tabular'].get('position_bins'),
    velocity_bins=CONFIG['tabular'].get('velocity_bins'),
)
estado_ejemplo = np.array([0.02, -0.1, 0.03, 0.2])
print('Observación:', estado_ejemplo, '→ casilla:', discretizador.encode(estado_ejemplo))
print('Límites inferiores:', discretizador.low)
print('Fronteras del ángulo en grados:', np.rad2deg(discretizador.edges[2]))
for bins in CONFIG['bins']:
    shape = (CONFIG['tabular'].get('position_bins') or bins, CONFIG['tabular'].get('velocity_bins') or bins, bins, bins)
    count = int(np.prod(shape))
    print(f'Intervalos {shape} → {count:,} estados → {2*count:,} valores Q')
""")
    md(r"""
### 3.2. Elegir y aprender

`choose_action` explora con probabilidad ε; en otro caso escoge el mayor Q.
Q estima la **recompensa futura descontada**, no una probabilidad.
Con recompensa +1 y gamma=0.99, el retorno descontado verdadero no supera 100.
Eso es distinto de los hasta 500 puntos sin descuento que medimos por episodio.

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\,[r + \gamma\max_{a'}Q(s',a') - Q(s,a)]
\]

Si hay fallo real (`terminated=True`), el término futuro es cero. Si solo se llega
al límite de tiempo (`truncated=True`), conservamos la estimación futura.

**Inicialización optimista:** empezar con Q=100 cuando gamma=0.99 supone que una
acción aún desconocida podría producir el máximo retorno descontado. Al fallar,
Q-learning reduce su estimación con la recompensa real. No cambiamos la recompensa
del entorno ni ponemos acciones correctas en la tabla. El efecto se comprueba con
entrenamientos completos, no se da por hecho que siempre mejore.

**Alpha decreciente:** el bucle de entrenamiento puede reducir alpha linealmente
hasta `alpha_end`. Al principio incorpora cambios mayores y al final hace ajustes
más pequeños. La fórmula Q-learning y la recompensa se conservan.
""")
    code("""
codigo(QLearningAgent)
# Ejemplo verificable: Q viejo=5, recompensa=1, mejor Q siguiente=10.
ejemplo = QLearningAgent((2,), alpha=0.1, gamma=0.9, seed=123)
ejemplo.q[:] = [[5, 0], [10, 2]]
print('Antes:', ejemplo.q.copy())
ejemplo.update((0,), 0, reward=1, next_state=(1,), terminated=False)
print('Después:', ejemplo.q, '| Cálculo: 5 + 0.1 * (1 + 0.9*10 - 5) = 5.5')
""")
    md("""
## 4. DQN: aprender con una red neuronal

### 4.1. La red Q
Recibe los cuatro números y devuelve **dos estimaciones Q**, una por acción. Las
capas ocultas usan ReLU; la salida es lineal. No usamos softmax: los Q no son probabilidades.
La red aprende a aproximar valores para estados continuos, sin agruparlos en intervalos.
""")
    code("""
codigo(QNetwork)
red_ejemplo = QNetwork(hidden_width=CONFIG['dqn']['hidden_widths'])
with torch.inference_mode():
    valores = red_ejemplo(torch.tensor(estado_ejemplo, dtype=torch.float32))
print(red_ejemplo)
print('Q izquierda / derecha, sin entrenar:', valores.numpy())
print('Parámetros ajustables de la red:', sum(p.numel() for p in red_ejemplo.parameters()))
""")
    md("""
### 4.2. Memoria de experiencias

Guardamos `(estado, acción, recompensa, siguiente estado, terminated, truncated)`.
En lugar de aprender únicamente de la experiencia más reciente, muestreamos un lote
al azar. La memoria es circular: cuando se llena, sustituye las experiencias más antiguas.
""")
    code("""
codigo(ReplayBuffer)
memoria_ejemplo = ReplayBuffer(capacity=10, seed=123)
memoria_ejemplo.add(estado_ejemplo, 0, 1, estado_ejemplo + 0.01, False, False)
lote = memoria_ejemplo.sample(1)
print('Experiencias guardadas:', len(memoria_ejemplo))
print('Estado:', lote.observations, '| Acción:', lote.actions, '| Recompensa:', lote.rewards)
""")
    md("""
### 4.3. Actualización de DQN y red objetivo

La red **online** aprende; la **target** permanece fija entre copias. Para cada experiencia:

1. Consultamos el Q de la acción que realmente se tomó.
2. Construimos el objetivo: recompensa + gamma × mejor Q siguiente de la red target.
3. Eliminamos el término futuro cuando `terminated=True`.
4. Medimos el error, calculamos gradientes y ajustamos solo online con Adam.
5. Copiamos online a target cada cierto número de pasos.

Seguimos usando **DQN estándar**: no es Double DQN. La extensión de Tier 3 sigue pendiente.
""")
    code("""
codigo(DQNAgent)
""")
    md("""
## 5. ¿Cómo entrenamos y cómo evaluamos?

El entrenamiento interactúa con el entorno y actualiza tabla o red. La evaluación
usa otro entorno, ε=0 y no aprende. Una copia o un generador independiente impide
que evaluar cambie los números aleatorios del entrenamiento.

- Presupuesto común: 200000 pasos por ejecución formal.
- Evaluación periódica: 10 episodios cada 10000 pasos.
- Evaluación inicial/final: 100 episodios; confirmación de 100 si un control obtiene ≥475.
- Tres semillas: informamos **media y desviación estándar entre las medias de las semillas**.
- Guardamos el modelo al final del presupuesto, aunque haya rendido mejor antes.

Cada celda siguiente muestra la función real que se ejecuta. Los registros de
entrenamiento y evaluación se guardan por separado.
""")
    code("""
codigo(evaluate_network)
codigo(evaluar_tabular)
""")
    md("### 5.1. Bucle tabular: seguir observación → acción → recompensa → actualización")
    code("codigo(entrenar_tabular)")
    md("### 5.2. Bucle DQN: añadir memoria, muestreo y sincronización de target")
    code("codigo(entrenar_dqn)")
    md("""
### 5.3. Prueba corta para aprender a ejecutar

Activar `EJECUTAR_PRUEBA` para entrenar **un DQN de 10000 pasos y una semilla**.
Sirve para comprobar el recorrido, **no es un resultado formal** ni asegura llegar a 475.
Se crea una carpeta nueva con fecha y hora; nunca se reemplazan los resultados entregados.
""")
    code("""
EJECUTAR_PRUEBA = False
if EJECUTAR_PRUEBA:
    from datetime import datetime
    prueba = copy.deepcopy(CONFIG)
    prueba['steps'] = 10_000
    prueba['seeds'] = [42]
    prueba['dqn']['learning_starts'] = 1000
    destino_prueba = ROOT / 'results/notebook' / datetime.now().strftime('prueba_%Y%m%d_%H%M%S_%f')
    entrenar(prueba, destino_prueba, metodo='dqn')
else:
    print('Prueba corta desactivada. Cambia EJECUTAR_PRUEBA a True para ejecutarla.')
""")
    md("""
### 5.4. Repetir el experimento formal completo

Activa esta celda solo si quieres volver a entrenar **las tres granularidades y DQN,
cada uno con tres semillas**. Puede tardar varios minutos o más según el equipo.
Conservar los pasos, las tres semillas y la evaluación común para una comparación válida.
Editar CONFIG permite investigar una configuración nueva, no reescribir el historial.
""")
    code("""
EJECUTAR_EXPERIMENTO_COMPLETO = False
if EJECUTAR_EXPERIMENTO_COMPLETO:
    from datetime import datetime
    if len(set(CONFIG['seeds'])) < 3 or len(set(CONFIG['bins'])) < 3:
        raise ValueError('El experimento formal requiere al menos tres semillas y tres granularidades.')
    destino_completo = ROOT / 'results/notebook' / datetime.now().strftime('formal_%Y%m%d_%H%M%S_%f')
    nuevos_resultados = entrenar(CONFIG, destino_completo, metodo='todos')
    print('Experimento nuevo guardado en:', destino_completo)
else:
    print('Entrenamiento formal desactivado. Los resultados existentes se pueden consultar abajo.')
""")
    md("""
## 6. Resultados guardados: evaluar el conjunto

Esta sección muestra los resultados de la versión indicada por `results/tier1_actual.json`.
**No dependen de cambios que acabes de hacer en CONFIG**: son mediciones guardadas de
entrenamientos terminados. La tabla cuenta cuántas de las tres semillas alcanzan ≥475.
La DE mide variación entre entrenamientos, no es un intervalo de confianza del 95%.

La búsqueda adicional de Q-learning, incluidos los intentos que fallaron y la
resolución por variable, está en [el estudio tabular](report/busqueda_qlearning.md).
""")
    code("""
ACTUAL = actual()
print('Versión mostrada:', ACTUAL['description'])
filas = resumen()
display(Image(filename=str(ROOT / ACTUAL['comparison'] / 'tier1_comparacion.png')))
""")
    md("""
### Cómo leer la gráfica

- **Horizontal:** pasos de entrenamiento, no episodios.
- **Vertical izquierda:** recompensa de evaluación sin explorar. Más alto es mejor.
- **Bandas:** IC t del 95% puntuales entre tres semillas. La interpretación supone normalidad aproximada y tiene limitaciones con tres réplicas; consulta Analisis_Teorico_e_Intervalos.ipynb.
- **Panel derecho:** retorno durante entrenamiento, donde el agente todavía explora.
- La línea de 475 orienta; el éxito se confirma con **100 episodios**, no solo con 10.

Para mantener el tamaño de evaluación de la curva, al inicio y al final se muestran
los **primeros 10** de los 100 episodios guardados. La tabla final usa los **100**;
por eso el último punto de la curva puede diferir de su valor en la tabla.

### Antes y después de las mejoras
Conservamos los experimentos originales. No ocultamos las ejecuciones que fallaron ni
seleccionamos una semilla exitosa como si representara a todas.
""")
    code("""
anteriores = json.loads((ROOT / 'results/parte_15_mse_completo/comparacion/summary.json').read_text(encoding='utf-8'))
for fila in anteriores:
    print(f"Antes | {fila['method']}: {fila['final_mean_100_episodes']:.2f} ± {fila['std_across_seed_means']:.2f}")
print('\\nDiscusión actual:')
display(Markdown((ROOT / ACTUAL['report']).read_text(encoding='utf-8')))
""")
    md("""
### Ver los parámetros exactos de un modelo

Escoge una semilla en esta celda. El archivo corresponde a un entrenamiento concreto;
la evaluación adicional de nuevos inicios se documenta en la discusión.
""")
    code("""
SEMILLA = 42  # Opciones: 42, 43, 44. No cambiar para buscar solo el mejor resultado.
CARPETA_DQN = ROOT / ACTUAL['dqn_root'] / f'seed_{SEMILLA}'
config_guardada = json.loads((CARPETA_DQN / 'config.json').read_text(encoding='utf-8'))
print('Configuración DQN guardada:')
print(json.dumps(config_guardada, indent=2, ensure_ascii=False))
print('Resumen:', json.loads((CARPETA_DQN / 'summary.json').read_text(encoding='utf-8')))
CARPETA_TABULAR = ROOT / ACTUAL['tabular_root'] / f"bins_{ACTUAL.get('tabular_bins', 10)}" / f'seed_{SEMILLA}'
config_tabular = json.loads((CARPETA_TABULAR / 'config.json').read_text(encoding='utf-8'))
claves = ['bins_per_dimension', 'training_steps', 'initial_q', 'alpha', 'alpha_end',
          'alpha_decay_steps', 'gamma', 'epsilon_decay_steps', 'epsilon_end', 'low', 'high']
print('Configuración Q-learning guardada:')
print(json.dumps({k: config_tabular[k] for k in claves if k in config_tabular}, indent=2, ensure_ascii=False))
print('Resumen:', json.loads((CARPETA_TABULAR / 'summary.json').read_text(encoding='utf-8')))
""")
    md("""
### Volver a comprobar los 100 episodios del modelo guardado

No se actualizan pesos. Se vuelven a usar las condiciones documentadas para comprobar
reproducibilidad; esto **no equivale a entrenar otra semilla independiente**.
""")
    code("""
REEVALUAR = False
if REEVALUAR:
    retornos = evaluar_guardado(CARPETA_DQN, episodes=100)
    print('Media:', np.mean(retornos), '| ¿≥475?:', np.mean(retornos) >= 475)
else:
    print('Reevaluación opcional desactivada; la tabla anterior usa evaluaciones ya guardadas.')
""")
    md("""
## 7. Ver el carrito dentro del notebook

Selecciona `AGENTE='dqn'` o `'tabular'` y una semilla. Se reproduce el primer episodio
de su evaluación. **No aprende durante esta demostración.** La animación incluye
controles de reproducción; el resultado de un episodio no sustituye la media de 100.

Activar `VER_CARRITO` genera la animación. También puedes ejecutar
`demostrar(carpeta, inline=False)` para abrir una ventana independiente.
""")
    code("""
VER_CARRITO = False
AGENTE = 'dqn'        # 'dqn' o 'tabular'
SEMILLA_DEMO = 42     # 42, 43 o 44
BINS_DEMO = ACTUAL.get('tabular_bins', 10)  # Puedes cambiarlo a 6, 10 o 20.
if VER_CARRITO:
    if AGENTE not in ('dqn', 'tabular'):
        raise ValueError('Escoge dqn o tabular')
    base = (ROOT / ACTUAL['dqn_root']) if AGENTE == 'dqn' else (ROOT / ACTUAL['tabular_root'] / f'bins_{BINS_DEMO}')
    carpeta = base / f'seed_{SEMILLA_DEMO}'
    demostrar(carpeta, episodes=1, inline=True)
else:
    print('Cambia VER_CARRITO a True y ejecuta esta celda para ver la animación.')
""")
    md("""
## 8. Qué debemos poder explicar los dos

1. ¿Qué representan las cuatro observaciones y los dos valores Q?
2. ¿Qué información perdemos al discretizar? ¿Por qué más intervalos no siempre ayudan?
3. ¿Qué hacen alpha, gamma y epsilon? ¿Por qué evaluar con epsilon=0?
4. ¿Para qué sirven replay y target? ¿Qué cambia con terminated y truncated?
5. ¿Por qué repetimos el entrenamiento con tres semillas y usamos el mismo presupuesto?
6. ¿Qué diferencia existe entre un episodio de 500 y una media ≥475 en 100 episodios?
7. ¿Qué mejoró y qué limitaciones siguen presentes?

**Archivos:** `agents/` algoritmos · `envs/` discretización · `experiments/` ejecución
· `configs/` parámetros · `results/` evidencias · `report/` discusión y uso de IA.

El notebook cubre Tier 1. Siguen pendientes Tier 2, Tier 3 para la pareja, el análisis
teórico, el artículo final y preparar la defensa. El docente exige que ambos entiendan
todo el código, incluyendo lo desarrollado con ayuda de IA.

**Referencias:** [Enunciado local](Project_01_Reinforcement_Learning_DQN.md),
[Gymnasium](https://gymnasium.farama.org/environments/classic_control/cart_pole/),
[CleanRL DQN](https://docs.cleanrl.dev/rl-algorithms/dqn/).
El agente entregado sigue siendo nuestra implementación.
""")
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (Project 01 .venv)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    })
    nbf.validate(nb)
    for index, cell in enumerate(cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook_cell_{index}", "exec")
    path = root / "Tier1_Explicado.ipynb"
    nbf.write(nb, path)
    print(f"Notebook generado: {path} ({len(cells)} celdas)")


if __name__ == "__main__":
    main()
