"""Crear dos entradas educativas, con código compartido y resultados verificables."""

from pathlib import Path
import nbformat as nbf
from experiments.seccion_mejora import add_refinement_cells

ROOT=Path(__file__).resolve().parents[1]
SETUP="""
from pathlib import Path
import sys, json, copy
import numpy as np
import torch
from IPython.display import display, Markdown
ROOT = next((p for p in (Path.cwd(), *Path.cwd().parents)
             if (p / 'agents/dqn.py').exists()), None)
if ROOT is None:
    raise RuntimeError('Abre la carpeta completa del proyecto en VS Code.')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.lunarlander import configuration, plan, make_env, train, evaluate
from experiments.ejecutar_tier23 import run
from experiments.cuaderno_lunar import OUTPUT, resumen, figura, discusion, demostrar
from experiments.cuaderno_tier1 import codigo
from agents.dqn import QNetwork, DQNAgent
from agents.replay import ReplayBuffer
from agents.double_dqn import DoubleDQNAgent
from agents.dqn_without_target import DQNWithoutTargetAgent
torch.set_num_threads(1)
torch.manual_seed(123)
CONFIG = configuration()
print('Python:', sys.executable)
print('Resultados iniciales:', OUTPUT)
"""


def make(tier):
    cells=[]
    def md(text): cells.append(nbf.v4.new_markdown_cell(text.strip()))
    def code(text): cells.append(nbf.v4.new_code_cell(text.strip()))
    title='Tier 2 — DQN en LunarLander' if tier==2 else 'Tier 3 — Double DQN en LunarLander'
    md(f"""
# {title}

Este notebook es una entrada independiente del Tier 1. Comparte los algoritmos de
`agents/` y la ejecución de `experiments/`; no contiene una copia alternativa del agente.

**VS Code:** abre la carpeta del proyecto, selecciona el kernel `.venv\\Scripts\\python.exe`
y usa **Shift+Enter** o **Ejecutar todo**. Tras actualizar el código, reinicia el kernel.
Las extensiones Python y Jupyter de Microsoft son necesarias.
La instalación de dependencias está documentada en el README.

Los entrenamientos y la animación se activan expresamente en sus celdas. Ejecutar todo
consulta resultados guardados y ejemplos. Si un estudio sigue en curso, se indica como pendiente.

La sección **Mejora: confirmación en las tres semillas** añade la nueva ronda.
Los estudios iniciales se conservan identificados como tales.
""")
    code(SETUP)
    if tier==2:
        md("""
## 1. Del carrito a una nave

En Tier 1 equilibrábamos una vara. Aquí intentamos aterrizar una nave en **LunarLander-v3**.
La observación contiene **8 valores**: posición horizontal/vertical, velocidades horizontal/vertical,
ángulo, velocidad angular y contacto de cada una de las dos patas con el suelo.
Son las observaciones del entorno, sin discretización tabular ni recompensa añadida.

Las **4 acciones** son: 0 no activar motores; 1 activar motor lateral izquierdo;
2 activar motor principal; 3 activar motor lateral derecho.

El objetivo del enunciado es **retorno medio ≥200 en 100 episodios**. Se premia acercarse,
reducir velocidad y aterrizar; usar motores tiene coste y estrellarse penaliza.
""")
        code("""
env = make_env(CONFIG)
try:
    obs, _ = env.reset(seed=123)
    print('Observación inicial:', obs)
    print('Observaciones:', env.observation_space)
    print('Acciones:', env.action_space)
    print('Máximo de pasos por episodio:', env.spec.max_episode_steps)
    print('Umbral declarado por la versión instalada:', env.spec.reward_threshold)
finally:
    env.close()
""")
        md("""
## 2. Parámetros y diseño del experimento

El diccionario es editable. Cambiarlo afecta entrenamientos nuevos, no los modelos ya guardados.
El presupuesto formal es `steps=500000`. El texto descriptivo `selection` del protocolo
congelado arrastra una errata que dice 200000; no gobierna el entrenamiento. Se conserva
el archivo histórico y se aclara aquí para mantener sus huellas de reproducción.
Usamos **tres semillas y el mismo presupuesto por modelo**. El baseline aparece en ambos
estudios y se reutiliza, evitando entrenar otra copia idéntica.

| Estudio | Qué cambia | Qué se mantiene |
|---|---|---|
| Tasa de aprendizaje | Tres valores de `learning_rate` | Los demás parámetros de base |
| Red objetivo | Tres valores de `target_sync_every` | Los demás parámetros de base |
| Ablación | Eliminar target y usar online en el objetivo | Replay, red, optimizador y presupuesto |

`hidden_widths`: neuronas; `batch_size`: experiencias por lote; `buffer_size`: memoria;
`learning_starts`: inicio de ajustes; `train_every`: pasos entre ajustes;
`gradient_steps`: ajustes realizados cada vez; `epsilon_decay_steps` y `epsilon_end`: exploración.
""")
        code("""
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
for nombre, trabajo in plan(CONFIG).items():
    print(nombre, '→', trabajo['role'], '| variante:', trabajo['variant'])
""")
        md("""
## 3. Reutilizamos nuestro DQN

La red ahora recibe 8 valores y devuelve 4 valores Q. Replay almacena transiciones de ocho
componentes. La red objetivo y la exploración cumplen la misma función que en Tier 1.
Estos bloques muestran **el código real** que ejecutan los experimentos.
""")
        code("codigo(QNetwork)\ncodigo(ReplayBuffer)")
        code("codigo(DQNAgent)")
        md(r"""
## 4. Cómo se construye el objetivo

\[
y=r+\gamma(1-\mathrm{terminated})\max_a Q_{target}(s',a)
\]

El fallo/fin real elimina el término futuro. Un corte por tiempo no lo elimina.
El valor Q no es una probabilidad. Tampoco usamos el límite Q≤100 de CartPole:
las recompensas de LunarLander son diferentes.
""")
        code("""
reward, gamma, next_best_q = 1.0, 0.9, 7.0
for terminated, truncated in [(False, False), (True, False), (False, True)]:
    target = reward + gamma * (not terminated) * next_best_q
    print('terminated:', terminated, '| truncated:', truncated, '→ objetivo:', target)
""")
        md("""
## 5. Ablación: ¿para qué sirve target?

La ablación conserva replay y todos los parámetros. Elimina la red independiente y calcula
el bootstrap con **online**, sin propagar gradiente por ese objetivo. Si empeora, aporta
evidencia de estabilización en esta configuración; no asumimos el resultado antes de medirlo.
""")
        code("codigo(DQNWithoutTargetAgent)")
        md("""
## 6. Entrenamiento y evaluación separados

Cada resultado formal incluye tres semillas. Cada cierto número de pasos evaluamos 10
episodios greedy; al inicio/final, 100. Confirmamos con 100 cuando un control supera 200.
Se guarda el modelo final, aunque haya tenido un control anterior mejor.

Evaluar usa otro entorno, una copia de la red y un generador independiente. Entrenamiento,
pérdidas y evaluaciones se guardan por separado en CSV.
""")
        code("codigo(evaluate)")
        code("codigo(train)")
    else:
        md("""
## 1. Qué exige Tier 3 a nuestro equipo

Como trabajamos en pareja, realizamos la opción **B: mejoras de DQN** en el mismo entorno
de Tier 2. Implementamos **Double DQN** y lo comparamos con el **DQN base**, con los mismos
parámetros, semillas, presupuesto y evaluación. No añadimos Dueling ni otra extensión.

El baseline es exactamente el mismo resultado de Tier 2, no otro entrenamiento elegido después.
""")
        code("""
print('Entorno:', CONFIG['environment'])
print('Pasos por modelo:', CONFIG['steps'], '| Semillas:', CONFIG['seeds'])
print('Parámetros comunes:', json.dumps(CONFIG['base'], indent=2))
""")
        md(r"""
## 2. DQN frente a Double DQN

En DQN, la red objetivo **elige y valora** la siguiente acción:
\[
y_{DQN}=r+\gamma Q_{target}(s',\arg\max_a Q_{target}(s',a)).
\]

En Double DQN, **online elige** y **target valora**:
\[
y_{Double}=r+\gamma Q_{target}(s',\arg\max_a Q_{online}(s',a)).
\]

En ambos se elimina el futuro cuando terminated=True. La idea busca reducir el sesgo
asociado a elegir máximos de estimaciones imperfectas. **No garantiza mejor recompensa
en todas las tareas o semillas**; eso es lo que comparamos experimentalmente.
""")
        code("codigo(DQNAgent.bootstrap_values)\ncodigo(DoubleDQNAgent)")
        md("""
## 3. Ejemplo numérico

Online prefiere la acción 0; target prefiere la 1. DQN tomará el valor 7 de target.
Double DQN consultará en target el valor de la acción elegida por online: 2.
Esto cambia el objetivo de aprendizaje, no los motores disponibles.
""")
        code("""
dqn = DQNAgent(obs_dim=8, n_actions=4, hidden_width=16, gamma=0.9, seed=123)
double = DoubleDQNAgent(obs_dim=8, n_actions=4, hidden_width=16, gamma=0.9, seed=123)
for agent in (dqn, double):
    with torch.no_grad():
        for p in agent.online.parameters(): p.zero_()
        for p in agent.target.parameters(): p.zero_()
        agent.online.layers[-1].bias.copy_(torch.tensor([9., 1., -1., -2.]))
        agent.target.layers[-1].bias.copy_(torch.tensor([2., 7., -1., -2.]))
        future = agent.bootstrap_values(torch.zeros((1, 8))).item()
    print(type(agent).__name__, '| valor futuro:', future, '| objetivo con r=1:', 1 + 0.9*future)
""")
        md("""
## 4. Compartimos el resto del código

Usamos la misma red, replay, función de pérdida, optimizador y calendario. Solo cambia
`bootstrap_values`. `learn` calcula el objetivo dentro de no_grad y actualiza online;
target permanece fija hasta la siguiente copia. Los episodios de evaluación no entrenan.
""")
        code("codigo(DQNAgent.learn)\ncodigo(DQNAgent.sync_target)")
        code("codigo(evaluate)")

    md("""
## Ejecutar una prueba corta (opcional)

La prueba usa un modelo pequeño y una semilla para comprobar el recorrido. **No es un
resultado formal** y no pretende alcanzar el objetivo. Se guarda en una carpeta nueva.
""")
    code("""
EJECUTAR_PRUEBA = False
if EJECUTAR_PRUEBA:
    from datetime import datetime
    prueba = copy.deepcopy(CONFIG)
    prueba.update(steps=2000, eval_every=2000, seeds=[123])
    prueba['base'].update(hidden_widths=[32,32], buffer_size=2000, batch_size=32,
                         learning_starts=128, gradient_steps=1)
    destino = ROOT / 'results/notebook_lunar' / datetime.now().strftime('prueba_%Y%m%d_%H%M%S_%f')
    run(prueba, destino, groups=['base'])
else:
    print('Prueba corta desactivada.')
""")
    md(f"""
## Repetir estudios iniciales de Tier {tier} (opcional, entrenamiento largo)

Activa la celda solo para volver a entrenar. La carpeta debe ser nueva si cambias parámetros.
Para continuar el protocolo original puedes usar OUTPUT: se reutilizan resultados completos
con el mismo protocolo y se rechaza sobrescribir ejecuciones parciales.

Los estudios formales necesitan al menos tres semillas. Tier 2 comprende seis grupos únicos;
Tier 3 comprende base y Double DQN. Si base ya existe, puede reutilizarse.
Para repetir la nueva comparación usa EJECUTAR_MEJORA en la sección siguiente.
""")
    groups="[n for n in plan(CONFIG) if n != 'double']" if tier==2 else "['base', 'double']"
    code(f"""
EJECUTAR_TIER{tier} = False
if EJECUTAR_TIER{tier}:
    from datetime import datetime
    if len(set(CONFIG['seeds'])) < 3:
        raise ValueError('El experimento formal requiere tres semillas independientes.')
    destino = ROOT / 'results/notebook_lunar' / datetime.now().strftime('tier{tier}_%Y%m%d_%H%M%S_%f')
    # Para continuar el mismo protocolo original: destino = OUTPUT
    run(CONFIG, destino, groups={groups})
else:
    print('Entrenamiento formal desactivado. Los resultados guardados se consultan abajo.')
""")
    add_refinement_cells(tier,md,code)
    md("""
## Estudios iniciales: 500000 pasos

Las tablas usan 100 episodios por modelo. La DE describe variación entre las medias de
las tres semillas. Las curvas usan 10 episodios por control, incluidos los primeros 10
al inicio/final; por ello su último punto puede diferir de la tabla final de 100.
Las bandas actuales son IC t del 95% puntuales entre tres semillas; sus supuestos
y límites se explican en Analisis_Teorico_e_Intervalos.ipynb.
Editar CONFIG no cambia estos resultados: corresponden al protocolo guardado en OUTPUT.
""")
    code(f"filas = resumen(tier={tier})")
    if tier==2:
        md("### Estudio de tasa de aprendizaje: solo cambia learning_rate")
        code("figura('learning_rate')")
        md("### Estudio de target: solo cambia target_sync_every")
        code("figura('target_sync_every')")
        md("### Ablación: DQN base frente a DQN sin red objetivo")
        code("figura('ablation')")
    else:
        code("figura('double_dqn')")
    code(f"discusion(tier={tier})")
    md("""
## Ver la nave dentro del notebook

Se muestra el primer episodio de la semilla escogida. No aprende durante la animación.
La recompensa de un episodio no sustituye la evaluación de 100 ni la repetición de semillas.
""")
    group="base" if tier==2 else "double"
    code(f"""
VER_NAVE = False
GRUPO = '{group}'
SEMILLA = 42  # 42, 43 o 44
if VER_NAVE:
    demostrar(GRUPO, SEMILLA, inline=True)
else:
    print('Cambia VER_NAVE a True para generar la animación.')
""")
    md("""
## Preguntas para la defensa

- ¿Qué observan las ocho entradas y qué representan las cuatro salidas Q?
- ¿Cómo distinguimos entrenar de evaluar? ¿Por qué tres semillas y 100 episodios finales?
- ¿Qué cambia exactamente en cada comparación y qué permanece igual?
- ¿Qué evidencia dan las curvas, la variación y los resultados negativos?
- ¿Por qué un Q grande o una brecha entre redes no prueba por sí sola sobreestimación verdadera?

El artículo final debe incorporar los resultados e interpretación. Los notebooks no sustituyen
el paper ni la defensa. Uso de IA: `report/uso_ia.md`.

Referencias: [Gymnasium](https://gymnasium.farama.org/environments/box2d/lunar_lander/),
[parámetros de referencia](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml),
[Double DQN](https://arxiv.org/abs/1509.06461).
""")
    nb=nbf.v4.new_notebook(cells=cells,metadata={"kernelspec":{"display_name":"Python (Project 01 .venv)","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12.0"}})
    for i,cell in enumerate(cells):
        if cell.cell_type=='code': compile(cell.source,f'tier{tier}_cell_{i}','exec')
    nbf.validate(nb)
    name='Tier2_LunarLander.ipynb' if tier==2 else 'Tier3_DoubleDQN.ipynb'
    nbf.write(nb,ROOT/name)
    print(f'Generado {name}: {len(cells)} celdas')


if __name__=='__main__':
    make(2)
    make(3)
