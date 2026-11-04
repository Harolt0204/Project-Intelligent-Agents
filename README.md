# Project 01 — Aprendizaje por refuerzo

Proyecto de dos integrantes, desarrollado en una sola computadora, por etapas.
El enunciado de referencia está en `Project_01_Reinforcement_Learning_DQN.md`.

**Estado actual:** Tier 1 experimental consolidado. Resumen vigente en
`report/qlearning_objetivo.md`; los ensayos exploratorios anteriores se conservan.
Tier 2 y Tier 3 tienen implementación y cuadernos separados; sus estudios están
en ejecución. El artículo final y la defensa siguen pendientes.

## Tier 2 y Tier 3: notebooks separados

- [Tier2_LunarLander.ipynb](Tier2_LunarLander.ipynb): DQN propio, estudios de tasa de aprendizaje y frecuencia de target, y ablación sin target.
- [Tier3_DoubleDQN.ipynb](Tier3_DoubleDQN.ipynb): Double DQN y comparación con el mismo DQN base.

Guía: [COMO_USAR_TIER2_TIER3.md](COMO_USAR_TIER2_TIER3.md). Los algoritmos se comparten
entre notebooks. Los entrenamientos largos están desactivados inicialmente; los
resultados pendientes se identifican sin presentarlos como evaluaciones terminadas.
La comparación formal tiene siete grupos únicos × tres semillas, con 500000 pasos
por modelo. DQN base se reutiliza en ambos estudios y Tier 3.

Cada experimento puede reproducirse por separado. Estos comandos reutilizan
ejecuciones completas; para volver a entrenar desde cero añade `--output` con
una carpeta nueva, como explica la guía.

| Figura / experimento | Comando desde esta carpeta |
|---|---|
| Tasa de aprendizaje | `.\.venv\Scripts\python.exe -m experiments.paralelo_tier23 --groups base learning_rate_0p0001 learning_rate_0p001` |
| Frecuencia de target | `.\.venv\Scripts\python.exe -m experiments.paralelo_tier23 --groups base target_sync_every_500 target_sync_every_8000` |
| Ablación | `.\.venv\Scripts\python.exe -m experiments.paralelo_tier23 --groups base no_target` |
| Double DQN | `.\.venv\Scripts\python.exe -m experiments.paralelo_tier23 --groups base double` |

Regenerar las cuatro figuras desde los CSV incluidos, sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.analizar_tier23
```

## Estudiar el código en VS Code

Abre **Tier1_Explicado.ipynb** y selecciona el kernel de `.venv`. Incluye explicaciones,
código real, parámetros editables, ejemplos, resultados y animación del carrito.
Guía: [Cómo usar el notebook](COMO_USAR_NOTEBOOK.md). En otra computadora, después
de preparar el proyecto, ejecuta **Preparar_Notebook.bat** una vez.

## Para el compañero: empezar aquí

1. Clona este repositorio o descarga y extrae su ZIP.
2. En Windows con Python 3.12, abre **Preparar_Proyecto.bat** una vez.
3. Abre **Mostrar_Tier1.bat** para ver los agentes ya entrenados.
4. Lee [los resultados actuales](report/qlearning_objetivo.md) y la [guía completa de reproducción](REPRODUCIR_TIER1.md).

Se incluyen modelos y resultados: no necesitas repetir los entrenamientos para
verlos funcionar. El entorno virtual se crea en cada computadora y no se sube a Git.

**Para verlo funcionar:** abre `Mostrar_Tier1.bat` con doble clic y selecciona
una simulación o la gráfica. Guía de presentación: `COMO_MOSTRAR_TIER1.md`.
Se cargan los modelos ya entrenados; no se vuelve a entrenar.

## Resultados actuales del Tier 1

| Método seleccionado | Media final ± DE | Nuevos inicios ± DE | Semillas ≥475 |
|---|---:|---:|---:|
| Q-learning (2/2/6/6 intervalos) | 495.73 ± 6.04 | 496.47 ± 4.99 | 3/3 en ambas evaluaciones |
| DQN propio ajustado | 498.60 ± 1.98 | 498.61 ± 1.97 | 3/3 en ambas evaluaciones |

Cada modelo entrena 200000 pasos y se evalúa en 100 episodios. Q-learning pasó
por 487.19, 500 y 500; con nuevos inicios obtuvo 489.41, 500 y 500.
Se entregan modelos finales, sin elegir su mejor checkpoint.

La tabla conserva cuatro observaciones: 2 intervalos para posición, 2 para velocidad,
6 para ángulo y 6 para velocidad angular (144 estados). Empieza con Q=100 y reduce
alpha de 0.1 a 0.01. Las variantes angulares de 10 y 20 también se probaron con los
mismos parámetros y presupuesto; no alcanzaron 475 y se conservan en la comparación.

Informe: [qlearning_objetivo.md](report/qlearning_objetivo.md). La [búsqueda tabular](report/busqueda_qlearning.md)
incluye controles y resultados negativos. El cierre anterior permanece en
[mejora_tier1.md](report/mejora_tier1.md); las partes 1–17 de abajo son el historial previo.

## Ruta de trabajo

1. **Completado:** entender CartPole y el ciclo observación → acción → recompensa.
2. **Completado:** discretización y agente Q-learning propio.
3. **Completado:** comparar 3 granularidades de discretización con múltiples semillas (Parte 6).
4. **Completado, con limitaciones documentadas:** comparación DQN/Q-learning a 200 000 pasos y evaluación adicional con inicios nuevos.
5. Tier 2: DQN en LunarLander, dos estudios de hiperparámetros y una ablación.
6. Tier 3: Double DQN sobre el entorno de Tier 2.
7. Análisis teórico, resultados reproducibles, artículo y defensa.

CartPole, LunarLander y Double DQN son el alcance inicial. La viabilidad de
LunarLander se comprobará antes de iniciar Tier 2.

## Preparación (PowerShell, desde esta carpeta)

Se utiliza Python 3.12 con un entorno propio del proyecto.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

No es necesario activar el entorno: se invoca su Python directamente.
Las dependencias incluyen las etapas tabulares, gráficas y PyTorch para CPU.

## Parte 1: explorar CartPole

```powershell
.\.venv\Scripts\python.exe experiments\01_explorar_cartpole.py
```

Para observar la animación en tu computadora:

```powershell
.\.venv\Scripts\python.exe experiments\01_explorar_cartpole.py --render
```

El programa muestra el estado inicial, las primeras tres transiciones y la
duración de cinco episodios. Usa acciones aleatorias: **todavía no aprende**.
Esta demostración no constituye un resultado experimental para el artículo.

### Conceptos para entender antes de avanzar

- **Observación:** posición y velocidad del carrito, ángulo y velocidad angular
  del poste. Son cuatro números continuos; el ángulo está en radianes.
- **Acción:** `0` empuja a la izquierda y `1` a la derecha.
- **Recompensa:** con la configuración estándar, +1 por paso, incluido el de caída.
- **Retorno:** suma de las recompensas de un episodio.
- **Paso:** una interacción con el entorno mediante `step(action)`.
- **Episodio:** desde `reset()` hasta un final; CartPole-v1 permite hasta 500 pasos.
- **terminated:** final por condiciones de la tarea, como la caída del poste.
- **truncated:** final por un límite externo, como agotar el tiempo permitido.
- **Semilla:** permite repetir la secuencia aleatoria con el mismo programa y
  versiones. El entorno y el muestreador de acciones se siembran por separado.

Para reiniciar se comprueba `terminated or truncated`. Cuando implementemos
Q-learning, esta diferencia también importará al calcular el valor objetivo.

Antes de avanzar, debemos poder explicar: ¿por qué este agente no aprende?,
¿qué diferencia hay entre paso y episodio?, ¿por qué más retorno es mejor aquí?

## Parte 2: convertir observaciones en casillas

Desde esta carpeta:

```powershell
.\.venv\Scripts\python.exe -m experiments.discretizar_cartpole
```

Una tabla no puede almacenar una fila por cada combinación de números reales.
Dividimos el rango de cada variable en intervalos (bins) y representamos cada
observación mediante cuatro índices. Con 6 bins, cada índice va de 0 a 5.
Dos observaciones diferentes pueden compartir la misma casilla: ganamos una
representación finita a costa de perder detalle.

El programa crea tablas vacías para visualizar su tamaño. Los ceros son valores
iniciales; no representan conocimiento aprendido ni resultados experimentales.
Cada casilla tiene dos valores Q, uno por acción, que estimarán el retorno futuro
descontado si tomamos esa acción y seguimos la política adecuada.

| Bins por variable | Casillas (bins⁴) | Valores Q (2 por casilla) |
|---|---:|---:|
| 6 | 1 296 | 2 592 |
| 10 | 10 000 | 20 000 |
| 20 | 160 000 | 320 000 |

### Rangos iniciales de diseño

| Variable | Rango discretizado | Motivo |
|---|---|---|
| Posición | −2.4 a 2.4 m | Límites de fallo del carrito |
| Velocidad | −3 a 3 m/s | Elección provisional; revisar cobertura durante entrenamiento |
| Ángulo | −12° a 12° (convertido a radianes) | Límites de fallo del poste |
| Velocidad angular | −3.5 a 3.5 rad/s | Elección provisional; revisar cobertura durante entrenamiento |

Estos rangos no son los límites completos de `observation_space`. Las velocidades
se declaran sin límites finitos. Una observación fuera de nuestros rangos se
asigna a la casilla del extremo correspondiente; no se modifica la simulación.
Mediremos la frecuencia de estos casos al entrenar para justificar los rangos.
Las fronteras exactas entre intervalos se asignan al intervalo de la derecha.

Más bins no garantiza mejor aprendizaje: distingue más situaciones, pero cada
casilla puede recibir menos visitas con el mismo presupuesto de pasos.
El siguiente paso será actualizar los valores Q a partir de la experiencia.

## Parte 3: aprender de una transición

```powershell
.\.venv\Scripts\python.exe -m experiments.aprender_un_paso
```

El agente propio está en `agents/tabular.py`. Contiene una tabla inicialmente
en cero, elección epsilon-greedy y la actualización:

```text
objetivo = recompensa + gamma × mayor Q de la siguiente casilla
nuevo Q = Q anterior + alpha × (objetivo − Q anterior)
```

- `alpha`: tasa de aprendizaje, cuánto corregimos en cada experiencia.
- `gamma`: descuento de recompensas futuras.
- `epsilon`: probabilidad de seleccionar una acción aleatoria.

Primero se reproduce el ejemplo inventado del chat: 5 → 5.5. Después se crea
otra tabla vacía y se ejecuta un episodio real mostrando sus primeros ajustes.
Los números inventados nunca se usan para inicializar el episodio real.
La demostración usa alpha=0.1, gamma=0.9 y epsilon=0.5 fijo por claridad;
estos valores todavía no son la configuración del experimento formal.

Si `terminated=True`, el objetivo es solo la recompensa inmediata, porque la
tarea terminó. Si únicamente `truncated=True` por el límite temporal, se conserva
la estimación futura. En ambos casos se reinicia el episodio. En los empates de
la tabla se elige al azar entre las acciones con el mayor valor.

Este ejercicio permite ver que la tabla cambia; no prueba que el agente ya sea
mejor. Para eso necesitaremos entrenamientos y evaluaciones separados.
El siguiente paso es repetir este ciclo durante muchos episodios, conservar la
tabla entre episodios y reducir gradualmente epsilon.

Comprobaciones del agente:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Parte 4: primer entrenamiento repetido

```powershell
.\.venv\Scripts\python.exe -m experiments.entrenar_tabular
```

Piloto didáctico con 10 bins, 30 000 pasos por semilla y semillas 42, 43 y 44.
Cada semilla comienza con una tabla vacía y la conserva entre episodios.
Se utiliza alpha=0.1 y gamma=0.99. Epsilon baja linealmente de 1 a 0.05 en
los primeros 24 000 pasos; permanece en 0.05 durante el resto.

Entrenar significa ejecutar acciones y actualizar la tabla. Evaluar significa
usar una copia de la tabla con epsilon=0 y sin actualizarla. La evaluación usa
otro entorno y otro generador aleatorio, sin interferir con el entrenamiento.
Los empates entre acciones se siguen resolviendo al azar.

Se evalúa antes de aprender y al final sobre 100 episodios, y cada 5 000 pasos
intermedios sobre 10. Se repiten las condiciones iniciales de evaluación entre
checkpoints para comparar; no constituyen una prueba de generalización a nuevas
condiciones. El tiempo de entrenamiento excluye las evaluaciones; también se
registra el tiempo total. Los pasos de evaluación no gastan el presupuesto de
pasos de entrenamiento.

En `results/parte_04_tabular` se guardan, por semilla:

- `config.json`: parámetros, rangos, versiones y datos disponibles del hardware.
- `training.csv`: retornos de entrenamiento por episodio; `budget_cut` identifica
  un intento que quedó incompleto al agotar el presupuesto.
- `evaluation.csv`: resultados de evaluación, incluidos los retornos individuales.
- `q_table.npy`: tabla aprendida, reutilizable sin volver a entrenar.
- `summary.json`: rendimiento antes/después, tiempos y cobertura de casillas.
- Los conteos fuera de rango se calculan sobre los estados de entrada a las
  acciones de entrenamiento, no sobre estados terminales ni evaluaciones.

El archivo `aggregate.json` resume las tres semillas. La desviación entre medias
de semillas es diferente de la variación entre episodios de una sola evaluación.
La comparación formal con 3 granularidades, gráficas y DQN sigue pendiente.
Que la tabla cambie no garantiza una mejora continua o que llegue al umbral 475.

Para repetir sin sobrescribir resultados anteriores:

```powershell
.\.venv\Scripts\python.exe -m experiments.entrenar_tabular --output results/parte_04_repeticion
```

## Parte 5: observar el comportamiento aprendido

```powershell
.\.venv\Scripts\python.exe -m experiments.ver_agente
```

Carga la tabla de la semilla 42, sin entrenar de nuevo. Genera
`results/parte_05_visual/comparacion.gif`: izquierda con tabla en cero y derecha
con tabla aprendida. En ambos casos epsilon=0; la tabla vacía elige al azar al
estar todas las acciones empatadas. Se usa el mismo estado inicial, con semilla
10042, correspondiente al primer episodio de evaluación. No se selecciona un
intento especialmente exitoso. El lado que termina primero conserva su último
fotograma mientras el otro continúa.

La animación representa un único intento, no el promedio de los 100 episodios
evaluados. `demostracion.json` registra sus semillas, retornos y causas de final.
Las tablas originales no se actualizan. Se comprueba que los rangos y dimensiones
de la tabla cargada coincidan con los del discretizador actual.

## Parte 6: comparar 6, 10 y 20 intervalos por variable

Entrenar las nueve configuraciones (tres granularidades × tres semillas) y generar
las figuras, desde esta carpeta:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_bins
```

Solo regenerar las figuras a partir de los logs guardados:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_bins --plot-only
```

Para un nuevo entrenamiento sin sobrescribir los anteriores, agregar
`--output results/parte_06_repeticion`. El protocolo se guarda antes de entrenar:

- 6, 10 y 20 bins; semillas independientes 42, 43 y 44 para cada granularidad.
- 200 000 pasos por ejecución; nueve ejecuciones secuenciales en la misma máquina.
- Alpha=0.1, gamma=0.99; epsilon de 1 a 0.05 en 160 000 pasos y luego fijo.
- Igual rango de discretización, recompensas, calendario y condiciones de evaluación.
- Cada 10 000 pasos, evaluación greedy de 10 episodios. Las evaluaciones inicial
  y final utilizan 100. Los estados iniciales se repiten entre checkpoints y
  granularidades para controlar la comparación.

`results/parte_06_bins` contiene los logs y tablas por semilla, `summary.csv/json`,
los datos agregados de las curvas en `curves.csv` y las figuras PNG/PDF.
Las líneas muestran la media entre tres semillas. El área sombreada muestra
±una desviación estándar entre las medias por semilla (`ddof=0`); no es un
intervalo de confianza ni la variación de los 100 episodios de un solo agente.
No se suavizan las curvas. Para mantener el mismo número de episodios en todos
los puntos, las curvas usan siempre los primeros 10 episodios de cada checkpoint,
incluso cuando el log inicial/final contiene 100. El resumen final sí usa los 100.

Para afirmar éxito se usa la evaluación de 100 episodios, no un checkpoint de
solo 10. Este protocolo confirma el criterio al inicio y al final; no localiza
exactamente el primer paso en que un agente podría haber alcanzado 475.
La comparación con DQN y el análisis final de Tier 1 siguen pendientes.

Resultados de esta ejecución: retorno final medio 295.33 (6 bins), 359.70
(10 bins) y 209.00 (20 bins). Ninguna de las nueve ejecuciones alcanzó 475 en
la evaluación final de 100 episodios. Interpretación y límites en
`report/notas_discretizacion.md`; estos resultados no establecen una configuración
universalmente óptima.

## Parte 7: primera red Q (sin entrenar)

```powershell
.\.venv\Scripts\python.exe -m experiments.explorar_red_q
```

La clase `QNetwork` de `agents/dqn.py` recibe directamente las observaciones
continuas de CartPole, sin discretizarlas. Arquitectura inicial elegida por el
equipo: 4 → 64 → 64 → 2, con ReLU entre capas y salida lineal. No es una
arquitectura impuesta por el docente ni se afirma que sea óptima.

Las cuatro entradas son posición, velocidad, ángulo y velocidad angular. Las
dos salidas son Q(izquierda) y Q(derecha). Las capas intermedias combinan los
números mediante pesos y sesgos; ReLU pone a cero las salidas negativas de las
capas intermedias para introducir una transformación no lineal. La salida final
no lleva ReLU: un valor Q puede ser negativo, no es un porcentaje y no debe
sumar 1 con las otras acciones.

La red tiene 4 610 parámetros ajustables, inicializados aleatoriamente con
semilla 42. Consultar la red todavía no modifica sus parámetros. La demostración
registra una predicción real para el estado inicial en
`results/parte_07_red_q/demostracion.json`, marcada explícitamente como no entrenada.
Sus valores Q aún no son estimaciones útiles de la recompensa futura.

Esta clase es solo un componente de DQN: todavía faltan replay buffer, red
objetivo, política de exploración y bucle de actualización de pesos. Las pruebas
verifican compatibilidad de estados individuales y lotes, que se puedan calcular
correcciones para todas las capas y que la consulta no cambie los pesos.

Fuentes consultadas:

- [Instalación oficial de PyTorch](https://pytorch.org/get-started/locally/).
- [Capas lineales de PyTorch](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html).

## Parte 8: memoria de experiencias (replay buffer)

```powershell
.\.venv\Scripts\python.exe -m experiments.explorar_memoria
```

Una experiencia guarda: observación, acción, recompensa, observación siguiente
y los indicadores `terminated` y `truncated`. La memoria propia está en
`agents/replay.py`; usa arrays NumPy y no contiene ningún agente prefabricado.

El ejemplo ejecuta 12 pasos con acciones aleatorias en CartPole y los añade a
una memoria con capacidad de 8. Quedan las ocho experiencias más recientes.
Después recupera cuatro al azar, sin repetición dentro de ese lote. Muestrear
no borra recuerdos; una experiencia puede volver a usarse en otro lote.
Las observaciones se copian al guardarlas y los lotes no comparten arrays con
la memoria. Se conservan ambos tipos de final para calcular correctamente los
objetivos de DQN en la siguiente etapa.

¿Por qué mezclar recuerdos? Los pasos consecutivos suelen parecerse mucho.
Tomar experiencias al azar reduce la dependencia temporal del lote y permite
reutilizar interacciones anteriores. No garantiza experiencias independientes
ni una mejora por sí solo; mediremos su efecto con una ablación en Tier 2.

La capacidad 8 y el lote de 4 son solo para entender el funcionamiento; no son
la configuración de entrenamiento final. La demostración se guarda en
`results/parte_08_memoria/demostracion.json`. No se modifica ninguna red.

## Parte 9: red objetivo y primer ajuste de los pesos

```powershell
.\.venv\Scripts\python.exe -m experiments.explorar_red_objetivo
```

`DQNAgent` en `agents/dqn.py` contiene dos redes independientes. La red **online**
elige acciones y recibe los ajustes del optimizador. La red **target** empieza
como una copia idéntica y permanece fija entre sincronizaciones. No representa
respuestas correctas conocidas: es una estimación mantenida temporalmente fija.
Esto evita cambiar la referencia con cada ajuste, aunque no garantiza estabilidad.

Para cada recuerdo del lote calculamos:

```text
predicción = online(estado)[acción que se tomó]
objetivo = recompensa + gamma × mayor Q de target(estado siguiente)
```

Si `terminated=True`, el objetivo es solo la recompensa. Un corte temporal
(`truncated=True` sin `terminated`) conserva el término futuro. Los objetivos
no reciben gradientes y solo se ajustan los parámetros de online.
Se usa pérdida Smooth L1 (Huber), optimizador Adam y límite 10 para la norma
del gradiente. La tasa de aprendizaje inicial de esta demostración es 0.001.

El ejemplo guarda 64 experiencias, muestrea ocho y hace un único ajuste. Registra
el error antes y después contra **los mismos objetivos**. Comprueba que online
cambie y target permanezca igual; después copia online a target explícitamente.
Se guarda el resultado en `results/parte_09_red_objetivo/demostracion.json`.

La frecuencia de copia de la red objetivo se fijará en el bucle de entrenamiento;
no se copia tras cada llamada a `learn()`. Esta demostración copia una sola vez
para comprobar el mecanismo. Un menor error en un lote no demuestra mejora del
retorno: el siguiente paso será integrar las piezas en un entrenamiento evaluado.

## Parte 10: entrenamiento DQN y comparación Tier 1

Entrenar DQN desde cero (tres semillas independientes, 200 000 pasos cada una):

```powershell
.\.venv\Scripts\python.exe -m experiments.entrenar_dqn
```

Generar las figuras de evaluación y entrenamiento usando los logs existentes
de DQN y las tres granularidades tabulares:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_tier1
```

Generar un GIF del modelo final de la primera semilla, sin reentrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.ver_dqn
```

Configuración fijada antes de mirar los resultados: red 4-64-64-2, Adam con
learning rate 0.0005, gamma=0.99, replay buffer de 50 000 experiencias, lote de
64 y primer ajuste en el paso 1 000. Se realiza un ajuste cada cuatro pasos
y se copia la red objetivo cada 500 pasos (después del ajuste si coinciden).
Epsilon decae de 1 a 0.05 en los primeros 160 000 pasos. El entrenamiento utiliza
un hilo de PyTorch en CPU y algoritmos deterministas. El tiempo registrado
depende de esta implementación y configuración de hilos; no es una comparación
universal de velocidad entre familias de algoritmos.

Las semillas 42/43/44 y el presupuesto de 200 000 pasos coinciden con la
comparación tabular. Cada 10 000 pasos se evalúa una copia sin exploración ni
aprendizaje en 10 episodios; al inicio y al final se evalúa en 100. El entorno
de evaluación tiene las mismas semillas de inicio que el protocolo tabular y
no afecta a la memoria, pesos ni generadores del entrenamiento.

Si un control intermedio de 10 episodios obtiene ≥475, se realiza una
comprobación adicional de 100 episodios. `threshold_checks.json` registra cada
comprobación. El primer control confirmado no identifica el instante exacto
de aprendizaje ni demuestra que el rendimiento se mantenga después.
El protocolo tabular anterior solo verificó 100 episodios al inicio y al final;
por eso no se comparan los tiempos de primer éxito como si su detección fuera
idéntica. El modelo entregado es siempre el final, sin seleccionar el mejor.

En `results/parte_10_dqn/seed_*` se guardan configuraciones, retornos separados
de entrenamiento y evaluación, pérdidas por ajuste, comprobaciones del umbral,
modelos finales online/target y resúmenes de tiempos. `aggregate.json` resume las
tres semillas. Los resultados anteriores no se sobrescriben; para repetir usar
`--output results/parte_10_repeticion` y pasar esa ruta con `--dqn-root` al graficar.

Las figuras PNG/PDF y tablas CSV/JSON están en `results/parte_10_dqn/comparacion`.
La curva de evaluación usa siempre los primeros 10 episodios de cada control;
la tabla final usa los 100. La curva de entrenamiento agrupa episodios completos
según su paso de finalización en bloques de 10 000 pasos, sin mezclar evaluación.
Las bandas son ±una desviación estándar entre las tres medias por semilla.
Las curvas incluyen las tres discretizaciones tabulares para evitar ocultar
configuraciones menos favorables. La comparación no demuestra significancia
estadística ni generalización fuera de las condiciones evaluadas.

Diagnóstico reproducible de las estimaciones finales:

```powershell
.\.venv\Scripts\python.exe -m experiments.diagnosticar_dqn
```

Consulta cada modelo final en 1 000 estados obtenidos mediante acciones
aleatorias, con semillas nuevas `90000 + seed`. Guarda extremos de Q, frecuencias
fuera del intervalo teórico y pérdidas iniciales/finales en `q_diagnostic.json`.
Con recompensas entre 0 y 1 y gamma=0.99, los retornos **descontados** verdaderos
están entre 0 y 100. Una aproximación puede salirse de ese intervalo; encontrar
valores muy superiores indica sobreestimación en esos estados, pero no identifica
por sí solo la causa del fallo. Este límite no es el máximo retorno sin descuento
de un episodio de CartPole, que es 500.

## Parte 11: estudiar una tasa de aprendizaje menor

Hipótesis: reducir la tasa de 0.0005 a 0.0001 puede hacer más estables los
ajustes de la red. No se presupone mejora. Solo cambia este hiperparámetro;
se conservan arquitectura, semillas, pasos, memoria, exploración y red objetivo.

```powershell
.\.venv\Scripts\python.exe -m experiments.entrenar_dqn --learning-rate 0.0001 --output results/parte_11_lr
```

Generar el mismo diagnóstico de estimaciones Q y después comparar ambas tasas:

```powershell
.\.venv\Scripts\python.exe -m experiments.diagnosticar_dqn --root results/parte_11_lr
.\.venv\Scripts\python.exe -m experiments.comparar_learning_rate
```

También se puede regenerar la comparación completa contra las tres tablas:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --dqn-root results/parte_11_lr
```

Para el GIF del primer modelo final de esta nueva configuración:

```powershell
.\.venv\Scripts\python.exe -m experiments.ver_dqn --run results/parte_11_lr/seed_42
```

Los resultados anteriores permanecen en `parte_10_dqn`. La comparación verifica
automáticamente que las configuraciones guardadas solo difieran en learning_rate.
El resumen usa 100 episodios finales y presenta las diferencias por semilla.
Las curvas usan 10 episodios por checkpoint, con media y DE entre tres semillas.
Q-learning de 10 intervalos aparece como referencia seleccionada en el estudio
tabular anterior; no representa una configuración universalmente óptima.

Estas son dos tasas en CartPole, elegidas secuencialmente después de examinar
el fallo inicial. Esto no completa el estudio de Tier 2, que exige al menos dos
hiperparámetros con tres valores cada uno, ni demuestra generalización a nuevas
semillas de evaluación. El modelo final siempre se conserva sin elegir el mejor.

Resultado: reducir la tasa empeoró las tres semillas; media final 9.37 ± 0.01
frente a 48.47 ± 25.25 del experimento inicial. La sobreestimación persistió y
se agravó en la muestra diagnóstica. Las 17 pruebas pasaron y se reprodujeron
exactamente las evaluaciones finales desde los modelos guardados. Detalles en
`report/notas_learning_rate.md`; no se concluye que toda tasa menor sea peor.

## Parte 12: prueba corta del ritmo de copia de target

Hipótesis: conservar la referencia durante 2 000 pasos, en vez de 500, podría
frenar el crecimiento inestable de las estimaciones. Se vuelve a la tasa original
0.0005. La comparación local cambia solo el intervalo de copia de target.

```powershell
.\.venv\Scripts\python.exe -m experiments.probar_target
```

Son seis ejecuciones secuenciales: 50 000 pasos × tres semillas × dos intervalos.
Se reentrena también la referencia de 500 para evaluar ambos finales con
100 episodios. Epsilon conserva el calendario original: baja de 1 a 0.05 en
160 000 pasos, aunque la prueba termina en 50 000. No se acelera la exploración
por acortar el presupuesto. Por tanto, aún no se llega a la fase de epsilon fijo.

Antes de aprender se recogen 256 estados de una política aleatoria, con semilla
90000 + seed, como banco fijo de diagnóstico. No entran al replay buffer. En
cada checkpoint se consulta el banco para registrar mínimos/máximos de Q y
fracción superior al límite teórico 100. Estas consultas no actualizan la red
ni consumen el generador aleatorio del entrenamiento. Sus pasos no cuentan
como pasos de entrenamiento; el tiempo de construir el banco tampoco se incluye
en el tiempo de entrenamiento registrado.

Los resultados están en `results/parte_12_target_corto`. La figura muestra media
y DE entre semillas del retorno de 10 episodios por control y, por separado,
el máximo Q diagnóstico de cada semilla. Ese panel usa escala simétrica
logarítmica para poder observar crecimientos grandes sin ocultar el inicio.
El resumen final usa 100 episodios. Comprobaciones automáticas: solo cambia
el intervalo entre configuraciones y los primeros 10 retornos de cada control
de la referencia reproducen los correspondientes controles del experimento
original de 200 000 pasos.

Regenerar la figura sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.probar_target --plot-only
```

Para repetir sin sobrescribir resultados, agregar `--output` con otra carpeta.
Una mejora a 50 000 pasos no confirma estabilidad a 200 000; no se comparan estos
finales cortos con los finales largos como si tuvieran el mismo presupuesto.

Resultado de la prueba: 104.10 ± 35.98 con copia cada 500 y 75.37 ± 27.89 cada
2 000. Las estimaciones crecieron más despacio con 2 000, pero no mejoró el
retorno final promedio. Interpretación y límites en `report/notas_target_corto.md`.

## Parte 13: auditoría del agente

```powershell
.\.venv\Scripts\python.exe -m experiments.auditar_dqn
```

Comprueba experiencias reales, reinicios, objetivos y pérdida contra cálculos
NumPy independientes y un gradiente por diferencias finitas. También ejecuta
un MDP de dos estados con valores óptimos conocidos en tres semillas. Guarda
`results/parte_13_auditoria/audit.json`. Es una auditoría, no otro entrenamiento
de CartPole. Todas las comprobaciones pasaron; no se confirmó un error en las
partes revisadas ni se demuestra ausencia de cualquier error posible.

La comparación con fuentes primarias de CleanRL y Gymnasium y sus límites se
documentan en `report/auditoria_dqn.md`. Se encontraron diferencias de diseño,
como Smooth L1 frente a MSE, que pueden estudiarse de forma aislada. No se copió
un agente prefabricado ni se sustituyeron los resultados negativos anteriores.

## Parte 14: comparar Smooth L1 frente a MSE

```powershell
.\.venv\Scripts\python.exe -m experiments.probar_loss
```

Se reutilizan los tres controles Smooth L1 de `parte_12_target_corto/target_500`
y se entrenan tres modelos nuevos con MSE. Ambos tienen 50 000 pasos, semillas
42/43/44, tasa 0.0005, target cada 500 pasos y decaimiento de epsilon de 160 000.
La comparación verifica que solo difiera el campo de pérdida entre configuraciones.
Las reglas y semillas de recolección son iguales; los recuerdos pueden diferir
cuando el cambio de aprendizaje produce acciones distintas.

MSE calcula la media de errores al cuadrado. Smooth L1 es cuadrática para errores
pequeños y lineal para grandes. Cambiar la pérdida modifica la ponderación y
escala de los ajustes, en interacción con Adam y el límite común de gradiente.
No se comparan sus valores numéricos de pérdida como si fueran la misma métrica;
se comparan retorno y puntuaciones Q, que sí conservan su significado.

El modelo mantiene Smooth L1 como opción por defecto. Para usar MSE directamente
con el entrenador general existe `--loss mse`. No se sustituyen los experimentos
anteriores ni se cambia su configuración guardada.

Resultados y figuras PNG/PDF: `results/parte_14_loss`. Curvas con 10 episodios por
checkpoint, media y DE entre semillas, sin suavizado. Resumen final con 100
episodios por agente. La magnitud Q usa el mismo banco de 256 estados, que no
participa en el entrenamiento. Una mejora a 50 000 pasos no confirma estabilidad
a 200 000 ni equivale a superar el umbral 475 en evaluación final.

Regenerar figuras sin reentrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.probar_loss --plot-only
```

La prueba numérica nueva usa objetivos conocidos 1 y 10 y predicciones cero:
MSE debe dar (1² + 10²)/2 = 50.5. Se comprueba que los objetivos no cambien.

Resultado: MSE obtuvo 244.68 ± 29.68 frente a 104.10 ± 35.98 con Smooth L1;
mejoró las tres semillas a 50 000 pasos, sin alcanzar 475. También mostró valores
Q por encima del máximo teórico en parte del banco diagnóstico. Interpretación
y límites en `report/notas_loss.md`. Las 18 pruebas y la reproducción exacta de
las evaluaciones finales desde los modelos guardados pasaron.

GIF del primer modelo final:

```powershell
.\.venv\Scripts\python.exe -m experiments.ver_dqn --run results/parte_14_loss/mse/seed_42
```

## Parte 15: MSE completo y cierre experimental de Tier 1

Entrenar MSE a 200 000 pasos, con las tres semillas y calendario original:

```powershell
.\.venv\Scripts\python.exe -m experiments.entrenar_dqn --loss mse --output results/parte_15_mse_completo
```

No se añade el cambio de target a 2 000. El único cambio de aprendizaje respecto
al DQN inicial completo es la pérdida; la instrumentación de diagnóstico no
participa en el entrenamiento. Los primeros 50 000 pasos conservan el calendario
de la prueba corta MSE. Se empieza desde cero porque aquella prueba no guardó
el estado completo necesario para una continuación exacta (optimizador y replay).

Comparación con los tres tabulares y con el primer DQN fallido, todos a igual
presupuesto. Este comando reproduce las figuras de entrenamiento/evaluación:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --dqn-root results/parte_15_mse_completo --dqn-label "DQN (MSE)" --extra-dqn-root results/parte_10_dqn --extra-dqn-label "DQN (Smooth L1 inicial)"
```

Diagnóstico y GIF del modelo final de la primera semilla:

```powershell
.\.venv\Scripts\python.exe -m experiments.diagnosticar_dqn --root results/parte_15_mse_completo
.\.venv\Scripts\python.exe -m experiments.ver_dqn --run results/parte_15_mse_completo/seed_42
```

Evaluación adicional de los cinco métodos con 100 condiciones iniciales nuevas
por semilla de entrenamiento, sin modificar los modelos ni usarlas para ajustar:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluar_tier1_nuevas_semillas
```

Las semillas de reset son `1010000 + 1000 * seed + episodio`; los intervalos de
100 episodios no se solapan entre las tres semillas entrenadas. Cada algoritmo
se compara en las mismas condiciones de su semilla correspondiente. Los datos
se guardan en `results/parte_15_mse_completo/evaluacion_nueva`; el comando protege
el resultado existente para evitar escoger otra muestra después de verlo.
Para reproducir la misma evaluación sin sobrescribirla, usar `--output` con otra
carpeta: las semillas permanecen fijas, no se elige una muestra nueva al repetir.
Esto complementa la evaluación usada durante el desarrollo, pero no prueba
rendimiento universal ni sustituye las demás limitaciones del estudio.

Regenerar el resumen de cierre desde las métricas guardadas:

```powershell
.\.venv\Scripts\python.exe -m experiments.resumir_tier1
```

Resultados completos: Q-learning de 10 intervalos obtuvo 359.70 ± 102.57;
DQN MSE, 234.10 ± 188.02; DQN inicial, 48.47 ± 25.25. Una de las tres semillas
MSE obtuvo media 500 en 100 episodios; las otras dos quedaron cerca de 101.
La evaluación nueva de 1 500 episodios sobre los 15 modelos mantuvo el patrón:
352.87 para el tabular de 10 intervalos y 234.19 para MSE, que volvió a resolver
los 100 episodios de su semilla exitosa. La sobreestimación Q persiste.

Las verificaciones incluyen presupuestos, 49 751 ajustes y 400 copias de target
por modelo DQN nuevo, coincidencia del prefijo de evaluación de la prueba corta
y reproducción exacta de los retornos finales desde los modelos guardados.
El archivo `report/nota_avance_tier1.md` sirve como borrador de la nota corta
de progreso solicitada por el enunciado; no se ha enviado al docente.

## Protocolo que aplicaremos a los experimentos

La primera comparación completa está documentada en
`report/notas_dqn_cartpole.md`: DQN obtuvo 48.47 ± 25.25 de retorno final y
mostró sobreestimación, mientras que Q-learning con 10 intervalos obtuvo
359.70 ± 102.57. Se conservan estos resultados negativos; no son una comparación
universal entre algoritmos. Las 17 pruebas y la reproducción exacta de las
evaluaciones finales desde los modelos guardados pasaron.

- Al menos 3 semillas independientes por configuración (5 recomendadas en Tier 1).
- Evaluaciones periódicas de 10 episodios greedy, separadas del entrenamiento.
- Evaluación sobre 100 episodios para contrastar los umbrales del enunciado.
- Curvas en pasos del entorno, con media y banda de variabilidad.
- Igual presupuesto de entrenamiento para comparar Q-learning y DQN en Tier 1.
- Registrar configuración, tiempo, hardware, logs y pesos; generar videos/GIFs.
- No usar agentes prefabricados como implementación entregada.

## Fuentes de la primera etapa

- [CartPole — documentación oficial](https://gymnasium.farama.org/environments/classic_control/cart_pole/)
- [Uso básico de Gymnasium](https://gymnasium.farama.org/introduction/basic_usage/)

El criterio del proyecto es ≥475 de retorno medio en 100 episodios.
El script imprime también el umbral registrado por la versión instalada;
son datos diferentes y conservamos el criterio del docente para el proyecto.
