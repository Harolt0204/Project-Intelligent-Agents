# Registro de uso de IA

Herramienta: Codex (OpenAI).

## Etapa 1 — Preparación y comprensión de CartPole


- Verificación: consultar las fuentes oficiales de Gymnasium y ejecutar el
  ejemplo local. Los resultados de esta demostración no son entrenamientos.


Actualizar este registro en cada etapa; servirá para la declaración y el apéndice
de prompts del artículo final. No sustituye la revisión de los integrantes.

## Etapa 2 — Discretización

- Asistencia: discretizador propio, demostración de tablas con 6, 10 y 20 bins,
  explicación de pérdida de detalle y crecimiento del número de casillas.
- Las velocidades utilizan rangos provisionales de diseño, pendientes de
  justificar mediante cobertura observada en entrenamientos.


## Etapa 3 — Actualización de Q-learning


- Asistencia: agente tabular propio con política epsilon-greedy, desempate
  aleatorio y actualización Q-learning; ejemplo inventado y un episodio real.
- Comprobación: pruebas del ajuste numérico, final de tarea frente a corte por
  tiempo, selección de acciones, empates y reproducibilidad por semilla.


## Etapa 4 — Entrenamiento piloto

- Asistencia: bucle de entrenamiento con tabla persistente, epsilon decreciente,
  evaluación independiente sin aprendizaje y almacenamiento de logs y tablas.
- Configuración piloto: 10 bins, 30 000 pasos y 3 semillas; no constituye la
  comparación formal de Tier 1 ni una selección final de hiperparámetros.
- Verificación: ejecución del piloto y prueba de que evaluar no modifica la
  tabla ni el generador aleatorio del agente de entrenamiento.
- Los resultados reales se consultan en `results/parte_04_tabular`; no se
  presupone que alcance el umbral ni que cada checkpoint mejore al anterior.

## Etapa 5 — Demostración visual


- Asistencia: cargar la tabla existente y generar GIF comparando tabla vacía y
  entrenada desde el mismo inicio. No se reentrena ni se selecciona el mejor episodio.
- Evidencia: GIF y registro del intento en `results/parte_05_visual`.
- El GIF es una ilustración de un intento; no reemplaza los promedios obtenidos
  sobre 100 episodios y tres semillas.

## Etapa 6 — Comparación de discretizaciones

- Asistencia: protocolo fijo de 6, 10 y 20 bins con 200 000 pasos y tres semillas
  por configuración; agregación y figuras con bandas de desviación estándar.
- Se conserva el resto de parámetros para estudiar solo la granularidad.
- Evidencia verificable: protocolo, configuraciones, CSV, tablas y figuras en
  `results/parte_06_bins`. Las interpretaciones deben contrastarse con estos datos.

## Etapa 7 — Red Q inicial

- Asistencia: instalación de PyTorch CPU en el entorno del proyecto, arquitectura
  propia 4-64-64-2, demostración sin entrenamiento y pruebas de dimensiones,
  cálculo de gradientes y conservación de pesos durante consultas.
- Evidencia: código en `agents/dqn.py`, ejemplo en `experiments/explorar_red_q.py`
  y resultado marcado sin entrenar en `results/parte_07_red_q`.
- No se ha completado DQN ni se presentan puntuaciones iniciales como resultados
  de aprendizaje. Los entrenamientos tabulares anteriores se conservan.

## Etapa 8 — Memoria de experiencias

- Asistencia: replay buffer propio circular, muestreo uniforme sin reemplazo
  dentro de cada lote y demostración con experiencias reales de CartPole.
- Verificación: reemplazo de recuerdos antiguos, copias de observaciones,
  muestreo reproducible, ausencia de eliminación al muestrear y conservación
  de los indicadores de fin de episodio.
- Evidencia: `results/parte_08_memoria/demostracion.json`. No es un entrenamiento;
  la capacidad y el lote pequeños son exclusivamente didácticos.

## Etapa 9 — Red objetivo y actualización DQN

- Asistencia: agente DQN propio, copia independiente de la red objetivo, política
  epsilon-greedy, pérdida Smooth L1 y ajuste Adam de la red online.
- Verificación: objetivos conocidos para final real frente a límite temporal,
  cambios solo en online, ausencia de gradientes en target y sincronización.
- La demostración utiliza ocho experiencias de una colección de 64 y un ajuste.
  Su error de lote no se presenta como evidencia de éxito en CartPole.
- Evidencia en `results/parte_09_red_objetivo/demostracion.json`. El entrenamiento
  completo y la comparación con Q-learning quedan para la siguiente etapa.

## Etapa 10 — Entrenamiento completo DQN en CartPole

- Asistencia: bucle reproducible con replay, red objetivo, exploración decreciente,
  evaluaciones independientes y comparación con todos los resultados tabulares.
- Diseño: 200 000 pasos y semillas 42/43/44; parámetros fijados antes de entrenar,
  sin elegir el mejor checkpoint. Se verifica con 100 episodios cualquier señal
  intermedia de éxito detectada en una evaluación de 10 episodios.
- Evidencia verificable: configuraciones, logs, pérdidas, pesos, curvas y GIF en
  `results/parte_10_dqn`. Las afirmaciones deben basarse en los archivos reales.
- Verificación: pruebas del aislamiento de evaluación y comprobaciones de pasos,
  objetivos de aprendizaje, modelos guardados y reproducción de resultados.

## Etapa 11 — Reducir únicamente la tasa de aprendizaje

- Hipótesis fijada antes de ejecutar: bajar learning rate de 0.0005 a 0.0001
  podría reducir sobreestimación e inestabilidad. No se presupone éxito.
- Se mantienen semillas 42/43/44, 200 000 pasos y todos los demás parámetros.
- Asistencia: parametrización de la tasa, nuevas ejecuciones separadas y figura
  comparativa con verificación automática de diferencias de configuración.
- Evidencia en `results/parte_11_lr`; los resultados previos no se sustituyen.
- Esta etapa no equivale al estudio completo de hiperparámetros de Tier 2.

## Etapa 12 — Copia de la red objetivo, prueba corta


- Hipótesis: espaciar la copia de 500 a 2 000 pasos podría mejorar estabilidad.
- Referencia declarada: tasa original 0.0005. Ambas opciones usan 50 000 pasos,
  tres semillas y el calendario original de exploración de 160 000 pasos.
- Asistencia: parametrización del intervalo, diagnóstico Q durante el proceso,
  repetición controlada de la referencia y gráficos de rendimiento y estimaciones.
- Verificación: pruebas del código, diferencias de configuraciones y reproducción
  de retornos de los controles iniciales del experimento original.
- Evidencia en `results/parte_12_target_corto`. Es una prueba corta, no evidencia
  de éxito o estabilidad a largo plazo ni sustitución de los experimentos previos.

## Etapa 13 — Auditoría contra fuentes y cálculos independientes

- Asistencia: lectura de fuentes primarias CleanRL/Gymnasium, comparación de
  componentes y configuración, auditoría de transiciones y cálculos NumPy y
  prueba del agente en un problema con valores óptimos conocidos.
- Evidencia: `results/parte_13_auditoria/audit.json` y `report/auditoria_dqn.md`.
- Las comprobaciones pasaron. No se identificó un error en esos casos ni se
  afirma que esté demostrada la corrección total o estabilidad en CartPole.
- No se ejecutó un agente externo, no se añadió dependencia y no se cambió la
  implementación de aprendizaje en esta etapa. Se conservan los resultados previos.

## Etapa 14 — Comparación de funciones de pérdida

- Hipótesis previa: MSE podría mejorar el resultado al ponderar de otra manera
  los errores. No se presupone mejora ni estabilidad posterior.
- Asistencia: selección de pérdida parametrizable conservando Smooth L1 por
  defecto, prueba numérica MSE y tres entrenamientos nuevos de 50 000 pasos.
- Referencia: los tres controles Smooth L1 ya verificados de la etapa 12,
  con igual presupuesto y calendario de exploración. Solo cambia la pérdida.
- Evidencia en `results/parte_14_loss`. Los valores de pérdida de ambas funciones
  no se interpretan como métricas numéricas directamente comparables.

## Etapa 15 — Entrenamiento MSE completo y cierre de Tier 1

- Asistencia: entrenamiento MSE completo con 200 000 pasos y tres semillas;
  comparación contra los tres tabulares y el DQN inicial, sin ocultar fallos.
- No se combina el cambio de target ni se selecciona el mejor checkpoint.
- Evaluación adicional declarada antes de ver sus resultados: 100 episodios
  por modelo con semillas nuevas `1010000 + 1000*seed + episodio`; sin ajustes
  posteriores de estos modelos. Se conserva el historial de selección exploratoria.
- Evidencia en `results/parte_15_mse_completo`, incluyendo diagnósticos, modelos,
  curvas y evaluación adicional. El cierre experimental no equivale a tener
  terminado el paper, Tier 2, Tier 3 ni la defensa.

## Preparación para compartir Tier 1 en GitHub

- Asistencia: preparación del repositorio, guía de reproducción, instalador
  Windows y comprobación de dependencias/modelos sin entrenamiento.
- Se incluyen pesos, tablas, registros y resultados negativos; se excluye el
  entorno virtual local y archivos temporales de Python.
- Validación previa: copia independiente del contenido preparado, compilación
  de módulos, 18 pruebas y carga de ambos agentes reproduciendo su intento
  guardado. Se usó el entorno Python ya instalado para esta comprobación.
- La instalación documentada está comprobada para Windows con Python 3.12;
  no se presenta una validación nueva en Linux o macOS.

## Notebook y mejora de Tier 1

- Asistencia: notebook educativo que utiliza los módulos originales, parámetros expuestos, instalación y ejecución de celdas mediante un kernel real. No se duplican los algoritmos dentro del cuaderno.
- Optimización DQN inspirada en la configuración pública de CleanRL (red 120/84, lr 0.00025, memoria 10000, lote 128, inicio 10000 y ajuste cada 10 pasos); se conserva el agente propio, 200000 pasos y evaluación final sin selección del mejor checkpoint.
- También se investigan rangos de discretización y exploración tabular. Los cambios conjuntos no se presentan como una ablación causal.
- Se conservarán todas las semillas y los intentos anteriores, con configuración y resultados por ejecución. Las nuevas condiciones de evaluación se fijan en el protocolo antes de medirlas.
- Referencias: https://docs.cleanrl.dev/rl-algorithms/dqn/ y https://gymnasium.farama.org/environments/classic_control/cart_pole/ .

- Resultado de la mejora: DQN 498.60±1.98 (3/3 semillas ≥475), evaluación adicional 499.00±1.41. Mejor tabular actual: 6 intervalos, 414.48±62.85; las otras granularidades empeoraron y no se ocultaron.
- Verificación: 22 pruebas unitarias, ejecución del notebook con kernel real, animación HTML y dos pruebas cortas de 1000 pasos que comprobaron parámetros editables y protección contra sobrescritura. Los retornos finales de los 12 modelos se reprodujeron exactamente y se evaluaron otros 1200 episodios con el protocolo fijado.

## Búsqueda para que Q-learning alcance el umbral

- Asistencia: ensayos de inicialización optimista, espaciado de intervalos y alpha decreciente; diagnóstico de causas de terminación; prueba de resolución distinta por variable. Se mantienen las recompensas originales y la actualización Q-learning.
- Las tres granularidades por dimensión del enunciado ya se estudiaron en las etapas anteriores. La búsqueda adicional prueba (2,2,6,6), (2,2,10,10), (2,2,20,20), preservando las cuatro observaciones y dedicando más detalle a las variables angulares.
- Cada resultado agregado usa semillas 42/43/44 y 200000 pasos por ejecución. Se conservan los resultados negativos; no se sustituyen los modelos finales por sus mejores controles intermedios.
- Evidencia en results/parte_18_qlearning; los diagnósticos reproducen las evaluaciones previas y documentan fallos por inclinación de la vara. Eso motiva el diseño, sin probar una causa única.

- Resultado final de la búsqueda: Q-learning 2/2/6/6 con Q inicial=100 y alpha 0.1→0.01 obtuvo 487.19, 500 y 500 (495.73±6.04). Las variantes angulares 10 y 20 no alcanzaron el objetivo y permanecen en el repositorio.
- La muestra fijada con base 5000000 se ejecutó tras fijar la configuración: 489.41, 500 y 500 (496.47±4.99), sin ajustes posteriores. También se reevaluó el DQN previamente guardado, sin reentrenarlo. Los 12 modelos finales reprodujeron exactamente sus 100 retornos de desarrollo.
- El notebook y el menú usan la resolución guardada por dimensión. Las pruebas cubren inicialización, formas de tabla y fronteras; la variante no uniforme quedó como resultado negativo, no como configuración seleccionada.

- Validación de esta actualización: 25 pruebas unitarias pasaron; se ejecutaron las 18 celdas del notebook y la animación tabular con kernel real. Una prueba de 1000 pasos desde los parámetros del notebook comprobó forma 2/2/6/6, alpha final, carga/evaluación reproducible y protección contra sobrescritura; no es un resultado formal.

## Tier 2 y Tier 3 — LunarLander y Double DQN

- Asistencia: instalación de Box2D en Python 3.12, verificación física, entrenamiento/carga de prueba y evaluación reproducida; DQN general de ocho entradas/cuatro acciones, ablación sin target y Double DQN propio.
- Diseño: dos estudios de tres valores (learning_rate y target_sync_every), cambiando un parámetro a la vez; base reutilizada para ablación y Tier 3. Siete grupos únicos con tres semillas.


## Mejora solicitada para las tres semillas

- Objetivo: DQN y Double DQN ≥200 individualmente en 42/43/44, no solo la media agregada. Los estudios anteriores y sus resultados se conservan como historial.
- Diagnóstico: regresiones tras controles exitosos y políticas que no terminan el aterrizaje; Double 43 agotó el límite de tiempo en 100/100 episodios. Se propone más optimización, tasa decreciente y selección común por validación, en vez de elegir semillas distintas.




