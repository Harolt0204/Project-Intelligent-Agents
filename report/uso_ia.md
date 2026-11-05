# Registro de uso de IA

Herramienta: Codex (OpenAI).

## Etapa 1 — Preparación y comprensión de CartPole

- Tareas: comparación del enunciado con la guía local, organización del alcance
  para dos integrantes, preparación del entorno y script didáctico con acciones
  aleatorias. Redacción de explicaciones y comandos de reproducción.
- Instrucción importante del usuario: «mejor trabajemos con esta computadora
  nomás. relee las instrucciones y pongamosnos manos a la obra parte por parte
  para ir entendiendo lo que etsamos haciendo».
- Verificación: consultar las fuentes oficiales de Gymnasium y ejecutar el
  ejemplo local. Los resultados de esta demostración no son entrenamientos.
- Pendiente para los integrantes: revisar el código y explicar observaciones,
  acciones, recompensa, retorno, semilla y las dos causas de final de episodio.

Actualizar este registro en cada etapa; servirá para la declaración y el apéndice
de prompts del artículo final. No sustituye la revisión de los integrantes.

## Etapa 2 — Discretización

- Instrucción del usuario: «entendido, continuemos», manteniendo el trabajo por
  partes para comprender cada componente.
- Asistencia: discretizador propio, demostración de tablas con 6, 10 y 20 bins,
  explicación de pérdida de detalle y crecimiento del número de casillas.
- Las velocidades utilizan rangos provisionales de diseño, pendientes de
  justificar mediante cobertura observada en entrenamientos.
- Esta etapa no incluye entrenamiento ni comparaciones de rendimiento.

## Etapa 3 — Actualización de Q-learning

- Instrucción del usuario: «dale, continuemos», tras explicar agrupación de
  estados, valores Q, exploración y el ejemplo numérico de actualización.
- Asistencia: agente tabular propio con política epsilon-greedy, desempate
  aleatorio y actualización Q-learning; ejemplo inventado y un episodio real.
- Comprobación: pruebas del ajuste numérico, final de tarea frente a corte por
  tiempo, selección de acciones, empates y reproducibilidad por semilla.
- No se ha completado entrenamiento formal ni demostrado mejora del agente.

## Etapa 4 — Entrenamiento piloto

- Instrucción del usuario: «listo continuemos».
- Asistencia: bucle de entrenamiento con tabla persistente, epsilon decreciente,
  evaluación independiente sin aprendizaje y almacenamiento de logs y tablas.
- Configuración piloto: 10 bins, 30 000 pasos y 3 semillas; no constituye la
  comparación formal de Tier 1 ni una selección final de hiperparámetros.
- Verificación: ejecución del piloto y prueba de que evaluar no modifica la
  tabla ni el generador aleatorio del agente de entrenamiento.
- Los resultados reales se consultan en `results/parte_04_tabular`; no se
  presupone que alcance el umbral ni que cada checkpoint mejore al anterior.

## Etapa 5 — Demostración visual

- Instrucción del usuario: «dale, continuemos» después del entrenamiento piloto.
- Asistencia: cargar la tabla existente y generar GIF comparando tabla vacía y
  entrenada desde el mismo inicio. No se reentrena ni se selecciona el mejor episodio.
- Evidencia: GIF y registro del intento en `results/parte_05_visual`.
- El GIF es una ilustración de un intento; no reemplaza los promedios obtenidos
  sobre 100 episodios y tres semillas.

## Etapa 6 — Comparación de discretizaciones

- Instrucción del usuario: «dale, continuemos» tras la demostración visual.
- Asistencia: protocolo fijo de 6, 10 y 20 bins con 200 000 pasos y tres semillas
  por configuración; agregación y figuras con bandas de desviación estándar.
- Se conserva el resto de parámetros para estudiar solo la granularidad.
- Evidencia verificable: protocolo, configuraciones, CSV, tablas y figuras en
  `results/parte_06_bins`. Las interpretaciones deben contrastarse con estos datos.

## Etapa 7 — Red Q inicial

- Instrucción del usuario: «continua», después de explicar la diferencia entre
  tabla Q y red neuronal.
- Asistencia: instalación de PyTorch CPU en el entorno del proyecto, arquitectura
  propia 4-64-64-2, demostración sin entrenamiento y pruebas de dimensiones,
  cálculo de gradientes y conservación de pesos durante consultas.
- Evidencia: código en `agents/dqn.py`, ejemplo en `experiments/explorar_red_q.py`
  y resultado marcado sin entrenar en `results/parte_07_red_q`.
- No se ha completado DQN ni se presentan puntuaciones iniciales como resultados
  de aprendizaje. Los entrenamientos tabulares anteriores se conservan.

## Etapa 8 — Memoria de experiencias

- Instrucción del usuario: «dale, sigamos» tras construir la primera red Q.
- Asistencia: replay buffer propio circular, muestreo uniforme sin reemplazo
  dentro de cada lote y demostración con experiencias reales de CartPole.
- Verificación: reemplazo de recuerdos antiguos, copias de observaciones,
  muestreo reproducible, ausencia de eliminación al muestrear y conservación
  de los indicadores de fin de episodio.
- Evidencia: `results/parte_08_memoria/demostracion.json`. No es un entrenamiento;
  la capacidad y el lote pequeños son exclusivamente didácticos.

## Etapa 9 — Red objetivo y actualización DQN

- Instrucción del usuario: «continua», después de implementar replay buffer.
- Asistencia: agente DQN propio, copia independiente de la red objetivo, política
  epsilon-greedy, pérdida Smooth L1 y ajuste Adam de la red online.
- Verificación: objetivos conocidos para final real frente a límite temporal,
  cambios solo en online, ausencia de gradientes en target y sincronización.
- La demostración utiliza ocho experiencias de una colección de 64 y un ajuste.
  Su error de lote no se presenta como evidencia de éxito en CartPole.
- Evidencia en `results/parte_09_red_objetivo/demostracion.json`. El entrenamiento
  completo y la comparación con Q-learning quedan para la siguiente etapa.

## Etapa 10 — Entrenamiento completo DQN en CartPole

- Instrucción del usuario: «listo hagamoslo» tras integrar las piezas de DQN.
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

- Instrucción del usuario: «dale, continua» tras proponer un cambio aislado.
- Hipótesis fijada antes de ejecutar: bajar learning rate de 0.0005 a 0.0001
  podría reducir sobreestimación e inestabilidad. No se presupone éxito.
- Se mantienen semillas 42/43/44, 200 000 pasos y todos los demás parámetros.
- Asistencia: parametrización de la tasa, nuevas ejecuciones separadas y figura
  comparativa con verificación automática de diferencias de configuración.
- Evidencia en `results/parte_11_lr`; los resultados previos no se sustituyen.
- Esta etapa no equivale al estudio completo de hiperparámetros de Tier 2.

## Etapa 12 — Copia de la red objetivo, prueba corta

- Instrucción del usuario: «dale continua» después de proponer revisar target.
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

- Instrucción del usuario: «dale, sigue», después de proponer revisión del agente.
- Asistencia: lectura de fuentes primarias CleanRL/Gymnasium, comparación de
  componentes y configuración, auditoría de transiciones y cálculos NumPy y
  prueba del agente en un problema con valores óptimos conocidos.
- Evidencia: `results/parte_13_auditoria/audit.json` y `report/auditoria_dqn.md`.
- Las comprobaciones pasaron. No se identificó un error en esos casos ni se
  afirma que esté demostrada la corrección total o estabilidad en CartPole.
- No se ejecutó un agente externo, no se añadió dependencia y no se cambió la
  implementación de aprendizaje en esta etapa. Se conservan los resultados previos.

## Etapa 14 — Comparación de funciones de pérdida

- Instrucción del usuario: «continua con el siguiente paso» después de la auditoría.
- Hipótesis previa: MSE podría mejorar el resultado al ponderar de otra manera
  los errores. No se presupone mejora ni estabilidad posterior.
- Asistencia: selección de pérdida parametrizable conservando Smooth L1 por
  defecto, prueba numérica MSE y tres entrenamientos nuevos de 50 000 pasos.
- Referencia: los tres controles Smooth L1 ya verificados de la etapa 12,
  con igual presupuesto y calendario de exploración. Solo cambia la pérdida.
- Evidencia en `results/parte_14_loss`. Los valores de pérdida de ambas funciones
  no se interpretan como métricas numéricas directamente comparables.

## Etapa 15 — Entrenamiento MSE completo y cierre de Tier 1

- Instrucción del usuario: «dale, entonces continua», tras explicar el estado
  del proyecto y la necesidad de consolidar la comparación.
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

- Instrucción del usuario: publicar en `Harolt0204/Project-Intelligent-Agents`
  para que su compañero pueda reproducir y revisar el Tier 1 completo.
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

- Solicitud del usuario: «dale, hazlo y mejoremos los resultados para cumplir con las instrucciones del proyecto», tras valorar un notebook central para explicar código y editar parámetros.
- Asistencia: notebook educativo que utiliza los módulos originales, parámetros expuestos, instalación y ejecución de celdas mediante un kernel real. No se duplican los algoritmos dentro del cuaderno.
- Optimización DQN inspirada en la configuración pública de CleanRL (red 120/84, lr 0.00025, memoria 10000, lote 128, inicio 10000 y ajuste cada 10 pasos); se conserva el agente propio, 200000 pasos y evaluación final sin selección del mejor checkpoint.
- También se investigan rangos de discretización y exploración tabular. Los cambios conjuntos no se presentan como una ablación causal.
- Se conservarán todas las semillas y los intentos anteriores, con configuración y resultados por ejecución. Las nuevas condiciones de evaluación se fijan en el protocolo antes de medirlas.
- Referencias: https://docs.cleanrl.dev/rl-algorithms/dqn/ y https://gymnasium.farama.org/environments/classic_control/cart_pole/ .

- Resultado de la mejora: DQN 498.60±1.98 (3/3 semillas ≥475), evaluación adicional 499.00±1.41. Mejor tabular actual: 6 intervalos, 414.48±62.85; las otras granularidades empeoraron y no se ocultaron.
- Verificación: 22 pruebas unitarias, ejecución del notebook con kernel real, animación HTML y dos pruebas cortas de 1000 pasos que comprobaron parámetros editables y protección contra sobrescritura. Los retornos finales de los 12 modelos se reprodujeron exactamente y se evaluaron otros 1200 episodios con el protocolo fijado.

## Búsqueda para que Q-learning alcance el umbral

- Solicitud del usuario: «solo Q-learning no llega al objetivo verdad?, hay como ir mejorando algo para que llegue al objetivo?»
- Asistencia: ensayos de inicialización optimista, espaciado de intervalos y alpha decreciente; diagnóstico de causas de terminación; prueba de resolución distinta por variable. Se mantienen las recompensas originales y la actualización Q-learning.
- Las tres granularidades por dimensión del enunciado ya se estudiaron en las etapas anteriores. La búsqueda adicional prueba (2,2,6,6), (2,2,10,10), (2,2,20,20), preservando las cuatro observaciones y dedicando más detalle a las variables angulares.
- Cada resultado agregado usa semillas 42/43/44 y 200000 pasos por ejecución. Se conservan los resultados negativos; no se sustituyen los modelos finales por sus mejores controles intermedios.
- Evidencia en results/parte_18_qlearning; los diagnósticos reproducen las evaluaciones previas y documentan fallos por inclinación de la vara. Eso motiva el diseño, sin probar una causa única.

- Resultado final de la búsqueda: Q-learning 2/2/6/6 con Q inicial=100 y alpha 0.1→0.01 obtuvo 487.19, 500 y 500 (495.73±6.04). Las variantes angulares 10 y 20 no alcanzaron el objetivo y permanecen en el repositorio.
- La muestra fijada con base 5000000 se ejecutó tras fijar la configuración: 489.41, 500 y 500 (496.47±4.99), sin ajustes posteriores. También se reevaluó el DQN previamente guardado, sin reentrenarlo. Los 12 modelos finales reprodujeron exactamente sus 100 retornos de desarrollo.
- El notebook y el menú usan la resolución guardada por dimensión. Las pruebas cubren inicialización, formas de tabla y fronteras; la variante no uniforme quedó como resultado negativo, no como configuración seleccionada.

- Validación de esta actualización: 25 pruebas unitarias pasaron; se ejecutaron las 18 celdas del notebook y la animación tabular con kernel real. Una prueba de 1000 pasos desde los parámetros del notebook comprobó forma 2/2/6/6, alpha final, carga/evaluación reproducible y protección contra sobrescritura; no es un resultado formal.

## Tier 2 y Tier 3 — LunarLander y Double DQN

- Solicitud: «Hagámos el Tier 2 y Tier 3 ... ipynb separados ... o el mismo ... Tier 1». Se crean notebooks independientes que comparten algoritmos. Se conserva el notebook de Tier 1 modificado por el usuario.
- Asistencia: instalación de Box2D en Python 3.12, verificación física, entrenamiento/carga de prueba y evaluación reproducida; DQN general de ocho entradas/cuatro acciones, ablación sin target y Double DQN propio.
- Diseño: dos estudios de tres valores (learning_rate y target_sync_every), cambiando un parámetro a la vez; base reutilizada para ablación y Tier 3. Siete grupos únicos con tres semillas.
- Antes de fijar el estudio se midió coste de configuración de referencia (red 256/256 y cuatro ajustes cada cuatro pasos) y red 128/128 con ajustes cada cuatro pasos. Las ejecuciones parciales se conservan como perfiles de coste, no como resultados formales.
- Protocolo formal fijado: red 128/128, lote 64, replay 100000, gamma 0.99, Smooth L1, Adam 0.0005, inicio 5000, un ajuste cada ocho pasos, target cada 2000, epsilon de 1 a 0.05 en 150000 pasos; 500000 pasos por ejecución. Se estudiarán lr=[0.0001,0.0005,0.001] y target=[500,2000,8000].
- Semillas en procesos independientes para reducir tiempo de espera; un hilo por proceso. Los tiempos no son benchmarks aislados. No se modifican recompensas ni se selecciona el mejor checkpoint.
- Fuentes primarias consultadas: documentación Gymnasium de LunarLander, configuración pública DQN de RL Baselines3 Zoo y artículo Double DQN (van Hasselt, Guez y Silver). Solo se usan como referencias, no como agentes entregados.
- Pruebas numéricas distinguen los objetivos DQN y Double; la ablación no conserva una target independiente. También se verifica que los estudios cambien un solo parámetro.
- Verificación inicial: 28 pruebas unitarias pasaron y ambos notebooks se ejecutaron con kernel real y entrenamiento desactivado. Una prueba corta de LunarLander reprodujo exactamente los retornos al cargar el modelo guardado.
- Una interrupción del primer grupo de tasa baja fue causada por la codificación de texto de la consola de Windows, no por los valores de aprendizaje. Se conserva el intento parcial en results/tier23_interrupted_encoding, marcado como no formal; se corrigió la consola UTF-8 y se repitió únicamente ese grupo desde cero con las mismas semillas. La base completa se reutilizó.
- Errata documental: selection en el protocolo congelado menciona 200000 por arrastre de un piloto. El presupuesto numérico steps=500000 gobierna todas las ejecuciones formales. Se conserva el protocolo histórico y la aclaración aparece en el notebook, guía y discusión; no se alteran las huellas de resultados ya entrenados.
- La implementación de entrenamiento quedó registrada en el commit 460af7c. Sus cinco archivos principales coinciden con las huellas de código guardadas por los 21 modelos. Tras iniciar los 21 modelos se generalizó únicamente un comentario de DQN que decía «dos salidas», para que también describa las cuatro de LunarLander; el cálculo no cambia. La comprobación está en results/tier23_lunarlander/audit.json.
- Se redujo la memoria al exportar GIF: los fotogramas con título se conservan en paleta y se liberan las matrices RGB antes de codificar cuando no se necesita animación inline. Se mantienen todos los fotogramas, el recorrido y sus retornos.
- Cierre: 21 modelos × 500000 pasos, sin reemplazar modelos finales por controles intermedios. Medias: base 194.77, lr=0.0001 49.85, lr=0.001 226.04, target=500 166.42, target=8000 149.78, sin target 161.76 y Double 122.85. La mejor media de Tier 2 supera 200 con dos de tres semillas; no se afirma éxito uniforme.
- Double no mejoró la base: diferencias por semilla +61.31, -169.33 y -107.73. La evaluación nueva mantuvo el patrón: base 203.84, Double 124.13 y la variante lr=0.001 228.22. Esta variante se eligió después de observar desarrollo; las condiciones iniciales nuevas ya estaban fijadas y no se usaron para reajustar.
- Validación final: 28 pruebas unitarias; protocolos y contadores revisados; los 21 modelos guardados reprodujeron exactamente sus 2100 retornos finales. Se realizaron 900 episodios con nuevos inicios. Ambos notebooks se ejecutaron sin errores (17 y 12 celdas de código), además de las dos animaciones. Los GIF corresponden al primer episodio documentado y sus retornos coinciden con la evaluación guardada.
- Se revisaron visualmente las cuatro gráficas y las vistas previas de ambos GIF. El notebook de Tier 1 modificado por el usuario conserva su hash original y no se incluye en estos commits. El análisis teórico, el paper y la preparación de la defensa continúan pendientes.

## Mejora solicitada para las tres semillas

- Prompt: «podemos mejorar? para que cumpla en las tres semillas y el tier 3 igual mejorar porfa, para que no tengamos los rsultados negativos, y ahorrarnos un poco de explicacion».
- Objetivo: DQN y Double DQN ≥200 individualmente en 42/43/44, no solo la media agregada. Los estudios anteriores y sus resultados se conservan como historial.
- Diagnóstico: regresiones tras controles exitosos y políticas que no terminan el aterrizaje; Double 43 agotó el límite de tiempo en 100/100 episodios. Se propone más optimización, tasa decreciente y selección común por validación, en vez de elegir semillas distintas.
- Configuración congelada antes de entrenar: configs/tier23_mejora_v1.json; código en commit 8a5d309. Un millón de pasos, lote128, ajuste cada4, target1000, lr0.00063 hasta300000 y decaimiento a0.00005 hasta800000, epsilon1→0.05 en200000. Misma configuración y presupuesto en ambos métodos, desde cero.
- Validación con base30000000; 10 episodios cada50000 pasos, ampliados a100 cuando alcanzan200 y siempre al final. Se elige la mayor media confirmada, manteniendo aparte los pesos del último paso. Se exige margen de230 en validación antes de abrir las dos pruebas nuevas, bases40000000 y50000000.
- Las pruebas se abren después de fijar las huellas de los seis modelos. Si la validación no basta, la ronda se detiene en desarrollo sin consultar prueba. Se documentan también los intentos que no cumplan; no se eliminan ni se presentan como éxitos.
- Comprobación previa: 29 pruebas unitarias pasaron. Una integración real corta de2000 pasos por agente verificó calendario,469 ajustes,8 copias de target, reproducción del modelo seleccionado y separación entre entrenamiento y pruebas. Sus episodios usaron semilla123 y no se solapan con las muestras finales de42/43/44; no son resultados formales.
- Los notebooks mantienen una sola entrada por Tier y añaden la nueva ronda; no se modifica el notebook de Tier1 del usuario. Se actualizarán los modelos de presentación después de comprobar el objetivo, manteniendo accesibles los anteriores.
- Resultado: DQN 272.70,277.61,267.25; Double 278.71,282.33,271.08, cada uno en100 episodios nuevos por semilla. La confirmación independiente dio DQN269.08,279.00,273.00 y Double274.18,278.67,281.35. Ambos alcanzan3/3 en ambas muestras, sin combinar las dos muestras para ocultar una media insuficiente.
- Medias entre semillas: DQN272.52±4.23 y Double277.37±4.69; en confirmación273.69±4.08 y278.07±2.96. La diferencia media de prueba de Double fue+4.85. Es evidencia de estas tres repeticiones, no una superioridad universal.
- Auditoría de los seis modelos: presupuesto1000000,248751 ajustes y1000 copias de target por modelo, protocolo emparejado, políticas iniciales coincidentes, fuentes iguales al commit8a5d309, pesos congelados antes de pruebas. Se reprodujeron600 retornos de validación; se ejecutaron1200 episodios de prueba desde los modelos cargados y se repitieron exactamente12 episodios iniciales durante la auditoría.
- La presentación usa ahora los checkpoints seleccionados por validación mediante results/tier23_actual.json. Los pesos de la última actualización se conservan aparte, aunque su desempeño pueda ser inferior. Las demostraciones muestran el primer episodio de la prueba con semilla42 y no se elige un episodio favorable.
- Cierre de presentación: ambos notebooks ejecutados con kernel real sin errores (23 y18 celdas de código), ambas animaciones inline verificadas y cuatro figuras revisadas visualmente. Los GIF tienen114 y115 fotogramas y reproducen exactamente el primer retorno de prueba:247.73 DQN y241.74 Double. Evidencia conjunta en results/tier23_mejora_v1/workflow_complete.json. El notebook de Tier1 conserva su huella original y queda fuera de estos commits.


## Análisis teórico e intervalos, sin reentrenamiento

- Solicitud: «completa el analisis teorico y añade esos intervalos de confianza ... significa volver a entrenar?».
- Se aclaró al releer el apartado 6.2 que DE o IC son opciones válidas; las bandas anteriores ya cumplían. Se añadieron IC a petición del equipo sin entrenar agentes ni generar episodios nuevos.
- Análisis obligatorio: BipedalWalker-v3, Hopper-v5, HalfCheetah-v5, Ant-v5, crecimiento k^d y b^m, memoria de tabla, PPO/SAC/TD3 y píxeles Atari. Dimensiones contrastadas con fuentes oficiales y archivos de Gymnasium 1.4.0 instalado; no se instaló MuJoCo ni se simularon resultados de esos entornos. Las referencias primarias enlazadas se comprobaron y los cálculos combinatorios son propios.
- IC t del 95% puntuales, n=3 medias de semillas, df=2, desviación muestral ddof=1. Las tablas descriptivas con DE ddof=0 se conservan. Se documentan normalidad no verificable con n pequeño, condicionamiento a selección, ausencia de cobertura simultánea y extremos fuera de límites físicos sin recorte. Los episodios no se tratan como entrenamientos independientes.
- Se añadieron diferencias emparejadas Double-DQN por semilla: prueba +4.85, IC [2.14,7.56]; confirmación +4.38, IC [-6.51,15.27]. No se afirma superioridad universal. Los puntajes originales no cambian.
- Se conservaron byte a byte los tres notebooks que ya tenían modificaciones del usuario. Un cuaderno complementario reúne el análisis, el cálculo y ocho figuras. El primer control detectó que faltaba incluir la comparación inicial Double DQN; se añadió y el control completo pasó.
- Validación: 31 pruebas unitarias; diez registros de IC comprobados mediante cálculo independiente; ocho celdas de código del nuevo cuaderno ejecutadas con kernel real sin errores y ocho figuras incrustadas; PDF de dos páginas renderizado y revisado visualmente, diez enlaces; comprobadas las huellas de 300 archivos protegidos, incluidos pesos, tablas, registros y los tres notebooks originales. No se cambió el código de entrenamiento.
- Entregables: report/analisis_teorico.md, output/pdf/analisis_teorico.pdf, report/intervalos_confianza.md y Analisis_Teorico_e_Intervalos.ipynb. El artículo completo de 6–8 páginas y la defensa oral siguen pendientes.


## Organización del repositorio para revisión académica

- Solicitud del equipo: retirar de GitHub lanzadores .bat y guías personales o redundantes antes de la revisión docente.
- Se centralizaron instalación, dependencias, configuración, comandos por experimento, figuras, evaluación y modelos en el README. El historial experimental previo permanece en report/historial_experimentos.md, identificado como histórico.
- Se retiraron del seguimiento cinco lanzadores y cuatro guías, conservando sus archivos locales y un respaldo ignorado por Git. Se mantienen el enunciado, los informes científicos, los resultados negativos, la ablación y esta declaración de uso de IA.
- Se corrigieron las referencias a auxiliares retirados. En los notebooks públicos Tier2 y Tier3 solo se modificaron dos líneas Markdown de instalación; las ediciones locales de los tres notebooks se preservaron sin publicarlas. No se modificaron agentes, configuraciones, pesos ni registros experimentales.
- Verificación de la reorganización: 24 enlaces relativos válidos en los documentos revisados; nueve auxiliares retirados del índice y conservados localmente; huellas de los tres notebooks locales sin cambios; celdas de código y salidas de los dos notebooks públicos idénticas a las anteriores. El verificador de instalación pasó y reprodujo exactamente los episodios guardados de Q-learning y DQN sin entrenar. No se modificaron resultados experimentales.
