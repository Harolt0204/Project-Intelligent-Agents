# Project 01 — Reinforcement Learning and DQN on Gymnasium Environments

> **Asignatura:** Agentes Inteligentes — Yachay Tech University  
> **Fuente oficial:** [Enunciado de Project 01 en EVA](https://eva.yachaytech.edu.ec/mod/page/view.php?id=19672)  
> **Última modificación indicada en EVA:** 22 de octubre de 2026, 12:07  
> **Idioma de esta guía:** Español. Se conservan los nombres técnicos, umbrales y entregables del enunciado original.

## 1. Resumen y propósito

Equipos de **1 o 2 estudiantes** deben implementar aprendizaje por refuerzo tabular (**Q-learning**; **SARSA** opcional) y **Deep Q-Networks (DQN)** usando entornos **Gymnasium**, y comparar ambos métodos experimentalmente.

- El proyecto está diseñado para completarse en uno o dos fines de semana, utilizando principalmente una laptop.
- El RL tabular requiere espacios de estados pequeños o discretizables; DQN requiere un espacio de **acciones discretas**.
- Se utilizan principalmente entornos de **Classic Control** y **LunarLander** (Box2D).
- **BipedalWalker** y la mayoría de tareas de locomoción MuJoCo no se implementan: se analizan por escrito.
- Atari es una extensión opcional que demanda GPU o Colab.
- Se permite IA generativa, pero la evaluación enfatiza el diseño experimental, la interpretación y la defensa oral del código.

## 2. Objetivos de aprendizaje

Al terminar, se espera que el equipo pueda:

1. Decidir si un problema es adecuado para RL tabular, DQN o ninguno, según sus espacios de estados y acciones.
2. Implementar Q-learning o SARSA con discretización de estados y justificar las decisiones.
3. Construir DQN con **red Q**, **experience replay**, **target network** y exploración **ε-greedy**.
4. Diseñar experimentos reproducibles con múltiples semillas, curvas de aprendizaje y bandas de confianza/variabilidad.
5. Ejecutar estudios de hiperparámetros y ablaciones, explicando sus efectos.
6. Comparar eficiencia muestral, estabilidad y sensibilidad a elecciones de diseño.
7. Explicar todas las líneas del código entregado, incluso las generadas con IA.

## 3. Organización, restricciones y alcance

| Aspecto | Requisito |
|---|---|
| Integrantes | 1 o 2 estudiantes |
| Conocimientos previos | MDP, Q-learning, DQN, Python y PyTorch básico |
| Tiempo estimado | Uno o dos fines de semana |
| Hardware | CPU de laptop suficiente para Tier 1 y Tier 2; GPU/Colab solo necesaria para Atari |
| Presupuesto total de entrenamiento | Aproximadamente **4–6 horas de CPU por equipo** para todos los experimentos |
| IA generativa | Permitida, con declaración de uso y verificación |

**Reglas según integrantes:**

- **Individual:** Tier 1 + Tier 2 + análisis teórico obligatorio; Tier 3 opcional como bonus.
- **Pareja:** Tier 1 + Tier 2 + análisis teórico obligatorio **+ una opción Tier 3 obligatoria**.
- Ambos integrantes deben poder defender **todo** el código, no solo la parte que desarrollaron.

## 4. Escenarios y niveles del proyecto

### 4.1. Tier 1 — RL tabular vs. DQN (obligatorio)

Elegir **uno** de los siguientes entornos Classic Control. Implementar Q-learning tabular con discretización (SARSA opcional), implementar DQN en el **mismo entorno** y comparar.

| Entorno | Observaciones / acciones | Umbral de éxito | Aspecto a investigar |
|---|---|---|---|
| **CartPole-v1** | 4 variables continuas / 2 acciones discretas | Retorno medio **≥ 475** en **100 episodios** | Efectos de cantidad y rangos de bins en RL tabular; estabilidad de DQN |
| **MountainCar-v0** | 2 variables continuas / 3 acciones discretas | Retorno medio **≥ −110** en **100 episodios** | Recompensa escasa; por qué RL tabular puede superar a un DQN ingenuo; exploración |

**Mínimos obligatorios:**

- **Tabular:** probar al menos **3 granularidades** de discretización (p. ej., 6, 10 y 20 bins por dimensión).
- **DQN:** incluir replay buffer, target network y política ε-greedy con decaimiento de ε.
- **Comparación:** curvas de aprendizaje de ambos métodos **en los mismos ejes**, con eje horizontal en **pasos del entorno**; comparar también tiempo real de ejecución (*wall-clock*) y desempeño final.
- Utilizar **el mismo presupuesto de pasos** para ambos métodos (ejemplos orientativos: **200 000 pasos en CartPole**, **500 000 en MountainCar**).

### 4.2. Tier 2 — DQN en un problema más difícil (obligatorio)

Elegir **uno** de los siguientes entornos con acciones discretas:

| Entorno | Observaciones / acciones | Umbral de éxito | Tiempo aproximado de entrenamiento en CPU | Nota |
|---|---|---|---|---|
| **LunarLander-v3 (discreto)** | 8 continuas / 4 discretas | Retorno medio **≥ 200** en **100 episodios** | **30 min–2 h** | Opción recomendada; requiere `gymnasium[box2d]` |
| **Acrobot-v1** | 6 continuas / 3 discretas | Retorno medio **≥ −100** en **100 episodios** | **10–30 min** | Alternativa para equipos con hardware limitado |

**Mínimos obligatorios:**

1. Entrenar un agente DQN propio.
2. Analizar al menos **2 hiperparámetros**, seleccionados entre: tasa de aprendizaje, programa de ε, tamaño del buffer, frecuencia de actualización de la red objetivo y ancho de red.
3. Para cada hiperparámetro estudiado, probar **al menos 3 valores**, variando un parámetro a la vez respecto a una configuración base documentada.
4. Ejecutar **una ablación**: DQN original vs. DQN sin **target network** **o** sin **replay buffer** (pueden estudiarse ambos).
5. Interpretar los resultados, incluso cuando no se alcance el umbral de éxito.

### 4.3. Tier 3 — Extensión (una opción como máximo)

**Obligatorio para parejas; opcional (bonus) para proyectos individuales.** Elegir **como máximo una** de estas alternativas:

| Opción | Entorno | Trabajo | Hardware |
|---|---|---|---|
| **A. Discretización de acciones** | `Pendulum-v1` o MuJoCo `InvertedPendulum-v5` | Discretizar el torque continuo en **3, 5 y 11 bins**, entrenar DQN y analizar la compensación entre granularidad y aprendizaje | CPU |
| **B. Mejoras de DQN** | Entorno seleccionado de Tier 2 | Implementar **Double DQN** y/o **Dueling DQN** y comparar con DQN estándar; *prioritized replay* es opcional | CPU |
| **C. Atari** | ALE Pong, a partir de píxeles | DQN con preprocesamiento de frames y apilamiento de frames; se busca una **tendencia de aprendizaje claramente ascendente**, no desempeño humano | GPU o Colab requerido |

### 4.4. Componente de análisis teórico (obligatorio, sin implementación)

Escribir **1–2 páginas** que expliquen:

1. Por qué **BipedalWalker** y las tareas MuJoCo de locomoción **Hopper**, **HalfCheetah** y **Ant** no son adecuadas para RL tabular o DQN, aportando **datos cuantitativos** (por ejemplo, tamaño del espacio de acciones tras discretizarlo).
2. Qué familias de algoritmos son más apropiadas (**PPO, SAC, TD3**, entre otras) y **por qué**.
3. Por qué los métodos tabulares no escalan a observaciones de alta dimensión, como los píxeles de Atari.

## 5. Implementación y herramientas

**Regla principal:** los algoritmos deben ser **implementados por el propio equipo** con Python y PyTorch. No se permite entregar un agente listo para usar proveniente de una librería. Sí se puede utilizar como **baseline de referencia**, identificado claramente.

| Componente | Recomendación del docente |
|---|---|
| Lenguaje | **Python 3.10+** |
| Entornos | `gymnasium`, `gymnasium[box2d]` para LunarLander; `gymnasium[mujoco]` para Tier 3A; `ale-py` para Tier 3C |
| Aprendizaje profundo | **PyTorch**; CPU suficiente para Tier 1–2 |
| Registro | **TensorBoard**, Weights & Biases, o **CSV + matplotlib** |
| Baseline opcional | **Stable-Baselines3 DQN**, solo como comparación, nunca como agente entregado |
| Reproducibilidad | `requirements.txt` o `environment.yml`; seeds fijas en entorno, NumPy y PyTorch |

### 5.1. Estructura de repositorio sugerida

```text
project/
├── agents/          # tabular.py, dqn.py
├── envs/            # wrappers, discretizers
├── experiments/     # configuraciones y scripts de ejecución
├── results/         # logs, gráficas, pesos entrenados, videos/GIFs
├── report/          # artículo final
├── README.md
└── requirements.txt
```

> El `README.md` **debe** explicar cómo reproducir **cada figura** del artículo, con **un comando por experimento**.

## 6. Protocolo experimental obligatorio

1. **Semillas:** cada resultado reportado debe provenir de **al menos 3 semillas independientes**. Para Tier 1 se recomiendan **5**.
2. **Curvas:** representar la **media** con una **banda sombreada** de desviación estándar o intervalo de confianza al **95 %**.
3. **Evaluación periódica:** cada **N pasos de entorno**, ejecutar **10 episodios greedy** (`ε = 0`) y registrar su retorno medio. El enunciado **no fija N**: elegirlo, documentarlo y usarlo consistentemente.
4. **Entrenamiento vs. evaluación:** registrar e informar los retornos de **entrenamiento y evaluación por separado**.
5. **Eje X:** usar **pasos de entorno**, **no episodios**.
6. **Comparación equitativa Tier 1:** fijar el mismo presupuesto de pasos para tabular y DQN.
7. **Barrido de hiperparámetros Tier 2:** variar **un parámetro a la vez**, con al menos **3 valores** alrededor de un baseline documentado, para **al menos 2 parámetros**.
8. **Ablación Tier 2:** comparar DQN vanilla con DQN sin target network y/o sin replay buffer.
9. **Métricas que deben informarse:**
   - Retorno final en evaluación.
   - Pasos hasta alcanzar el umbral de éxito o **“not reached”**.
   - Variación entre semillas.
   - Tiempo de reloj (*wall-clock*).
   - Hardware usado.
10. **Evidencia visual:** un **video corto o GIF** del agente entrenado **por cada entorno**.
11. **Resultados negativos:** son aceptables si existe un diagnóstico sólido. El enunciado indica que un DQN fallido en MountainCar puede alcanzar la puntuación completa si está bien explicado.

## 7. Entregables finales

| Entregable | Contenido exigido | Formato |
|---|---|---|
| **Repositorio de código** | Fuente, configuraciones, `README.md` con comandos de reproducción, `requirements.txt` | Repositorio Git |
| **Paper / artículo** | Título, abstract, introducción, pertinence analysis, métodos, configuración experimental y experimentos, resultados, discusión, componente de análisis sin implementación, declaración de uso de IA | **PDF de 6–8 páginas en doble columna** |
| **Paquete de resultados** | Logs, gráficas de aprendizaje, pesos de modelos entrenados, videos/GIFs | Dentro del repositorio |
| **Defensa oral** | **10 minutos de presentación + 10 minutos de revisión del código y preguntas** | Presencial o en línea |

### 7.1. Preguntas que debe responder la discusión del paper

- ¿Qué método aprendió más rápido en Tier 1 y por qué?
- ¿Qué tan sensible fue cada método a su decisión de diseño más importante (bins para tabular e hiperparámetros para DQN)?
- ¿Qué reveló la ablación acerca de la experiencia de replay y las redes objetivo?

**Importante:** incluir las justificaciones, no solo gráficas o descripciones de resultados.

## 8. Cronograma sugerido

| Momento | Actividades | Resultado esperado |
|---|---|---|
| **Fin de semana 1, día 1** | Instalación, análisis de pertinencia, Q-learning tabular y discretización Tier 1 | Agente tabular funcional y primeras curvas |
| **Fin de semana 1, día 2** | DQN en Tier 1; experimentos con múltiples semillas para ambos algoritmos | Figuras comparativas Tier 1 |
| **Checkpoint entre semana** | Nota corta de avance (**media página**) para el docente | Retroalimentación de alcance y problemas |
| **Fin de semana 2, día 1** | Tier 2: DQN, hiperparámetros, ablación; Tier 3 si corresponde | Resultados y figuras Tier 2/Tier 3 |
| **Fin de semana 2, día 2** | Redacción, componente analítico, videos, limpieza de código | Entrega final |
| **Semana siguiente** | Defensas orales | Evaluación |

Para un equipo que solo disponga de **un fin de semana**, la propuesta del docente es comprimir los días 1–2 al sábado, y los días 3–4 al domingo, **omitiendo Tier 3**. **Precaución:** esto debe conciliarse con la regla anterior que exige Tier 3 en equipos de dos personas; el texto no especifica una excepción expresa para parejas de un solo fin de semana. Se sugieren entrenamientos largos durante la noche o en paralelo a la redacción.

## 9. Uso de IA generativa

Se permite **ChatGPT, Claude, Copilot, Gemini** y herramientas similares para escribir código, depurar y apoyar la redacción, **siempre que**:

- **Divulgación:** el paper incluya una declaración de uso de IA indicando **herramientas**, **tareas** en las que se usaron y un **apéndice con los prompts más importantes**.
- **Dominio del trabajo:** **cada integrante** pueda explicar **cualquier línea de código y figura** durante la defensa. No entender una parte del trabajo reduce la nota de código/defensa.
- **Verificación:** los resultados, hiperparámetros, referencias y afirmaciones sugeridos por IA deben comprobarse. **Referencias o resultados inventados hacen reprobar el criterio del paper**.

Durante la defensa pueden solicitar una **modificación pequeña en vivo**, como cambiar el programa de ε o la discretización y predecir su efecto.

## 10. Rúbrica de evaluación (10 puntos)

> Para equipos de dos personas, Tier 3 cuenta **dentro de los mismos 10 puntos**.

| Criterio | Puntos | Para obtener la puntuación completa | Insuficiente cuando… |
|---|---:|---|---|
| Pertinencia y análisis sin implementación | **0.5** | Razonamiento correcto y cuantificado sobre espacios de estado/acción y familias algorítmicas | Afirmaciones genéricas o incorrectas |
| Implementación y comparación Tier 1 | **1.0** | Agentes tabular y DQN correctos; ≥3 discretizaciones; misma cantidad de pasos para comparar | Falta un método o comparación injusta |
| DQN Tier 2, hiperparámetros, ablación / Tier 3 | **1.0** | Tier 2 alcanza umbral o diagnostica fallo; ≥2 parámetros analizados; ablación explicada / Tier 3 terminado y analizado | Una sola ejecución, ningún estudio |
| Rigor experimental | **0.5** | ≥3 semillas, bandas de variabilidad, evaluación separada, configuraciones reproducibles | Una semilla o resultados no reproducibles |
| Discusión e interpretación | **0.5** | Explica causas, conecta resultados con teoría RL, reconoce limitaciones | Describe curvas sin explicarlas |
| Repositorio, calidad y reproducibilidad | **0.5** | Estructura limpia; README reproduce todas las figuras | Código desorganizado o imposible de ejecutar |
| **Defensa oral** | **5.0** | Cada integrante domina código y resultados y responde a cambios en vivo | No puede explicar su propio código |
| Paper | **1.0** | Secciones bien estructuradas y resumidas | Artículo de baja calidad |
| **TOTAL** | **10.0** | | |

**Dato clave:** la **defensa oral vale 5/10 puntos (50 % de la nota)**.

## 11. Recursos y referencias incluidos por el docente

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*, 2.ª ed. MIT Press. [Libro libre](http://incompleteideas.net/book/the-book-2nd.html).
2. Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518, 529–533. [Artículo](https://www.nature.com/articles/nature14236).
3. van Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement learning with Double Q-learning*. AAAI. [arXiv](https://arxiv.org/abs/1509.06461).
4. Wang, Z., et al. (2016). *Dueling network architectures for deep reinforcement learning*. ICML. [arXiv](https://arxiv.org/abs/1511.06581).
5. [Documentación de Gymnasium](https://gymnasium.farama.org/) — especificaciones de entornos, espacios de observación/acción y umbrales.
6. [CleanRL: implementaciones DQN](https://docs.cleanrl.dev/rl-algorithms/dqn/) — referencia de código en un archivo.
7. [Stable-Baselines3 DQN](https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html) — baseline comparativo opcional.

**Nota del enunciado:** comprobar las versiones de los entornos y umbrales de éxito con la documentación de Gymnasium correspondiente a la versión instalada.

## 12. Checklist de cumplimiento

### Antes de comenzar

- [ ] Definir si el equipo es individual o de dos personas.
- [ ] Seleccionar **un** entorno Tier 1.
- [ ] Seleccionar **un** entorno Tier 2.
- [ ] Si el equipo es de dos: seleccionar **una** extensión Tier 3.
- [ ] Crear repositorio Git y archivo de dependencias.
- [ ] Registrar versiones de Python, PyTorch, Gymnasium y hardware.
- [ ] Configurar y documentar seeds reproducibles.

### Tier 1

- [ ] Implementar Q-learning tabular propio.
- [ ] Implementar discretizador y justificar sus rangos.
- [ ] Probar al menos 3 granularidades.
- [ ] Implementar DQN propio con replay, target network y ε-greedy decay.
- [ ] Ejecutar ≥3 seeds por resultado (5 recomendadas).
- [ ] Comparar con el mismo presupuesto de pasos.
- [ ] Graficar retorno medio y banda de variabilidad en pasos de entorno.
- [ ] Registrar evaluación greedy, tiempo de entrenamiento y rendimiento final.

### Tier 2

- [ ] Implementar y entrenar DQN en el segundo entorno.
- [ ] Seleccionar al menos 2 hiperparámetros.
- [ ] Evaluar al menos 3 valores por parámetro, uno a la vez.
- [ ] Implementar ablación sin target network y/o sin replay.
- [ ] Ejecutar ≥3 seeds y documentar las configuraciones.
- [ ] Explicar los resultados y posibles fallos.

### Tier 3 (si corresponde)

- [ ] Completar una sola opción A, B o C.
- [ ] Comparar, medir y analizar la extensión.

### Análisis, presentación y entrega

- [ ] Escribir 1–2 páginas sobre BipedalWalker, MuJoCo, PPO/SAC/TD3 y píxeles Atari.
- [ ] Guardar logs, figuras, pesos y videos/GIFs para **cada entorno**.
- [ ] Preparar paper PDF de **6–8 páginas, doble columna**, con todas las secciones requeridas.
- [ ] Contestar explícitamente las tres preguntas de discusión.
- [ ] Incluir declaración de IA y apéndice de prompts importantes.
- [ ] Completar README con un comando para reproducir **cada figura**.
- [ ] Preparar presentación de 10 minutos y walkthrough/preguntas de 10 minutos.
- [ ] Asegurar que todos los integrantes entienden cada línea y pueden modificar código en vivo.
- [ ] Presentar nota de progreso de media página en el checkpoint entre semana.

## 13. Aclaraciones y aspectos que NO fija el enunciado

- **Fecha exacta de entrega:** no aparece en el texto de la página revisada. Consultar calendario/tarea del EVA o indicaciones del docente.
- **Valor de N** para evaluación periódica: no se define; debe documentarse.
- **Arquitectura exacta de la red**, tasas de aprendizaje y otros valores por defecto: no se prescriben; deben elegirse, documentarse y justificarse.
- **Plantilla específica del artículo** (IEEE, ACM, etc.): no se menciona; solo se exige **doble columna y 6–8 páginas**.
- **Número exacto de episodios de entrenamiento:** no se fija; el criterio de comparación es el número de **pasos de entorno**.
- Los umbrales de éxito deben cotejarse con la versión de Gymnasium instalada.

---

*Documento de referencia elaborado a partir del contenido del Project 01 publicado en EVA. Las tablas, condiciones y valores numéricos reflejan el enunciado consultado; las listas de verificación y aclaraciones están organizadas para facilitar la ejecución del proyecto.*
