# Tier 1: primera configuración DQN frente a Q-learning

## Diseño

CartPole-v1 estándar, semillas 42/43/44, 200 000 pasos por ejecución en CPU.
Mismas semillas de inicio de evaluación y checkpoints que la comparación tabular.
El protocolo se fijó antes de entrenar y se conserva el modelo final, sin elegir
el mejor checkpoint. Los parámetros completos están en los `config.json`.

DQN propio: red 4-64-64-2, Adam con learning rate 0.0005, gamma=0.99,
replay de 50 000 experiencias, lote de 64, inicio en el paso 1 000,
un ajuste cada cuatro pasos y copia de target cada 500 pasos. Epsilon baja de
1 a 0.05 en 160 000 pasos. Smooth L1 y límite de norma del gradiente de 10.
Un hilo PyTorch en CPU y operaciones deterministas.

## Resultados finales

Cada agente se evaluó en 100 episodios greedy. DE es la desviación estándar
entre las tres medias por semilla, no un intervalo de confianza.

| Método | Retorno medio ± DE | Tiempo medio de entrenamiento |
|---|---:|---:|
| Q-learning, 6 intervalos | 295.33 ± 109.06 | 48.64 s |
| Q-learning, 10 intervalos | 359.70 ± 102.57 | 45.85 s |
| Q-learning, 20 intervalos | 209.00 ± 39.25 | 44.83 s |
| DQN, primera configuración | 48.47 ± 25.25 | 183.30 s |

DQN dio 33.26, 84.06 y 28.10 por semilla. No se observó el umbral 475 en
ninguna comprobación de 100 episodios. Los controles intermedios de 10 episodios
tampoco activaron confirmaciones adicionales. Para tabular solo se comprobó el
criterio de 100 episodios al inicio/final, por lo que no se determina el instante
exacto de primer éxito de ningún método.

Los tiempos excluyen evaluaciones y dependen del código y los hilos elegidos;
no prueban una diferencia universal de velocidad entre algoritmos.

Las curvas muestran aprendizaje temprano de DQN seguido de pérdida de estabilidad.
En el control de 10 000 pasos, su media fue aproximadamente 240 frente a 72,
98 y 56 de los tabulares. Ese dato de 10 episodios no equivale a alcanzar el
umbral ni justifica reemplazar retrospectivamente el modelo final.

## Diagnóstico observado

Se consultó cada modelo final en 1 000 estados legales de una política aleatoria
con semillas nuevas 90042/90043/90044, sin entrenar con ellos. Con recompensa
entre 0 y 1 y gamma=0.99, el retorno descontado verdadero está entre 0 y
1/(1-0.99)=100. Este valor Q es distinto del retorno sin descuento del episodio,
que puede llegar a 500.

| Semilla | Mayor Q observado | Valores por encima de 100 | Pérdida media primeros → últimos 1 000 ajustes |
|---|---:|---:|---:|
| 42 | 1120.40 | 98.05 % | 0.152 → 49.804 |
| 43 | 420.63 | 94.60 % | 0.155 → 7.842 |
| 44 | 279.56 | 90.20 % | 0.177 → 8.990 |

También aparecieron valores negativos. Hay evidencia de sobreestimación y
pérdida creciente, coherente con un ajuste inestable. No se ha aislado una causa
única: no puede atribuirse el fallo solo a learning rate, copia de target,
exploración o replay. La red objetivo no garantiza estabilidad, y un menor
error en una actualización inicial tampoco demuestra buen control.

## Verificación y reproducción

- Pasaron 17 pruebas: red, objetivos terminales, replay, target y evaluación.
- Cada ejecución registró 200 000 pasos, 49 751 ajustes, 400 copias de target
  y pérdidas finitas.
- Se cargaron los tres modelos finales y se reprodujeron exactamente sus 100
  retornos finales. Evidencia: `results/parte_10_dqn/verification.json`.
- El GIF usa la primera semilla y el primer intento de evaluación, sin buscar
  el mejor. Dio retorno 16, frente a la media 33.26 de ese modelo; no representa
  el promedio.

Desde el entorno del proyecto:

```text
python -m experiments.entrenar_dqn
python -m experiments.comparar_tier1
python -m experiments.diagnosticar_dqn
python -m experiments.ver_dqn
```

Resultados en `results/parte_10_dqn`: configuraciones, logs, pesos, curvas y GIF.
La figura incluye los tres tabulares y DQN con igual eje de pasos. Se separan
entrenamiento y evaluación. Las curvas usan 10 episodios por control y la tabla
final usa 100; las bandas son DE entre semillas (ddof=0).

## Interpretación y siguiente experimento

Con estos parámetros y presupuesto, Q-learning de 10 intervalos obtuvo el mejor
promedio final y fue más rápido en tiempo real. No se concluye que Q-learning
siempre supere a DQN. El resultado negativo de esta configuración se conserva.

La siguiente etapa puede probar una tasa de aprendizaje menor como cambio
aislado, conservando semillas, pasos y demás parámetros. Permitirá investigar
si ajustes menores reducen la sobreestimación, sin presuponer éxito. Debe guardarse
como experimento nuevo, sin sustituir estas ejecuciones.
Tier 2, Tier 3, análisis general y artículo final siguen pendientes.
