# Cierre experimental de Tier 1 — CartPole

Este documento consolida los experimentos; no es el artículo final ni completa Tier 2/Tier 3.

## Protocolo y alcance

Q-learning propio con 6, 10 y 20 intervalos por variable frente a DQN propio.
Cada ejecución usa 200 000 pasos y semillas independientes 42, 43 y 44.
Las curvas usan pasos de entrenamiento y evaluación greedy separada de 10 episodios por control.
Los resúmenes finales usan 100 episodios por agente. Las bandas son una desviación estándar
entre las tres medias por semilla (ddof=0), sin suavizado; no son intervalos de confianza.

El DQN MSE se eligió después de pruebas exploratorias. Conserva el calendario original:
red 4-64-64-2, Adam 0.0005, gamma 0.99, replay 50 000, lote 64, ajuste cada cuatro pasos
desde el paso 1 000, copia target cada 500 y epsilon de 1 a 0.05 en 160 000 pasos.
Los controles previos, incluidos los fallos, permanecen guardados. Se entrega el modelo final,
sin sustituirlo por el mejor checkpoint. MSE es una variante de pérdida, no un agente prefabricado.

## Resultados con las condiciones de evaluación del desarrollo

| Método | Retorno final medio ± DE | Entrenamiento medio | Modelos finales con media ≥475 |
|---|---:|---:|---:|
| Q-learning (6 intervalos) | 295.33 ± 109.06 | 48.64 s | 0/3 |
| Q-learning (10 intervalos) | 359.70 ± 102.57 | 45.85 s | 0/3 |
| Q-learning (20 intervalos) | 209.00 ± 39.25 | 44.83 s | 0/3 |
| DQN (MSE) | 234.10 ± 188.02 | 250.15 s | 1/3 |
| DQN (Smooth L1 inicial) | 48.47 ± 25.25 | 183.30 s | 0/3 |

Los tiempos excluyen las evaluaciones y dependen de esta implementación y del equipo.
DQN usa un hilo PyTorch en CPU; no se afirma una diferencia universal de velocidad.

### Pasos hasta el umbral

- Q-learning (6 intervalos): 42: not reached en los controles de 100 episodios; 43: not reached en los controles de 100 episodios; 44: not reached en los controles de 100 episodios.
- Q-learning (10 intervalos): 42: not reached en los controles de 100 episodios; 43: not reached en los controles de 100 episodios; 44: not reached en los controles de 100 episodios.
- Q-learning (20 intervalos): 42: not reached en los controles de 100 episodios; 43: not reached en los controles de 100 episodios; 44: not reached en los controles de 100 episodios.
- DQN (MSE): 42: not reached en los controles de 100 episodios; 43: not reached en los controles de 100 episodios; 44: 200000.
- DQN (Smooth L1 inicial): 42: not reached en los controles de 100 episodios; 43: not reached en los controles de 100 episodios; 44: not reached en los controles de 100 episodios.

La detección no localiza el instante exacto de aprendizaje. Tabular comprobó 100 episodios
al inicio y al final; DQN también confirma con 100 cualquier control intermedio de 10 que dé ≥475.
Por esa diferencia no se comparan los tiempos de primer éxito como si su detección fuera idéntica.

## Evaluación adicional con condiciones iniciales nuevas

Se fijaron antes de ver sus resultados 100 episodios nuevos por modelo, con semillas de reset
1010000 + 1000*semilla_entrenada + episodio. Los métodos comparten condiciones emparejadas,
sin solapamiento de esos intervalos entre las tres semillas entrenadas. No se actualizan pesos
ni tablas ni se usan estos resultados para ajustar los modelos aquí evaluados.
Son nuevos inicios de la misma distribución, no nuevos entrenamientos ni prueba universal.

| Método | Retorno nuevo medio ± DE | Modelos con media ≥475 |
|---|---:|---:|
| Q-learning (6 intervalos) | 285.28 ± 112.59 | 0/3 |
| Q-learning (10 intervalos) | 352.87 ± 96.64 | 0/3 |
| Q-learning (20 intervalos) | 208.12 ± 47.74 | 0/3 |
| DQN (MSE) | 234.19 ± 187.96 | 1/3 |
| DQN (Smooth L1 inicial) | 47.08 ± 24.02 | 0/3 |

## Interpretación

- DQN MSE obtuvo mayor retorno temprano en el primer control de 10 000 pasos que los tabulares;
  eso no demuestra aprendizaje estable ni cumplimiento del umbral. Las curvas muestran pérdida
  posterior de rendimiento. La comparación de eficiencia muestral debe distinguir el inicio del final.
- En tabular, la mejor media final entre las granularidades probadas fue 10 intervalos.
  Más intervalos aportan detalle, pero reparten experiencia entre más casillas; menos intervalos
  agrupan situaciones que pueden requerir acciones distintas. No se afirma que 10 sea óptimo universal.
- El cambio a MSE mejoró la prueba corta de 50 000 pasos en las tres semillas, pero esa mejora
  no permitía anticipar el resultado de 200 000. Se conservan ambos presupuestos sin mezclarlos.
- La sobreestimación persiste en los modelos finales. No se identificó una causa única; las
  auditorías de transiciones, cálculos independientes y un MDP con solución conocida pasaron.
  Esto respalda componentes concretos, sin demostrar corrección total ni estabilidad general.

### Diagnóstico de las estimaciones MSE finales

Se consultaron 1 000 estados legales de una política aleatoria por semilla. Con recompensa ≤1
y gamma=0.99, el verdadero retorno descontado no supera 100. No es el retorno sin descuento
de un episodio, cuyo máximo es 500. La muestra diagnóstica no se utiliza para entrenar.

| Semilla | Menor Q | Mayor Q | Valores Q >100 |
|---|---:|---:|---:|
| 42 | -111.86 | 214.12 | 87.55 % |
| 43 | -127.62 | 244.97 | 90.75 % |
| 44 | -188.78 | 151.68 | 81.40 % |

## Reproducibilidad y evidencia

Python 3.12.0; PyTorch 2.15.0+cpu; Gymnasium 1.4.0.
Sistema: Windows-11-10.0.26200-SP0. CPU: AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD; 8 procesadores lógicos.
Los archivos config.json registran dependencias, semillas, rangos y parámetros por ejecución.
Los CSV separan entrenamiento y evaluación; se conservan pérdidas, tablas y modelos finales.
La verificación carga los modelos y reproduce sus evaluaciones finales. El video usa la primera
semilla y el primer episodio de evaluación, no el mejor intento ni un promedio.

Figuras y métricas: results/parte_15_mse_completo/comparacion.
Evaluación nueva: results/parte_15_mse_completo/evaluacion_nueva.
Diagnósticos y pesos: results/parte_15_mse_completo/seed_*.
Historial de pruebas: report/notas_discretizacion.md, notas_dqn_cartpole.md, notas_learning_rate.md,
notas_target_corto.md, auditoria_dqn.md y notas_loss.md. El uso de IA está en report/uso_ia.md.

## Estado del proyecto

La comparación experimental de Tier 1 queda consolidada con sus resultados y limitaciones.
Faltan Tier 2 (LunarLander, estudios y ablación), Tier 3 para la pareja (Double DQN), análisis
teórico general, integración del artículo PDF de 6–8 páginas y preparación de la defensa.
Este cierre no afirma que todos los agentes resolvieran el entorno ni que el proyecto completo esté terminado.
