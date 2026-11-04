# Tier 1: notebook y mejora de CartPole

## Protocolo

Implementaciones propias de Q-learning y DQN estándar. Cada agente entrena 200000 pasos
con semillas 42, 43 y 44. Evaluación greedy separada cada 10000 pasos (10 episodios),
evaluación final de 100 episodios y confirmaciones de 100 si un control da ≥475.
Se entrega el modelo final; no se elige el mejor checkpoint ni se descartan semillas.
Las bandas son una desviación estándar entre las tres medias por semilla, no IC del 95%.
Las curvas usan los primeros 10 episodios también al inicio/final; las tablas finales usan 100.
El objetivo del enunciado es ≥475; se registra por separado el umbral declarado por Gymnasium.

## Resultados finales

| Método | Media ± DE | Retornos por semilla (42, 43, 44) | Semillas ≥475 | Tiempo de entrenamiento |
|---|---:|---|---:|---:|
| Q-learning (6 intervalos) | 414.48 ± 62.85 | 350.75, 500.00, 392.69 | 1/3 | 58.89 s |
| Q-learning (10 intervalos) | 89.09 ± 5.48 | 93.94, 81.43, 91.89 | 0/3 | 81.07 s |
| Q-learning (20 intervalos) | 26.28 ± 2.40 | 28.48, 22.94, 27.43 | 0/3 | 50.70 s |
| DQN ajustado | 498.60 ± 1.98 | 495.79, 500.00, 500.00 | 3/3 | 243.48 s |
| DQN MSE anterior | 234.10 ± 188.02 | 101.67, 100.63, 500.00 | 1/3 | 250.15 s |

Los tiempos excluyen evaluación y son mediciones de esta computadora; no son benchmarks universales.
No se ejecutaron entrenamientos simultáneos. Se realizaron tareas de edición y comprobación durante el trabajo.

## Verificación y nuevos inicios

Los 100 retornos finales de cada uno de los 12 modelos se reprodujeron exactamente al cargarlo.
Después se evaluaron 100 inicios nuevos por modelo, sin entrenar ni ajustar parámetros.
La muestra se fijó antes de observarla: semilla de evaluación 3000000+1000*semilla_entrenada.
Son nuevos inicios de la misma distribución, no nuevas semillas de entrenamiento.

| Método | Media nueva ± DE | Semillas ≥475 |
|---|---:|---:|
| Q-learning (6 intervalos) | 425.15 ± 54.01 | 1/3 |
| Q-learning (10 intervalos) | 90.32 ± 15.49 | 0/3 |
| Q-learning (20 intervalos) | 28.08 ± 0.87 | 0/3 |
| DQN ajustado | 499.00 ± 1.41 | 3/3 |

La mayor media tabular entre estas tres granularidades fue Q-learning (6 intervalos): 414.48. Esa selección se hace con la media de las tres semillas, no con un episodio.
Si una granularidad no alcanza 475, se informa como resultado negativo; no se reemplaza por otra semilla.

## Qué cambió y qué podemos concluir

- DQN conserva replay, target, epsilon-greedy, recompensa original y corrección de cortes temporales.
- Red: [4, 120, 84, 2]; tasa 0.00025; memoria 10000; lote 128; inicio 10000; ajuste cada 10 pasos.
- Target cada 500; pérdida MSE; epsilon hasta 0.05 en 100000 pasos.
- Configuración DQN inspirada en CleanRL, pero ejecutada con nuestro propio agente. Se mantienen 200000 pasos,
  frente a los 500000 del default de referencia. También difieren generadores aleatorios y clipping de gradiente.
- Tabular: alpha=0.1, gamma=0.99, epsilon final=0.01; límites inferiores=[-2.4, -0.75, -0.20943951023931956, -0.8726646259971648]; superiores=[2.4, 0.75, 0.20943951023931956, 0.8726646259971648].
- Se modificaron varios parámetros para buscar una configuración funcional. Esto no es una ablación controlada
  ni permite atribuir toda la mejora a un único cambio. El estudio controlado de Tier 2 sigue pendiente.
- Las semillas 42/43/44 se usaron durante el desarrollo: no constituyen un test independiente de selección.
  La evaluación nueva comprueba otros inicios, sin demostrar éxito universal ni eliminar esa limitación.
- Más intervalos no garantizan mejor aprendizaje: cambian la resolución y el número de casillas por visitar.
- El ajuste tabular no mejoró todas las granularidades: 10 y 20 intervalos empeoraron respecto al diseño anterior.
  Los rangos y la exploración cambiaron juntos, por lo que no aislamos una causa única de ese deterioro.
- Los experimentos anteriores permanecen íntegros: DQN MSE anterior=234.10±188.02; mejor tabular anterior=359.70±102.57.

## Pasos hasta el objetivo

Primer control confirmado de 100 episodios; no identifica el instante exacto de aprendizaje:

- Q-learning (6 intervalos): 42: not reached, 43: 160000, 44: not reached
- Q-learning (10 intervalos): 42: not reached, 43: not reached, 44: not reached
- Q-learning (20 intervalos): 42: not reached, 43: not reached, 44: not reached
- DQN ajustado: 42: 190000, 43: 160000, 44: 160000
- DQN MSE anterior: 42: not reached, 43: not reached, 44: 200000

## Diagnóstico de valores Q

Máximos finales en los 256 estados de diagnóstico de cada semilla: [102.08712005615234, 101.60016632080078, 101.17017364501953].
Con gamma=0.99 y recompensa ≤1 el retorno descontado verdadero no supera 100.
Persisten pequeñas sobreestimaciones incluso cuando la política alcanza el objetivo.
La muestra diagnóstica no entrena al agente ni demuestra corrección en todos los estados.
Este valor descontado es distinto del retorno de hasta 500 puntos por episodio.

## Reproducibilidad

Python 3.12.0; paquetes {'gymnasium': '1.4.0', 'numpy': '2.5.4', 'torch': '2.15.0+cpu'}; plataforma Windows-11-10.0.26200-SP0.
CPU AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD; 8 procesadores lógicos; PyTorch CPU, un hilo.
Parámetros editables en configs/tier1_mejorado.json y notebook Tier1_Explicado.ipynb.
Los comandos están en REPRODUCIR_TIER1.md. El notebook muestra el código real y los archivos guardados.

## Referencias y alcance

- CleanRL DQN: https://docs.cleanrl.dev/rl-algorithms/dqn/
- Configuración de referencia: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py
- Gymnasium CartPole: https://gymnasium.farama.org/environments/classic_control/cart_pole/

Este bloque corresponde a Tier 1. Faltan Tier 2, Tier 3 para la pareja, análisis teórico,
integración del artículo final y preparación de defensa. El notebook no sustituye el paper.
