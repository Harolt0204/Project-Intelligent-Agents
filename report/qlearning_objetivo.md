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
| Q-learning (2/2/6/6 intervalos) | 495.73 ± 6.04 | 487.19, 500.00, 500.00 | 3/3 | 44.67 s |
| Q-learning (2/2/10/10 intervalos) | 163.84 ± 43.19 | 136.22, 224.83, 130.46 | 0/3 | 51.23 s |
| Q-learning (2/2/20/20 intervalos) | 170.21 ± 24.87 | 136.99, 196.82, 176.83 | 0/3 | 51.44 s |
| DQN ajustado | 498.60 ± 1.98 | 495.79, 500.00, 500.00 | 3/3 | 243.48 s |

Los tiempos excluyen evaluación y son mediciones de esta computadora; no son benchmarks universales.
No se ejecutaron entrenamientos simultáneos. Se realizaron tareas de edición y comprobación durante el trabajo.

## Verificación y nuevos inicios

Los 100 retornos finales de cada uno de los 12 modelos se reprodujeron exactamente al cargarlo.
Después se evaluaron 100 inicios nuevos por modelo, sin entrenar ni ajustar parámetros.
La muestra se fijó antes de observarla: semilla de evaluación 5000000+1000*semilla_entrenada.
Son nuevos inicios de la misma distribución, no nuevas semillas de entrenamiento.

| Método | Media nueva ± DE | Semillas ≥475 |
|---|---:|---:|
| Q-learning (2/2/6/6 intervalos) | 496.47 ± 4.99 | 3/3 |
| Q-learning (2/2/10/10 intervalos) | 167.22 ± 41.43 | 0/3 |
| Q-learning (2/2/20/20 intervalos) | 168.08 ± 23.81 | 0/3 |
| DQN ajustado | 498.61 ± 1.97 | 3/3 |

La mayor media tabular entre estas tres granularidades fue Q-learning (2/2/6/6 intervalos): 495.73. Esa selección se hace con la media de las tres semillas, no con un episodio.
Si una granularidad no alcanza 475, se informa como resultado negativo; no se reemplaza por otra semilla.

## Qué cambió y qué podemos concluir

- DQN conserva replay, target, epsilon-greedy, recompensa original y corrección de cortes temporales.
- Red: [4, 120, 84, 2]; tasa 0.00025; memoria 10000; lote 128; inicio 10000; ajuste cada 10 pasos.
- Target cada 500; pérdida MSE; epsilon hasta 0.05 en 100000 pasos.
- Configuración DQN inspirada en CleanRL, pero ejecutada con nuestro propio agente. Se mantienen 200000 pasos,
  frente a los 500000 del default de referencia. También difieren generadores aleatorios y clipping de gradiente.
- Tabular: alpha=0.1, gamma=0.99, epsilon final=0.01; límites inferiores=[-2.4, -0.75, -0.20943951023931956, -0.872664625997165]; superiores=[2.4, 0.75, 0.20943951023931956, 0.872664625997165].
- Inicialización de la tabla Q: 100.0 en todas las casillas y acciones; no cambia recompensa ni fórmula Q-learning.
- Alpha final: 0.01; espaciado de intervalos: potencia 1.0 (1=uniforme).
- Ejemplo de resolución (configuración angular 10): [2, 2, 10, 10] intervalos por variable; 400 estados tabulares.
- Se modificaron varios parámetros para buscar una configuración funcional. Esto no es una ablación controlada
  ni permite atribuir toda la mejora a un único cambio. El estudio controlado de Tier 2 sigue pendiente.
- Las semillas 42/43/44 se usaron durante el desarrollo: no constituyen un test independiente de selección.
  La evaluación nueva comprueba otros inicios, sin demostrar éxito universal ni eliminar esa limitación.
- Más intervalos no garantizan mejor aprendizaje: cambian la resolución y el número de casillas por visitar.
- Las granularidades que no alcancen 475 se conservan e informan; la mejor se selecciona por su media entre semillas.
- Los experimentos anteriores permanecen íntegros: DQN MSE anterior=234.10±188.02; mejor tabular anterior=359.70±102.57.

## Pasos hasta el objetivo

Primer control confirmado de 100 episodios; no identifica el instante exacto de aprendizaje:

- Q-learning (2/2/6/6 intervalos): 42: 70000, 43: 20000, 44: 60000
- Q-learning (2/2/10/10 intervalos): 42: not reached, 43: not reached, 44: 40000
- Q-learning (2/2/20/20 intervalos): 42: not reached, 43: not reached, 44: not reached
- DQN ajustado: 42: 190000, 43: 160000, 44: 160000

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
