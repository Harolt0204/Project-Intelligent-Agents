# Nota de avance — Project 01

El equipo de dos integrantes completó la comparación experimental de Tier 1
en CartPole-v1. Implementamos Q-learning con discretización y un DQN propio
con PyTorch, replay buffer, red objetivo y exploración epsilon-greedy decreciente.
Trabajamos con tres semillas por configuración, 200 000 pasos por entrenamiento,
evaluaciones periódicas separadas y evaluación final de 100 episodios por agente.

Q-learning obtuvo retornos finales medios de 295.33, 359.70 y 209.00 con 6, 10
y 20 intervalos por variable. El DQN inicial fue inestable; se conservaron los
resultados negativos y se estudiaron cambios aislados de tasa, copia de la red
objetivo y pérdida. Con MSE, el retorno final medio fue 234.10, con alta dispersión:
dos semillas quedaron cerca de 101 y una logró 500 en los 100 episodios.

Una evaluación adicional con condiciones iniciales nuevas mantuvo el patrón:
352.87 para Q-learning de 10 intervalos y 234.19 para DQN MSE. La semilla exitosa
volvió a obtener 500. Las estimaciones Q siguen presentando sobreestimación.
Se verificaron transiciones, cálculos independientes y reproducción de los
modelos; se guardaron configuraciones, registros, curvas, pesos y animaciones.

El siguiente bloque es Tier 2 en LunarLander: estudio de dos hiperparámetros
con al menos tres valores y una ablación. Después completaremos Double DQN
como Tier 3, el análisis teórico, el artículo y la defensa. El uso de IA está
registrado y ambos integrantes deberán explicar todo el código y los resultados.
