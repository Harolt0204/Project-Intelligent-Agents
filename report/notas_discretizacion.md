# Tier 1: primera comparación tabular de discretizaciones

## Diseño y evidencia

CartPole-v1, Q-learning propio, 6/10/20 intervalos uniformes por variable,
semillas 42/43/44 y 200 000 pasos de entrenamiento por ejecución. Alpha=0.1,
gamma=0.99 y epsilon lineal de 1 a 0.05 durante el primer 80% de pasos.
Los rangos son idénticos en las tres opciones. Se conserva el mismo presupuesto,
hardware y configuración restante; se ejecuta secuencialmente en una máquina.

Fuentes: `results/parte_06_bins/protocol.json`, configuraciones por semilla y CSV.
El protocolo de figuras está en `figure_protocol.json`. Las figuras se regeneran
con `python -m experiments.comparar_bins --plot-only` desde el entorno del proyecto.

## Resultados finales

Cada agente se evaluó durante 100 episodios greedy, sin actualizar su tabla.
La desviación estándar es entre las tres medias por semilla (`ddof=0`), no
entre los 300 episodios agrupados y no un intervalo de confianza.

| Intervalos | Casillas posibles | Retorno final medio ± DE | Tiempo de entrenamiento medio |
|---|---:|---:|---:|
| 6 | 1 296 | 295.33 ± 109.06 | 48.64 s |
| 10 | 10 000 | 359.70 ± 102.57 | 45.85 s |
| 20 | 160 000 | 209.00 ± 39.25 | 44.83 s |

Los nueve resultados individuales están en `summary.json`. Ninguna ejecución
alcanzó 475 en la evaluación final de 100 episodios. No se confirma que el
umbral se alcanzara antes: los controles intermedios solo usaron 10 episodios.
Este diseño no permite estimar con precisión el primer paso de éxito.
Los tiempos de entrenamiento excluyen las evaluaciones. El total registrado
de las nueve ejecuciones, incluidas evaluaciones, fue aproximadamente 8.51 min.
Las pequeñas diferencias de tiempo no bastan para afirmar superioridad computacional.

## Interpretación provisional

- Con estos rangos, parámetros y presupuesto, 10 bins obtuvo la mayor media final.
  La dispersión es amplia y solo se usaron tres semillas: no se establece una
  superioridad estadística ni que 10 sea óptimo para toda configuración.
- Más detalle no produjo una mejora final monotónica. Con 20 bins hay 160 000
  casillas frente a 10 000 con 10 bins: la experiencia se distribuye entre más
  situaciones discretas y se comparte menos entre observaciones cercanas.
  Esto es una explicación plausible, no una causalidad demostrada por separado.
- La agrupación también puede mezclar situaciones que requieren acciones
  diferentes. Con pocas casillas se pierde información útil. Las curvas muestran
  fluctuaciones: el desempeño no crece de manera continua.
- Promedio de casillas visitadas en entrenamiento: 368.67 (6 bins), 1 228.33
  (10 bins) y 3 823.33 (20 bins). No todas las combinaciones de la tabla son
  igualmente probables o alcanzables: un bajo porcentaje de cobertura por sí
  solo no prueba un defecto de exploración.
- Las entradas de entrenamiento fuera de los rangos elegidos ocurrieron solo
  en velocidad del carrito: 728/600 000 (6 bins), 133/600 000 (10 bins) y
  15/600 000 (20 bins). Las otras tres dimensiones registraron cero. Esto indica
  baja frecuencia de saturación en estas ejecuciones, pero no demuestra que
  los límites elegidos sean óptimos. Los conteos excluyen las observaciones
  posteriores al final de la tarea y las evaluaciones.

## Lectura de la figura

Eje X: pasos de entrenamiento. Eje Y: retorno medio de 10 episodios greedy.
En todos los checkpoints se usan los primeros 10 episodios registrados, incluso
cuando el log inicial/final contiene 100. Por eso los últimos puntos de las
curvas no tienen que coincidir con la tabla de rendimiento final de 100 episodios.
Cada línea es la media entre semillas; su banda es ±una DE entre esas medias.
No se aplica suavizado. La línea horizontal marca el criterio de 475, pero
cruzarla en una evaluación de 10 episodios no demostraría cumplir el criterio
de 100 episodios.

## Límites y trabajo pendiente

- Solo se varía el número de intervalos; los rangos de discretización no se barren.
- No hay aún comparación con DQN; Tier 1 completo sigue pendiente.
- No se ha ajustado alpha ni el calendario de exploración para cada granularidad.
  Se priorizó mantener los demás parámetros iguales para esta comparación.
- Las condiciones iniciales de evaluación se repiten entre checkpoints. Antes
  de la entrega conviene añadir una evaluación independiente con otras semillas
  para estudiar generalización, sin utilizarla para seleccionar configuraciones.
- Los resultados negativos deben conservarse y discutirse, sin sustituirlos por
  una ejecución afortunada. Es posible investigar mejoras como una nueva etapa
  explícita, sin modificar retrospectivamente estos experimentos.
