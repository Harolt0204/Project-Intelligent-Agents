# Prueba corta: copia de target cada 500 o 2 000 pasos

## Diseño

Se vuelve a la tasa original 0.0005. Cada opción se entrenó desde cero con
semillas 42/43/44 y 50 000 pasos. Solo cambió el intervalo de copia de la red
objetivo; esto se verificó comparando los archivos de configuración.
Se conservó el calendario de epsilon de 160 000 pasos: no se comprimió a 40 000
por acortar el presupuesto. Por tanto, no se llegó a la fase de exploración mínima.
Los seis entrenamientos fueron secuenciales en la misma máquina.

La referencia de 500 se repitió para evaluar su modelo a 50 000 pasos en 100
episodios. En el experimento original, ese control solo tenía 10 episodios.
La nueva referencia reprodujo exactamente los primeros 10 retornos de todos
los controles originales correspondientes, incluidos los de 50 000 pasos.

Se registraron valores Q en un banco fijo de 256 estados obtenidos con acciones
aleatorias (semillas 90042/90043/90044). No entraron al replay ni modificaron la
red. Las evaluaciones y el banco diagnóstico no consumieron pasos de entrenamiento.

## Resultados al finalizar 50 000 pasos

Retornos finales evaluados en 100 episodios por agente; DE entre medias de semillas.

| Intervalo de copia | Semilla 42 | Semilla 43 | Semilla 44 | Media ± DE |
|---|---:|---:|---:|---:|
| 500 pasos | 53.25 | 127.94 | 131.12 | 104.10 ± 35.98 |
| 2 000 pasos | 75.10 | 109.66 | 41.35 | 75.37 ± 27.89 |

Con la copia cada 2 000 hubo mejora en una semilla y empeoramiento en dos.
No se observó mejora del promedio final; la diferencia de medias fue −28.73.
La muestra no permite afirmar una superioridad estadística o universal.
Ninguna opción alcanzó el criterio 475 en su evaluación final.

Los máximos Q observados en el banco a 50 000 pasos fueron:

| Intervalo | Semilla 42 | Semilla 43 | Semilla 44 |
|---|---:|---:|---:|
| 500 | 79.43 | 72.43 | 75.56 |
| 2 000 | 24.78 | 24.13 | 24.58 |

Las copias menos frecuentes frenaron el crecimiento observado de Q en esta
prueba. Sin embargo, hubo 25 copias en vez de 100: la propagación de estimaciones
también avanza más despacio. Un Q menor no demuestra por sí solo que la estimación
sea más correcta, que la política sea mejor o que no vaya a divergir después.
Los máximos observados de ambas opciones estaban por debajo del límite teórico
100 en este banco y al finalizar esta prueba; no es una garantía para todos los
estados ni para los 150 000 pasos restantes del presupuesto largo.

La hipótesis de ralentizar el crecimiento tiene apoyo en el diagnóstico corto,
pero no se demostró una mejora del control ni estabilidad a largo plazo.
No deben compararse estas medias finales de 50 000 con las de 200 000 como si
representaran el mismo presupuesto. El resultado inicial favorable de una semilla
en 10 000 pasos tampoco se seleccionó para reemplazar sus resultados finales.

## Comprobaciones

- Pasaron las 17 pruebas antes de ejecutar la comparación.
- Cada modelo recibió 50 000 pasos y 12 251 ajustes de la red online.
- Se comprobaron las copias: 100 con intervalo 500, 25 con intervalo 2 000.
- Se verificó el calendario de exploración de 160 000 pasos y los 6 checkpoints.
- Los modelos guardados reprodujeron exactamente sus 100 retornos finales.
- Datos, modelos y verificaciones: `results/parte_12_target_corto`.

## Reproducción

```text
python -m experiments.probar_target
python -m experiments.probar_target --plot-only
```

El primer comando entrena y grafica; exige otra carpeta de salida si ya hay
resultados. El segundo regenera las figuras PNG/PDF y tablas desde los logs.
La figura izquierda usa 10 episodios por punto con media y DE entre semillas;
el resumen final usa 100. La derecha muestra el máximo diagnóstico por semilla
en escala simétrica logarítmica. No se suavizan las curvas.

## Próximo paso

No hay evidencia suficiente para tratar la copia cada 2 000 como una mejora
del agente. Antes de decidir otro entrenamiento largo, conviene auditar el flujo
de experiencias y actualización y contrastar la implementación/configuración
con una referencia primaria de DQN. Cualquier cambio posterior debe distinguir
una corrección de código de un nuevo experimento de hiperparámetros.
No se sobrescriben ni se ocultan los resultados negativos anteriores.
