# CartPole: reducir únicamente la tasa de aprendizaje DQN

## Pregunta y protocolo

Hipótesis fijada antes de ejecutar: pasar de learning rate 0.0005 a 0.0001
podría reducir sobreestimación e inestabilidad. Se mantuvieron semillas
42/43/44, 200 000 pasos, arquitectura, replay, lote, exploración y target.
La comparación automática de los archivos `config.json` confirma que el único
campo distinto entre ambas ejecuciones emparejadas es `learning_rate`.
Las nuevas ejecuciones están en `results/parte_11_lr`; las originales permanecen
en `results/parte_10_dqn`. No se seleccionó el mejor checkpoint de ninguna.

## Resultados finales (100 episodios por agente)

| Tasa DQN | Semilla 42 | Semilla 43 | Semilla 44 | Media ± DE entre semillas |
|---|---:|---:|---:|---:|
| 0.0005 | 33.26 | 84.06 | 28.10 | 48.47 ± 25.25 |
| 0.0001 | 9.36 | 9.38 | 9.38 | 9.37 ± 0.01 |
| Diferencia menor − original | −23.90 | −74.68 | −18.72 | −39.10 |

Ningún modelo nuevo alcanzó 475. La tasa menor empeoró el retorno final en las
tres semillas y no mostró una mejora en los controles intermedios. La baja DE
solo indica resultados consistentemente bajos: no implica buen aprendizaje.
Como referencia, Q-learning de 10 intervalos obtuvo 359.70 ± 102.57, pero fue
seleccionado entre las granularidades del estudio anterior, no fijado como
óptimo universal.

## Diagnóstico

Se aplicó el mismo protocolo a los tres modelos nuevos: 1 000 estados de una
política aleatoria con semillas 90042/90043/90044. Recompensa máxima 1 y
gamma=0.99 implican Q verdadero ≤100. Este es retorno descontado; el límite
sin descuento de un episodio de CartPole sigue siendo 500.

| Semilla | Máximo Q observado | Proporción de valores Q >100 | Acción preferida en los 1 000 estados |
|---|---:|---:|---|
| 42 | 1 314 222.38 | 100 % | Derecha en todos |
| 43 | 1 761 743.38 | 100 % | Izquierda en todos |
| 44 | 1 465 808.12 | 100 % | Izquierda en todos |

La pérdida media de los últimos 1 000 ajustes fue aproximadamente 236 946,
238 356 y 257 329, respectivamente; los primeros 1 000 ajustes registraron
medias entre 0.33 y 0.41. Las pérdidas y pesos permanecieron finitos, pero
las estimaciones crecieron enormemente. La política prefirió una sola acción
en toda la muestra diagnóstica, sin demostrar control dependiente del estado.
El GIF del primer modelo duró 10 pasos; es un intento ilustrativo, no una media.

Estos resultados no apoyan la hipótesis de que esta reducción aislada de tasa
solucionaría el problema. No prueban que todas las tasas menores sean peores,
ni que la tasa sea la única causa. Una corrección menor por ajuste no garantiza
estabilidad del proceso completo con objetivos estimados y aproximación neuronal.

## Verificación y reproducción

- Pasaron 17 pruebas de componentes y aislamiento de evaluación.
- Los inicios de evaluación coinciden exactamente con el experimento original.
- Cada semilla registró 200 000 pasos y 49 751 ajustes, con pérdidas finitas.
- Al cargar cada modelo final se reprodujeron exactamente sus 100 retornos.
- Evidencia en `results/parte_11_lr/verification.json` y configuraciones por semilla.

Comandos desde el entorno del proyecto:

```text
python -m experiments.entrenar_dqn --learning-rate 0.0001 --output results/parte_11_lr
python -m experiments.diagnosticar_dqn --root results/parte_11_lr
python -m experiments.comparar_learning_rate
python -m experiments.comparar_tier1 --dqn-root results/parte_11_lr
python -m experiments.ver_dqn --run results/parte_11_lr/seed_42
```

La figura de ambas tasas está en `results/parte_11_lr/comparacion_lr` y usa
10 episodios por punto, sin suavizado, con media y DE entre tres semillas.
El resumen final usa 100 episodios y los datos por semilla se conservan.

## Límites y siguiente pregunta

Son dos tasas elegidas secuencialmente después de observar la configuración
inicial. No es el estudio completo de Tier 2 (dos parámetros, tres valores por
parámetro), ni una prueba independiente de generalización. Antes de la entrega
se necesitará evaluar con otras condiciones iniciales sin usarlas para ajustar.

Antes de gastar más presupuesto en cambios simultáneos, conviene estudiar el
ritmo de copia de la red objetivo con un cambio aislado y revisar las estimaciones
Q durante una prueba corta. La referencia de ese experimento debe declararse
explícitamente; no debe reemplazar ni ocultar los fallos aquí registrados.
