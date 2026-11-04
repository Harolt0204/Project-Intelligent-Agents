# Búsqueda de mejora para Q-learning

Cada fila incluye los tres entrenamientos 42/43/44, 200000 pasos por agente y
evaluación final de 100 episodios. Se conserva el modelo final, sin elegir el mejor checkpoint.
No se modificaron las recompensas de CartPole ni la fórmula Q-learning.

| Diseño | Intervalos por variable | Q inicial | Alpha inicio → final | Retorno medio ± DE | Semillas ≥475 |
|---|---|---:|---|---:|---:|
| Anterior | 6/6/6/6 | 0 | 0.1 → 0.1 | 414.48 ± 62.85 | 1/3 |
| Anterior | 10/10/10/10 | 0 | 0.1 → 0.1 | 89.09 ± 5.48 | 0/3 |
| Anterior | 20/20/20/20 | 0 | 0.1 → 0.1 | 26.28 ± 2.40 | 0/3 |
| optimista | 6/6/6/6 | 100 | 0.1 → 0.1 | 402.40 ± 63.37 | 1/3 |
| optimista | 10/10/10/10 | 100 | 0.1 → 0.1 | 256.87 ± 83.98 | 0/3 |
| optimista | 20/20/20/20 | 100 | 0.1 → 0.1 | 99.26 ± 2.91 | 0/3 |
| centrado | 6/6/6/6 | 100 | 0.1 → 0.01 | 260.31 ± 39.04 | 0/3 |
| resolucion | 2/2/6/6 | 100 | 0.1 → 0.1 | 477.64 ± 31.62 | 2/3 |
| resolucion | 2/2/10/10 | 100 | 0.1 → 0.1 | 451.19 ± 35.11 | 1/3 |
| resolucion | 2/2/20/20 | 100 | 0.1 → 0.1 | 178.41 ± 43.14 | 0/3 |
| resolucion_suave | 2/2/6/6 | 100 | 0.1 → 0.01 | 495.73 ± 6.04 | 3/3 |
| resolucion_suave | 2/2/10/10 | 100 | 0.1 → 0.01 | 163.84 ± 43.19 | 0/3 |
| resolucion_suave | 2/2/20/20 | 100 | 0.1 → 0.01 | 170.21 ± 24.87 | 0/3 |

## Qué se probó

1. **Optimista:** cambia solamente Q inicial, de 0 a 100, frente al diseño anterior.
   Con gamma=0.99, 100 es el límite superior del retorno descontado con recompensa ≤1.
   Es una estimación inicial, no una recompensa añadida ni una política correcta precargada.
2. **Centrado:** combina intervalos más finos cerca de cero (potencia 2) y alpha decreciente.
   Esta combinación exploratoria no permite atribuir el resultado a un único factor.
3. **Resolución por variable:** respecto al control optimista, conserva sus parámetros y
   cambia la resolución de posición y velocidad del carrito a 2 intervalos cada una.
   Mantiene 6, 10 o 20 para ángulo y velocidad angular. No se elimina ninguna observación.
4. **Resolución suave:** conserva la tabla compacta e inicialización optimista y reduce alpha
   linealmente de 0.1 a 0.01 durante los 200000 pasos. Frente al control resolución,
   este ensayo modifica el calendario de alpha para investigar estabilidad al final.

## Por qué reducir casillas puede ayudar

Una tabla uniforme de 20 intervalos por variable contiene 20⁴=160000 estados.
La tabla 2×2×20×20 contiene 1600: cien veces menos, conservando detalle angular.
Esto permite reutilizar más experiencias por casilla. A cambio, pierde detalle de posición
y velocidad: el resultado no garantiza que esa simplificación sea adecuada en cualquier entorno.
El enunciado exige tres granularidades; también se conservan los estudios anteriores con
6/10/20 intervalos en cada una de las cuatro variables.

## Diagnóstico y límites

La evaluación guardada de las semillas débiles del diseño anterior terminó por inclinación
de la vara. Los detalles y retornos reproducidos están en diagnostico_fallos.json.
Esto orienta la búsqueda, sin probar una causa única de los fallos.
Todas estas semillas se usaron en el desarrollo: no son un test independiente de selección.
La evaluación adicional se fija antes de medirla y no se usa para ajustar la configuración elegida.
Los resultados parciales que subieron a 500 y luego bajaron se conservan en los CSV;
el desempeño final se informa aunque sea peor que un control anterior.

## Reproducir los intentos

Cada comando usa una carpeta que no debe existir; no reemplaza los resultados entregados.

```powershell
.\.venv\Scripts\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_optimista.json --metodo tabular --output results/repetir_optimista
.\.venv\Scripts\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_centrado.json --metodo tabular --output results/repetir_centrado
.\.venv\Scripts\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_resolucion.json --metodo tabular --output results/repetir_resolucion
.\.venv\Scripts\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_resolucion_suave.json --metodo tabular --output results/repetir_resolucion_suave
.\.venv\Scripts\python.exe -m experiments.resumir_busqueda_qlearning
```
