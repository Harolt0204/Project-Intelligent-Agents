# Prueba corta: Smooth L1 frente a MSE

## Diseño

Se cambió únicamente la función de pérdida. Se reutilizaron los tres controles
Smooth L1 ya verificados de `parte_12_target_corto/target_500`. Los nuevos modelos
MSE se guardaron en `parte_14_loss/mse`. Comparar los archivos de configuración
confirmó que solo difiere `loss`.

Ambas opciones usan 50 000 pasos por semilla, semillas 42/43/44, tasa 0.0005,
red 4-64-64-2, replay 50 000, lote 64, ajuste cada 4 pasos desde el paso 1 000,
target cada 500 y decaimiento de epsilon de 160 000 pasos. El modelo conservado
es el final. No se seleccionó el mejor control ni se redujo el ritmo de
exploración para adaptarlo al presupuesto corto.

Las semillas y reglas de recolección coinciden; no se afirma que todos los
recuerdos sean idénticos, porque una política que cambia produce acciones y
trayectorias diferentes. No se trata de una comparación sobre un dataset fijo.

## Resultado al finalizar 50 000 pasos

Cada agente fue evaluado sin exploración en 100 episodios. DE entre medias de
semillas, no un intervalo de confianza.

| Pérdida | Semilla 42 | Semilla 43 | Semilla 44 | Media ± DE |
|---|---:|---:|---:|---:|
| Smooth L1 | 53.25 | 127.94 | 131.12 | 104.10 ± 35.98 |
| MSE | 260.65 | 203.08 | 270.31 | 244.68 ± 29.68 |
| Diferencia MSE − Smooth L1 | +207.40 | +75.14 | +139.19 | +140.58 |

MSE mejoró el retorno final en las tres semillas de esta prueba. Ninguna alcanzó
475 en la evaluación final. La muestra de tres semillas y la selección secuencial
de parámetros no demuestran una mejora universal ni estabilidad a 200 000 pasos.
No se compara este resultado corto con los finales largos como si tuvieran
igual presupuesto.

## Estimaciones Q y límites de la interpretación

Los máximos Q a 50 000 pasos en el banco fijo de 256 estados fueron:

| Pérdida | Semilla 42 | Semilla 43 | Semilla 44 |
|---|---:|---:|---:|
| Smooth L1 | 79.43 | 72.43 | 75.56 |
| MSE | 123.48 | 128.39 | 146.14 |

Con recompensa máxima 1 y gamma=0.99, el verdadero retorno descontado no puede
exceder 100. Las estimaciones MSE superaron ese límite en parte del banco.
El mayor retorno y la sobreestimación coexisten: mejorar la política a corto
plazo no equivale a obtener valores correctos ni a resolver la inestabilidad.

MSE usa errores al cuadrado; Smooth L1 es cuadrática cerca de cero y lineal
para errores grandes. Se cambian la ponderación y escala del gradiente, en
interacción con Adam y el límite común de norma 10. Las pérdidas numéricas de
ambas funciones no se comparan como si estuvieran en la misma escala. El
experimento muestra el efecto total del cambio, no aísla todos esos mecanismos.

## Verificación

- Pasaron 18 pruebas, incluida una comprobación numérica nueva: predicciones
  cero y objetivos 1/10 deben dar MSE=50.5, conservando los objetivos de Bellman.
- Las configuraciones difieren únicamente en la pérdida.
- Los inicios de evaluación coinciden con los controles reutilizados.
- Cada modelo nuevo registra 50 000 pasos, 12 251 ajustes y 100 copias de target.
- Las pérdidas y parámetros son finitos. Los modelos guardados reprodujeron
  exactamente sus 100 retornos finales.
- Evidencia completa en `results/parte_14_loss/verification.json` y logs por semilla.

## Reproducción

```text
python -m experiments.probar_loss
python -m experiments.probar_loss --plot-only
python -m experiments.ver_dqn --run results/parte_14_loss/mse/seed_42
```

El primer comando exige otra salida si ya hay resultados. Se conservan los
controles anteriores en su ubicación original. La figura PNG/PDF y CSV/JSON
usan 10 episodios por control, con media y DE entre semillas; la tabla final
usa 100. El panel Q muestra cada semilla en escala simétrica logarítmica.
El GIF usa el primer episodio de la primera semilla, sin buscar el mejor intento.

## Siguiente pregunta

El resultado apoya MSE para esta prueba corta, pero la sobreestimación sigue
presente. Una pregunta útil es si copiar target cada 2 000 pasos conserva esta
mejora y contiene el crecimiento de Q. Esa comparación debe partir de MSE cada
500 y cambiar solo el intervalo, no presentar dos cambios simultáneos como uno.
Junto a los controles ya existentes, permitiría estudiar las cuatro combinaciones
de pérdida y frecuencia a igual presupuesto corto antes de decidir un ensayo largo.
