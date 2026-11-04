# Cómo ver y presentar Tier 1

## Abrir las simulaciones

En esta carpeta, haz doble clic en **Mostrar_Tier1.bat**. Escribe el número de
la opción y pulsa Enter. Las simulaciones abren una ventana de CartPole y usan
los modelos guardados: no hace falta volver a entrenarlos.

1. **Acciones al azar:** muestra el punto de partida sin aprendizaje.
2. **Q-learning entrenado:** usa la granularidad indicada en `results/tier1_actual.json`, semilla 42.
3. **DQN actual entrenado:** usa el modelo actual de la semilla 42.
4. **DQN anterior:** usa el DQN MSE previo a la mejora, semilla 42, para comparar.
5. **Gráfica:** abre las curvas de entrenamiento y evaluación.
6. **Resumen:** muestra en la consola los resultados finales de los cinco métodos.

Cada simulación muestra tres intentos. Al terminar vuelve al menú. Puedes
interrumpir con Ctrl+C en la consola; cerrar la ventana de la simulación también
puede finalizar el programa. Si necesitas volver a entrar, abre el archivo otra vez.
La primera carga de DQN puede tardar mientras se abre PyTorch.

## Qué explicar mientras lo muestras

El carrito intenta mantener el poste vertical. Solo puede empujar a izquierda
o derecha. Recibe un punto por paso y el intento dura como máximo 500 pasos.

- Q-learning agrupa las observaciones en intervalos y consulta una tabla.
- DQN recibe los cuatro números continuos y usa una red para puntuar acciones.
- En esta demostración ambos usan lo aprendido sin modificar tabla ni red.
- El menú imprime la media de 100 episodios del modelo mostrado y la media de
  las tres semillas. Los resultados actuales se indican en `results/tier1_actual.json`.
- La opción 4 conserva un modelo anterior cuyo promedio fue cercano a 102;
  permite explicar por qué investigamos una configuración más estable.

## Orden sugerido para una demostración corta

1. Muestra acciones aleatorias para explicar el problema.
2. Muestra Q-learning y explica sus casillas.
3. Muestra el DQN actual y explica sus dos puntuaciones de acción.
4. Abre la gráfica y explica que los promedios incluyen las tres semillas.
5. Muestra el DQN anterior para explicar la mejora; consulta las tres semillas
   actuales en el notebook y las tablas para valorar su consistencia.

Los números que aparecen al terminar cada intento son retornos de una sola
demostración. La evidencia experimental está en las evaluaciones de 100 episodios.

## Comandos alternativos

Abre una terminal dentro de Project 01 y ejecuta una opción:

```powershell
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent random
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent tabular
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent dqn
```

Para otra semilla, añade `--seed 43` o `--seed 44`. Para el modelo anterior,
añade `--original`. Para un solo intento, añade
`--episodes 1`. Los archivos de modelos y resultados se conservan.

También puedes abrir `Tier1_Explicado.ipynb` en VS Code y activar `VER_CARRITO`:
la animación se muestra dentro del notebook, junto con parámetros y explicaciones.
