# Cómo ver y presentar Tier 1

## Abrir las simulaciones

En esta carpeta, haz doble clic en **Mostrar_Tier1.bat**. Escribe el número de
la opción y pulsa Enter. Las simulaciones abren una ventana de CartPole y usan
los modelos guardados: no hace falta volver a entrenarlos.

1. **Acciones al azar:** muestra el punto de partida sin aprendizaje.
2. **Q-learning entrenado:** usa la tabla de 10 intervalos de la ejecución 44.
3. **DQN entrenado:** usa la ejecución exitosa 44, con pérdida MSE.
4. **Otro DQN:** usa la ejecución 42 para mostrar que los entrenamientos varían.
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
- La ejecución DQN 44 resolvió 100 intentos y otros 100 con inicios nuevos.
  Las otras dos ejecuciones quedaron cerca de 101; el promedio de las tres fue
  234.1. Mostrar el modelo exitoso no significa que todos los DQN fueran igual.

## Orden sugerido para una demostración corta

1. Muestra acciones aleatorias para explicar el problema.
2. Muestra Q-learning y explica sus casillas.
3. Muestra el DQN exitoso y explica sus dos puntuaciones de acción.
4. Abre la gráfica y explica que los promedios incluyen las tres semillas.
5. Si preguntan por estabilidad, muestra el otro DQN y los resultados completos.

Los números que aparecen al terminar cada intento son retornos de una sola
demostración. La evidencia experimental está en las evaluaciones de 100 episodios.

## Comandos alternativos

Abre una terminal dentro de Project 01 y ejecuta una opción:

```powershell
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent random
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent tabular
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent dqn
```

Para el DQN de otra ejecución, añade `--seed 42`. Para un solo intento, añade
`--episodes 1`. Los archivos de modelos y resultados se conservan.
