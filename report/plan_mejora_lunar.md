# Nueva mejora de LunarLander — 4 de noviembre de 2026

Objetivo solicitado: que DQN y Double DQN alcancen retorno medio ≥200 en cada
una de las semillas 42, 43 y 44. No basta con una media agregada favorable.

## Diagnóstico previo

El DQN con lr=0.001 obtuvo 165.83, 251.59 y 260.70. La semilla 42 había
confirmado más de 200 antes de retroceder. Double obtuvo 227.71, 5.26 y
135.60; su semilla 43 agotó el límite de tiempo en 100/100 episodios.
Los datos originales y los estudios de dos parámetros y ablación se conservan
en results/tier23_lunarlander. No se sustituyen ni se borran sus resultados.

## Nueva ronda común, desde cero

- Dos agentes propios: DQN y Double DQN, con semillas 42/43/44.
- Un millón de pasos por modelo, con los mismos ajustes en ambos métodos.
- Red 8→128→128→4, Adam, Smooth L1, gamma=0.99, replay de 100000.
- Lote de 128; un ajuste cada cuatro pasos desde el paso 5000.
- Copia de target cada 1000 pasos: aproximadamente 250 ajustes por copia.
- Epsilon baja de 1 a 0.05 en 200000 pasos y luego se mantiene.
- Tasa inicial 0.00063 hasta 300000 pasos; disminuye linealmente hasta 0.00005
  a los 800000 y permanece así hasta terminar.
- Las recompensas, la física y el límite de 1000 pasos del entorno no cambian.

La tasa inicial y el lote se orientan por la
[referencia pública de RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml).
Nuestra arquitectura, frecuencia de ajustes y calendario son decisiones propias;
no se importa un agente de esa biblioteca ni se afirma replicar su experimento.

## Selección y comprobación

1. Validar cada 50000 pasos con diez episodios greedy, base de reset 30000000.
2. Si el control alcanza 200, ampliarlo a 100 episodios. Hacer también un control
   de 100 al finalizar el presupuesto.
3. Conservar el checkpoint con la mayor media confirmada de validación. Guardar
   aparte los pesos del último paso, aunque sean peores. La selección es la misma
   para todos los modelos y no utiliza ninguna de las pruebas finales.
4. Reproducir desde el archivo elegido los 100 retornos de validación.
5. Antes de abrir las pruebas, pedir un margen de validación ≥230 en todos los
   modelos. Si falta, se detiene la ronda en desarrollo y se revisa sin consultar
   las muestras finales. Los casos difíciles se ejecutan primero para medirlo.
6. Fijar las huellas de los seis modelos seleccionados. Evaluar después 100
   episodios por modelo con base 40000000 y otros 100 con base 50000000.
   Las semillas de reset son base + 1000×semilla de entrenamiento + episodio.

La meta de esta ronda es 3/3 por método en **ambas** muestras nuevas. Son
1200 episodios finales, distintos de validación y de las muestras anteriores.
No se cambia una semilla ni se elige otro checkpoint después de mirar la prueba.
Si una prueba falla, se informa como tal; no se afirma que el objetivo se logró.

## Interpretación y coste

Los estudios originales de 500000 pasos siguen cumpliendo su función de variar
un parámetro a la vez y hacer una ablación. La nueva ronda es una confirmación
adicional: cambia conjuntamente varios ajustes y el presupuesto. No permite
atribuir una mejora a un cambio aislado. La comparación nueva entre DQN y Double
sí comparte presupuesto, ajustes, selección y condiciones de evaluación.

Los procesos tienen un hilo PyTorch y comparten la CPU, hasta tres a la vez.
Los tiempos se registran, sin tratarlos como un benchmark aislado. Esta ronda
añade coste al presupuesto orientativo del proyecto; no se presenta como parte
de las ejecuciones originales ni se oculta ese coste.

Configuración congelada: configs/tier23_mejora_v1.json.
Ejecución: `python -m experiments.mejorar_tier23`.
Los entrenamientos pueden tardar varias horas de CPU. El estado se consulta en
results/tier23_mejora_v1/progress.json sin volver a entrenar.
