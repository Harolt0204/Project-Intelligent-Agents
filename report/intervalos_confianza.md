# Cómo se calcularon e interpretan los intervalos

## Qué exige la guía

El apartado 6.2 permite **media con desviación estándar o IC del 95%**. Por tanto, las figuras previas con DE ya cumplían ese requisito. Se añaden IC a petición del equipo; esta actualización no requiere entrenar ni generar nuevos episodios. Los pesos, tablas, semillas y retornos originales se conservan.

## Unidad experimental y fórmula

En cada paso de evaluación se calcula primero el retorno medio de los diez episodios de **cada modelo entrenado**. Después se usan las tres medias correspondientes a las semillas 42, 43 y 44. Para entrenamiento, cada réplica aporta su media de episodios completos terminados dentro del bloque de pasos; un episodio cortado por presupuesto no se incluye. En las tablas finales, cada réplica aporta su media de 100 episodios, con igual peso entre semillas.

Con n=3, media m y desviación estándar **muestral** s (ddof=1), se calcula:

**IC 95% = m ± 4.302652729696142 × s / √3.**

El factor corresponde a t de Student con dos grados de libertad, no al factor normal 1.96. Implementación compartida en `experiments/intervalos.py`. Si falta una de las tres réplicas en un punto, no se presenta un IC de tres semillas para ese punto.

Los 300 episodios de tres agentes **no equivalen a 300 entrenamientos independientes**. Tratar todos los episodios como réplicas independientes subestimaría la incertidumbre entre entrenamientos. Las dos muestras nuevas de LunarLander se presentan por separado; no producen seis semillas de entrenamiento.

## Alcance y limitaciones

- Son IC **puntuales** de la media: cada paso se considera por separado. No significan una cobertura conjunta del 95% para toda la curva; los controles sucesivos están correlacionados.
- La interpretación t supone medias por semilla independientes y aproximadamente normales. Con solo tres semillas no se puede comprobar bien esa forma de distribución. Son intervalos orientativos, no una garantía robusta de cobertura en RL.
- Son condicionales al entorno y configuración evaluados. No incorporan toda la incertidumbre de la búsqueda de parámetros ni eliminan el posible optimismo del desarrollo. En LunarLander la selección del checkpoint precedió a ambas muestras nuevas, pero estas evalúan los mismos tres modelos, no nuevas réplicas de entrenamiento.
- **DE e IC son distintos.** Las tablas que dicen “media ± DE” conservan la DE descriptiva previa (ddof=0). Las bandas nuevas se etiquetan explícitamente “IC 95%”. No se reinterpretan cifras históricas de DE como IC.
- No se recortan los extremos para que parezcan más precisos. Un IC simétrico de CartPole puede superar su máximo de 500; ese extremo es una limitación del modelo estadístico, no un retorno observado posible. Cerca del techo y con n=3 conviene priorizar también los valores individuales.
- Que los tres modelos superen el umbral es un hecho descriptivo verificable; no garantiza el éxito de cualquier entrenamiento futuro. Que dos IC se solapen tampoco es por sí solo una prueba de igualdad de algoritmos.

## Comparación emparejada de Tier 3

Para cada semilla se resta DQN a Double DQN en la misma muestra de evaluación. Se calcula el mismo IC t sobre esas **tres diferencias**, manteniendo el emparejamiento. La comparación es exploratoria tras el desarrollo. En la muestra de confirmación el intervalo de la diferencia incluye cero: el resultado no sustenta una afirmación de superioridad general de Double DQN.

## Archivos y reproducción sin entrenamiento

Las figuras de comparación vigentes de CartPole, los cuatro estudios iniciales de LunarLander y las dos curvas de la mejora usan ahora IC. Las barras individuales por semilla se conservan como observaciones, sin confundirlas con una media entre semillas. La figura adicional `results/analisis_intervalos/resultados_ic95.png` muestra IC de las medias finales; las cifras exactas y diferencias emparejadas están en su `summary.json` y `report.md`.

Los comandos completos están en el bloque “Análisis teórico e intervalos” del README. `curves.csv` de Tier 1 y los archivos `*_intervals.csv` de LunarLander guardan los extremos de cada punto. Los tres cuadernos de trabajo del usuario se conservan; `Analisis_Teorico_e_Intervalos.ipynb` reúne el texto y las figuras nuevas sin ejecutar entrenamiento.

## Referencia estadística

[NIST/SEMATECH, Confidence Limits for the Mean](https://www.itl.nist.gov/div898/handbook/eda/section3/eda352.htm): fórmula t y significado frecuentista de cobertura. La fórmula procede de esta referencia; su uso condicionado a tres semillas, la unidad experimental y las limitaciones anteriores son decisiones metodológicas explícitas de este proyecto.
