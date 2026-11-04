# Tier 2 y Tier 3 en notebooks separados

Abre la carpeta completa del proyecto en VS Code. Conserva **Tier1_Explicado.ipynb**
para CartPole y usa:

- **Tier2_LunarLander.ipynb:** entorno, DQN, parámetros, dos estudios y ablación.
- **Tier3_DoubleDQN.ipynb:** cambio matemático, ejemplo numérico y comparación con DQN.

Selecciona el kernel **.venv\Scripts\python.exe** en cada notebook. Ejecuta las
celdas en orden con **Shift+Enter**. Tras actualizar código, reinicia el kernel.
Los notebooks usan los mismos agentes de `agents/`; no mantienen implementaciones duplicadas.

En otra computadora, con Python 3.12, abre **Preparar_Proyecto.bat** y después
**Preparar_Notebook.bat**. Box2D está incluido en los requisitos para LunarLander-v3.
El entorno se verificó en Windows, CPU, con Gymnasium 1.4.0 y Box2D 2.3.10.

## Consultar resultados y ver la nave

Ejecutar todo muestra ejemplos y los resultados disponibles. Si falta completar un
grupo de tres semillas, se indica como pendiente; no se inventan puntuaciones.

Para animar la nave, cambia **VER_NAVE=False** a **True** en su celda. Tier 2 usa
el modelo `base` y Tier 3 el modelo `double`. Puedes elegir semilla 42, 43 o 44.
La animación de un episodio no sustituye el promedio de 100 episodios por modelo.

También puedes abrir **Mostrar_LunarLander.bat** o **Mostrar_DoubleDQN.bat** para
ver una ventana independiente cuando los modelos estén disponibles.

## Ejecutar entrenamientos

Los indicadores **EJECUTAR_PRUEBA**, **EJECUTAR_TIER2** y **EJECUTAR_TIER3** están
desactivados inicialmente. La prueba corta es para comprobar funcionamiento y no
es un resultado formal. Las ejecuciones formales usan tres semillas y pueden tardar.

Cambiar CONFIG afecta nuevos entrenamientos. No modifica modelos ya guardados.
Para cambiar parámetros usa una carpeta nueva. Los resultados completos del mismo
protocolo pueden reutilizarse, especialmente el DQN base compartido entre Tiers.
Las ejecuciones parciales no se sobrescriben.

## Reproducción desde terminal

Ejecutar todos los estudios en orden, reutilizando resultados completos del mismo protocolo:

```powershell
.\.venv\Scripts\python.exe -m experiments.paralelo_tier23
```

Cada tratamiento ejecuta sus tres semillas en procesos independientes, con un hilo
PyTorch por proceso. Los parámetros y pasos no cambian. La alternativa secuencial
es `python -m experiments.ejecutar_tier23` usando el Python de `.venv`.

Solo Tier 2 o Tier 3:

```powershell
.\.venv\Scripts\python.exe -m experiments.ejecutar_tier23 --tier 2
.\.venv\Scripts\python.exe -m experiments.ejecutar_tier23 --tier 3
```

Repetir todo en una carpeta nueva:

```powershell
.\.venv\Scripts\python.exe -m experiments.paralelo_tier23 --output results/reproduccion_tier23
.\.venv\Scripts\python.exe -m experiments.analizar_tier23 --output results/reproduccion_tier23
.\.venv\Scripts\python.exe -m experiments.verificar_tier23 --root results/reproduccion_tier23
```

Regenerar todas las figuras desde los registros incluidos, sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.analizar_tier23
```

El comando genera `learning_rate`, `target_sync_every`, `ablation` y `double_dqn`,
cada figura en PNG y PDF. Solo genera una comparación cuando están completas sus tres semillas.

Comprobar modelos finales y la muestra adicional fijada en el protocolo, sin ajustar parámetros:

```powershell
.\.venv\Scripts\python.exe -m experiments.verificar_tier23
```

La carpeta de verificación debe ser nueva. Para repetir la misma muestra guardando otra copia:

```powershell
.\.venv\Scripts\python.exe -m experiments.verificar_tier23 --output results/reproduccion_verificacion_tier23
```

No es una muestra nueva distinta: repite las condiciones documentadas para comprobar reproducción.
Se comprueban base y Double DQN, además de la variante DQN de mayor media de
desarrollo de Tier 2. Esta última se selecciona después de observar desarrollo;
las condiciones iniciales adicionales estaban fijadas antes de entrenar. No se
usan sus resultados para volver a elegir parámetros o semillas.

Generar los GIF del primer episodio de las semillas 42:

```powershell
.\.venv\Scripts\python.exe -m experiments.mostrar_lunarlander --group base --gif
.\.venv\Scripts\python.exe -m experiments.mostrar_lunarlander --group double --gif
```

Comprobar los notebooks con kernel real:

```powershell
.\.venv\Scripts\python.exe -m experiments.verificar_notebooks_lunar
.\.venv\Scripts\python.exe -m experiments.verificar_notebooks_lunar --animacion 2
.\.venv\Scripts\python.exe -m experiments.verificar_notebooks_lunar --animacion 3
```

Los datos están en `results/tier23_lunarlander`. Los directorios `tier23_piloto_*`
y archivos de benchmark contienen mediciones de coste y pruebas parciales; no se
presentan como resultados formales de tres semillas.

El campo numérico `steps=500000` fija el presupuesto formal. El texto `selection`
del protocolo congelado arrastra una errata de un piloto que dice 200000; los
registros y modelos formales usan 500000. Se conserva el archivo histórico para
mantener sus huellas de reproducción. `diagnostics.json` permite revisar pérdidas
y regresiones observadas; una pérdida pequeña no demuestra un buen aterrizaje.
