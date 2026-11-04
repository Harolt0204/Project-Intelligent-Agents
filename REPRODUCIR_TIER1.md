# Reproducir y revisar Tier 1 en otra computadora

Este repositorio contiene el Tier 1 experimental completo: código propio,
configuraciones, registros de todas las semillas, modelos entrenados, figuras,
animaciones y discusión. Tier 2, Tier 3 y el artículo final siguen pendientes.
El entorno comprobado es **Windows con Python 3.12 de 64 bits, CPU**.
No hace falta GPU. La instalación necesita Internet para descargar dependencias.

## Notebook para estudiar y ejecutar en VS Code

Abre **Tier1_Explicado.ipynb**, selecciona el kernel `.venv\Scripts\python.exe`
y recorre las celdas con Shift+Enter. En otra computadora, después de preparar
el proyecto, abre **Preparar_Notebook.bat** una vez. Instrucciones ampliadas en
[COMO_USAR_NOTEBOOK.md](COMO_USAR_NOTEBOOK.md).

El notebook permite editar los parámetros, mostrar el código real, entrenar en
carpetas nuevas, revisar resultados y ver una animación dentro del mismo archivo.
Los entrenamientos y la animación se activan con sus indicadores True/False.

## 1. Obtener el proyecto

```powershell
git clone https://github.com/Harolt0204/Project-Intelligent-Agents.git
cd Project-Intelligent-Agents
```

También puedes usar **Code → Download ZIP** en GitHub y extraerlo. Todos los
comandos siguientes se ejecutan dentro de la carpeta descargada, donde está
`requirements.txt`. La carpeta `.venv` no se comparte: cada computadora crea la suya.

## 2. Preparar Python una sola vez

Instala Python 3.12 de 64 bits con el lanzador `py`. Después haz doble clic en
**Preparar_Proyecto.bat**. Crea el entorno, instala las versiones del proyecto
y comprueba que pueda cargar ambos agentes sin entrenarlos.

Alternativa desde PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m experiments.verificar_instalacion
```

No hay que activar el entorno ni cambiar la política de PowerShell.
PyTorch para CPU se obtiene desde el índice oficial indicado en requirements.
En Linux se pueden adaptar los comandos al Python de `.venv/bin/python`, pero
esa plataforma no se validó aquí. El archivo fijado de PyTorch `+cpu` está
preparado para Windows/Linux; macOS requiere adaptar esa dependencia y no se
presenta como una instalación comprobada.

## 3. Ver los agentes sin volver a entrenar

Abre **Mostrar_Tier1.bat** y elige:

- **1:** acciones aleatorias.
- **2:** Q-learning actual de 2/2/6/6 intervalos, semilla 42.
- **3:** DQN ajustado actual, semilla 42.
- **4:** DQN MSE anterior, semilla 42, para comparar antes y después.
- **5:** gráfica comparativa.
- **6:** resultados resumidos.

La demostración es de un modelo individual. Q-learning obtuvo 495.73 de media
entre tres semillas y DQN 498.60; los seis modelos seleccionados superan 475.
La evaluación adicional de nuevos inicios mantuvo ese resultado en las tres semillas
de cada método. El estudio tabular también conserva las variantes de 10 y 20
intervalos angulares, que no alcanzaron el objetivo.
Los resultados actuales están en report/qlearning_objetivo.md.
La guía de explicación está en `COMO_MOSTRAR_TIER1.md`.

Comandos equivalentes:

```powershell
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent tabular --episodes 1
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent dqn --episodes 1
```

## 4. Leer resultados y reconstruir figuras

- `report/qlearning_objetivo.md`: protocolo, métricas y límites actuales.
- `report/busqueda_qlearning.md`: todos los intentos tabulares, incluidos los negativos.
- `results/parte_18_qlearning`: modelos tabulares actuales, comparación y evaluación nueva.
- `report/mejora_tier1.md`: cierre anterior de la mejora DQN.
- `results/parte_16_estabilidad`: DQN ajustado, comparación, GIF y evaluación nueva.
- `results/parte_17_tabular_ajustado`: tres granularidades × tres semillas anteriores.
- `report/cierre_tier1.md`: cierre anterior conservado como historial.
- `results/parte_06_bins`: tres granularidades × tres semillas.
- `results/parte_10_dqn`: DQN inicial, incluidos sus fallos.
- `results/parte_15_mse_completo`: DQN MSE completo y sus modelos finales.
- `report/uso_ia.md`: registro del apoyo de IA.
- Las demás carpetas `parte_*` conservan los ejercicios y experimentos exploratorios.

Regenerar la figura actual desde los registros incluidos, sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --tabular-root results/parte_18_qlearning/resolucion_suave --dqn-root results/parte_16_estabilidad/dqn_referencia --dqn-label "DQN ajustado" --output results/parte_18_qlearning/comparacion
```

Regenerar las figuras anteriores desde los registros, sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_bins --plot-only
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --dqn-root results/parte_15_mse_completo --dqn-label "DQN (MSE)" --extra-dqn-root results/parte_10_dqn --extra-dqn-label "DQN (Smooth L1 inicial)"
```

Comprobar los componentes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Opcional: reproducir la evaluación adicional anterior (etapa 15) con las mismas condiciones,
guardando otra copia para conservar los resultados originales:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluar_tier1_nuevas_semillas --output results/reproduccion_evaluacion_nueva
```

## 5. Repetir la mejora del Tier 1

Los parámetros compartidos por el notebook están en `configs/tier1_mejorado.json`.
Posición y velocidad usan 2 intervalos; ángulo y velocidad angular usan 6, 10 o 20.
La inicialización es Q=100 y alpha decrece de 0.1 a 0.01. La comparación anterior
con 6, 10 y 20 intervalos en las cuatro variables permanece en el historial.
Para repetir el entrenamiento completo (tres granularidades y DQN, tres semillas cada
uno, 200000 pasos por modelo), usa una carpeta que todavía no exista:

```powershell
.\.venv\Scripts\python.exe -m experiments.cuaderno_tier1 --output results/reproduccion_mejora
```

Para generar la figura comparativa de esa reproducción:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --tabular-root results/reproduccion_mejora --dqn-root results/reproduccion_mejora/dqn --dqn-label "DQN ajustado"
```

Para comprobar exactamente los retornos de los modelos finales y evaluar la muestra
adicional documentada, sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.cerrar_mejora_tier1 --tabular-root results/reproduccion_mejora --dqn-root results/reproduccion_mejora/dqn --output results/reproduccion_mejora/evaluacion_nueva --evaluation-base 5000000
```

Esta evaluación tiene su propio informe derivado en `evaluacion_nueva/report.md`.
No reemplaza el informe de la versión entregada. Repetir los inicios documentados
verifica reproducibilidad; no constituye una muestra independiente nueva.

Los experimentos originales descritos a continuación se conservan como historial.


Verificar el notebook y la animación tabular con el kernel local:

```powershell
.\.venv\Scripts\python.exe -m experiments.verificar_notebook
.\.venv\Scripts\python.exe -m experiments.verificar_notebook --animacion --agente tabular
```

## 6. Repetir los entrenamientos anteriores

Los resultados originales ya existen. Por eso los siguientes comandos usan
carpetas nuevas; los entrenadores rechazan sobrescribir ejecuciones anteriores.
Se vuelven a entrenar tres semillas y 200 000 pasos por configuración.

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_bins --output results/reproduccion_tier1/tabular
.\.venv\Scripts\python.exe -m experiments.entrenar_dqn --loss smooth_l1 --output results/reproduccion_tier1/dqn_inicial
.\.venv\Scripts\python.exe -m experiments.entrenar_dqn --loss mse --output results/reproduccion_tier1/dqn_mse
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --tabular-root results/reproduccion_tier1/tabular --dqn-root results/reproduccion_tier1/dqn_mse --dqn-label "DQN (MSE)" --extra-dqn-root results/reproduccion_tier1/dqn_inicial --extra-dqn-label "DQN (Smooth L1 inicial)"
```

El tiempo depende de la computadora. Las versiones y semillas están fijadas;
no se garantiza identidad numérica entre sistemas o procesadores distintos.
Los registros incluidos permiten revisar las cifras originales sin reentrenar.
Antes de subir nuevas ejecuciones, revisen qué resultados desean añadir a Git.
