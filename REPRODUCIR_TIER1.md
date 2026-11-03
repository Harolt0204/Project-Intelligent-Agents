# Reproducir y revisar Tier 1 en otra computadora

Este repositorio contiene el Tier 1 experimental completo: código propio,
configuraciones, registros de todas las semillas, modelos entrenados, figuras,
animaciones y discusión. Tier 2, Tier 3 y el artículo final siguen pendientes.
El entorno comprobado es **Windows con Python 3.12 de 64 bits, CPU**.
No hace falta GPU. La instalación necesita Internet para descargar dependencias.

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
- **2:** Q-learning de 10 intervalos, ejecución 44.
- **3:** DQN MSE, ejecución exitosa 44.
- **4:** DQN MSE, ejecución 42, para mostrar variación.
- **5:** gráfica comparativa.
- **6:** resultados resumidos.

La opción 3 no representa todos los entrenamientos: el retorno medio final de
DQN MSE fue 234.10; una semilla logró 500 y las otras dos quedaron cerca de 101.
Q-learning de 10 intervalos obtuvo la mejor media tabular, 359.70.
La guía de explicación está en `COMO_MOSTRAR_TIER1.md`.

Comandos equivalentes:

```powershell
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent tabular --episodes 1
.\.venv\Scripts\python.exe -m experiments.mostrar_tier1 --agent dqn --episodes 1
```

## 4. Leer resultados y reconstruir figuras

- `report/cierre_tier1.md`: protocolo, métricas, evaluación adicional y límites.
- `results/parte_06_bins`: tres granularidades × tres semillas.
- `results/parte_10_dqn`: DQN inicial, incluidos sus fallos.
- `results/parte_15_mse_completo`: DQN MSE completo y sus modelos finales.
- `report/uso_ia.md`: registro del apoyo de IA.
- Las demás carpetas `parte_*` conservan los ejercicios y experimentos exploratorios.

Regenerar las figuras desde los registros incluidos, sin entrenar:

```powershell
.\.venv\Scripts\python.exe -m experiments.comparar_bins --plot-only
.\.venv\Scripts\python.exe -m experiments.comparar_tier1 --dqn-root results/parte_15_mse_completo --dqn-label "DQN (MSE)" --extra-dqn-root results/parte_10_dqn --extra-dqn-label "DQN (Smooth L1 inicial)"
```

Comprobar los componentes:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Opcional: reproducir la evaluación adicional con las mismas condiciones nuevas,
guardando otra copia para conservar los resultados originales:

```powershell
.\.venv\Scripts\python.exe -m experiments.evaluar_tier1_nuevas_semillas --output results/reproduccion_evaluacion_nueva
```

## 5. Repetir todos los entrenamientos principales

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
