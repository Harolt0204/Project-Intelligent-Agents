# Estudiar y ejecutar Tier 1 en VS Code

El análisis teórico y las figuras con intervalos del 95% están reunidos en
**Analisis_Teorico_e_Intervalos.ipynb**, con el mismo kernel. No entrena agentes.
Las figuras externas se actualizaron desde los datos guardados; las salidas ya
guardadas en tu notebook de trabajo pueden seguir mostrando la versión anterior.
Para consultar las nuevas con su explicación, abre el cuaderno complementario.

1. Abre la carpeta completa del proyecto con **Archivo → Abrir carpeta**.
2. Abre **Tier1_Explicado.ipynb**.
3. Si VS Code lo solicita, instala las extensiones **Python** y **Jupyter** de Microsoft.
4. En **Seleccionar kernel**, elige **Entornos de Python → .venv\Scripts\python.exe**.
5. Ejecuta las celdas en orden con **Shift+Enter** o **Ejecutar todo**.

Después de actualizar el proyecto desde GitHub, usa **Reiniciar kernel** y luego
**Ejecutar todo** para cargar las versiones nuevas del código y los parámetros.

En esta computadora las dependencias del notebook ya están instaladas. En otra,
ejecuta primero **Preparar_Proyecto.bat** y después **Preparar_Notebook.bat**.

El notebook muestra explicaciones, el código real de los módulos, ejemplos numéricos,
parámetros editables, tablas, gráficas y la discusión guardada. Para ver la animación
del carrito, busca **VER_CARRITO**, cámbialo a **True** y ejecuta esa celda.
Puedes escoger DQN o tabular, semilla 42/43/44 y granularidad tabular 6/10/20.

Los entrenamientos están desactivados inicialmente. **EJECUTAR_PRUEBA=True** lanza
una prueba corta de una semilla; **EJECUTAR_EXPERIMENTO_COMPLETO=True** repite el
experimento formal con los parámetros visibles. Cambiar parámetros no cambia modelos
guardados. Cada ejecución crea una carpeta nueva dentro de `results/notebook/`.

Los bloques de código mostrados en las explicaciones provienen de los módulos que se
ejecutan. Para modificar la fórmula del algoritmo, edita el `.py` correspondiente y
reinicia el kernel; para modificar parámetros, basta editar CONFIG y ejecutar las
celdas posteriores. Reiniciar el kernel permite volver a recorrer el cuaderno desde cero.

La tabla de resultados y las demostraciones usan `results/tier1_actual.json` para
identificar los modelos entregados. Un entrenamiento nuevo no sustituye ese registro
automáticamente: hay que evaluarlo y documentarlo antes de considerarlo una mejora.
