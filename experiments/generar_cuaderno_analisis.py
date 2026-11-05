"""Crear el cuaderno complementario sin alterar los tres notebooks de trabajo."""

from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parents[1]


def main():
    md=nbf.v4.new_markdown_cell
    code=nbf.v4.new_code_cell
    cells=[md('''# Análisis teórico e intervalos de confianza

Este cuaderno complementa los tres Tiers sin entrenar agentes. Los tres cuadernos
de trabajo se conservan. Ejecuta las celdas en orden con el kernel del proyecto.
El análisis de dos páginas está también en [PDF](output/pdf/analisis_teorico.pdf).
Las bandas históricas de desviación estándar eran válidas según la guía; aquí
se añaden IC del 95% y se explican sus supuestos.'''),code('''from pathlib import Path
import sys
from IPython.display import display, Markdown, Image
ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents)
            if (p / 'report/analisis_teorico.md').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
def texto(path):
    display(Markdown((ROOT / path).read_text(encoding='utf-8')))
def imagen(path):
    display(Image(filename=str(ROOT / path)))
print('Solo lectura de informes y figuras; sin entrenamiento.')'''),
        code("texto('report/analisis_teorico.md')"),
        md('## Cómo leer la incertidumbre'),code("texto('report/intervalos_confianza.md')"),
        md('## Cálculo visible: tres semillas, no 300 entrenamientos'),
        code('''import inspect, json
from IPython.display import Code
from experiments.intervalos import mean_ci95
display(Code(inspect.getsource(mean_ci95), language='python'))
datos = json.loads((ROOT / 'results/analisis_intervalos/summary.json').read_text(encoding='utf-8'))
ejemplo = next(r for r in datos['rows'] if r['tier'] == 2 and r['sample'] == 'test')
print('DQN LunarLander, medias individuales:', ejemplo['values'])
media, inferior, superior = mean_ci95(ejemplo['values'])
print(f'Media: {media:.2f}; IC 95% orientativo: [{inferior:.2f}, {superior:.2f}]')'''),
        code("texto('results/analisis_intervalos/report.md')\nimagen('results/analisis_intervalos/resultados_ic95.png')"),
        md('## Tier 1: mismo presupuesto, tres granularidades y DQN'),
        code("imagen('results/parte_18_qlearning/comparacion/tier1_comparacion.png')"),
        md('## Estudios iniciales: hiperparámetros y ablación de Tier 2; comparación de Tier 3'),
        code("for nombre in ('learning_rate', 'target_sync_every', 'ablation', 'double_dqn'):\n    imagen(f'results/tier23_lunarlander/figures/{nombre}.png')"),
        md('## Tier 2 y Tier 3: comparación mejorada con el mismo presupuesto'),
        code("imagen('results/tier23_mejora_v1/figures/learning.png')\nimagen('results/tier23_mejora_v1/figures/tests.png')"),
        md('''## Qué concluye el experimento

Los seis modelos de LunarLander superan 200 individualmente en las dos muestras.
Double DQN tiene una media algo mayor, pero la confirmación no permite afirmar
una superioridad general: el IC de sus tres diferencias emparejadas incluye cero.
Tres semillas aportan poca información sobre la distribución entre entrenamientos.
El análisis teórico y estas figuras están preparados para integrar en el artículo;
el artículo completo y la defensa oral son entregables separados.''')]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3 (proyecto)','language':'python','name':'python3'},
        'language_info':{'name':'python'}})
    nbf.write(nb,ROOT/'Analisis_Teorico_e_Intervalos.ipynb')
    print('Cuaderno complementario generado; originales conservados.')


if __name__=='__main__': main()
