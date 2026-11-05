"""Celdas compartidas para mantener una sola entrada por Tier."""


def add_refinement_cells(tier,md,code):
    md('''
## Mejora: confirmación en las tres semillas

Esta ronda añade entrenamiento y busca estabilidad. Conserva los estudios iniciales
de 500000 pasos que aparecen después. La nueva comparación usa **1000000 pasos**
por modelo para DQN y Double DQN, con parámetros y selección iguales.

La tasa de aprendizaje disminuye al final. Se guarda el modelo con mejor **validación
confirmada en 100 episodios**, además del último modelo. Después se comprueba el elegido
en **dos muestras distintas de 100 episodios** que no participan en la selección.
La meta es ≥200 para cada semilla 42, 43 y 44 en ambas muestras, no solo en la media.

Antes entrenábamos cada ocho pasos y copiábamos target cada 2000: unos 250 ajustes
por copia. Ahora entrenamos cada cuatro y copiamos cada 1000, manteniendo esa
relación aproximada. El ranking de un hiperparámetro depende de los demás ajustes.

CONFIG_MEJORA controla nuevos entrenamientos. Los resultados leídos pertenecen al
protocolo guardado; editar parámetros no modifica los modelos existentes.
''')
    code('''
from experiments.cuaderno_mejora import (
    refinement_config, refinement_root, resumen_mejora, figura_mejora, discusion_mejora
)
from experiments.refinar_lunar import learning_rate as lr_mejora, train as entrenar_mejora
CONFIG_MEJORA = refinement_config()
print('Resultados de la mejora:', refinement_root())
print('Pasos:', CONFIG_MEJORA['steps'], '| Semillas:', CONFIG_MEJORA['seeds'])
print('Objetivo de la guía:', CONFIG_MEJORA['threshold'], '| Margen de validación:', CONFIG_MEJORA['validation_gate'])
print(json.dumps(CONFIG_MEJORA['base'], indent=2))
print('Calendario de tasa:', CONFIG_MEJORA['learning_rate_schedule'])
''')
    code('''
codigo(lr_mejora)
for paso in (0, 300000, 550000, 800000, 1000000):
    print('Paso', paso, '→ tasa', lr_mejora(paso, CONFIG_MEJORA))
VER_ENTRENAMIENTO_MEJORA = False
if VER_ENTRENAMIENTO_MEJORA:
    codigo(entrenar_mejora)
''')
    code(f'modelos_mejorados = resumen_mejora(tier={tier})')
    prefix='dqn_' if tier==2 else ''
    code(f"figura_mejora('{prefix}learning')\nfigura_mejora('{prefix}tests')")
    code('discusion_mejora()')
    md('''
### Repetir la nueva comparación (opcional, entrenamiento largo)

La celda está desactivada. Si se activa, ejecuta los seis modelos en una carpeta
nueva con CONFIG_MEJORA. Mantén las semillas 42/43/44 para este protocolo.
Ambos métodos usan el mismo presupuesto. La prueba solo se abre si todos los modelos
alcanzan el margen predefinido de validación; de lo contrario se informa que falta ajustar.
''')
    code('''
EJECUTAR_MEJORA = False
if EJECUTAR_MEJORA:
    from datetime import datetime
    import subprocess
    destino = ROOT / 'results/notebook_lunar' / datetime.now().strftime('mejora_%Y%m%d_%H%M%S_%f')
    destino.mkdir(parents=True, exist_ok=False)
    archivo = destino / 'requested_config.json'
    archivo.write_text(json.dumps(CONFIG_MEJORA, indent=2), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'experiments.mejorar_tier23',
                    '--config', str(archivo), '--output', str(destino)], cwd=ROOT, check=True)
else:
    print('Entrenamiento de mejora desactivado: solo se consultan resultados guardados.')
''')
