"""Consultar la nueva confirmación sin iniciar entrenamientos."""

from html import escape
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REFINEMENT=ROOT/'results/tier23_mejora_v1'


def active_root():
    """La presentación cambia de modelos solo después de revisar una mejora completa."""
    original=ROOT/'results/tier23_lunarlander'
    pointer=ROOT/'results/tier23_actual.json'
    try:
        data=json.loads(pointer.read_text(encoding='utf-8'))
        candidate=(ROOT/data['root']).resolve()
        if not candidate.is_relative_to(ROOT/'results') or not data['reviewed']:
            return original
        summary=json.loads((candidate/'summary.json').read_text(encoding='utf-8'))
        return candidate if summary['complete'] and summary['goal_met'] else original
    except (OSError,ValueError,KeyError):
        return original


def refinement_root():
    active=active_root()
    return active if active!=ROOT/'results/tier23_lunarlander' else REFINEMENT


def refinement_config():
    output=refinement_root()
    path=output/'protocol.json'
    if not path.exists(): path=ROOT/'configs/tier23_mejora_v1.json'
    return json.loads(path.read_text(encoding='utf-8'))


def resumen_mejora(tier=2,output=None):
    from IPython.display import HTML,display
    output=Path(output) if output is not None else refinement_root()
    config=json.loads((output/'protocol.json').read_text(encoding='utf-8')) if (output/'protocol.json').exists() else refinement_config()
    groups=['base'] if tier==2 else ['base','double']
    rows=[]
    progress={}
    try:
        progress={(r['group'],r['seed']):r for r in json.loads((output/'progress.json').read_text(encoding='utf-8'))}
    except (OSError,ValueError): pass
    for group in groups:
        for seed in config['seeds']:
            folder=output/group/f'seed_{seed}'
            row={'group':group,'seed':seed,'state':'pendiente'}
            try:
                if (folder/'summary.json').exists():
                    run=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                    row.update(run,state='probado' if run['complete'] else 'validación lista')
                else:
                    state=progress.get((group,seed),{})
                    row.update(state='entrenando' if state else 'pendiente',
                               training_steps=state.get('steps',0),selected_validation_mean=state.get('best_validation_100'))
            except (OSError,ValueError): row['state']='actualizando'
            rows.append(row)
    def value(row,key):
        v=row.get(key)
        return '—' if v is None else (f'{v:.2f}' if isinstance(v,float) else str(v))
    body=''.join('<tr>'+''.join(f'<td>{escape(value(row,key))}</td>' for key in
        ('group','seed','state','selected_step','selected_validation_mean','test_mean','confirmation_mean'))+'</tr>' for row in rows)
    display(HTML('<table><tr><th>Método</th><th>Semilla</th><th>Estado</th><th>Checkpoint</th>'
        '<th>Validación 100</th><th>Prueba 100</th><th>Confirmación 100</th></tr>'+body+'</table>'))
    complete=[r for r in rows if r.get('complete',False)]
    print(f'{len(complete)}/{len(rows)} modelos con ambas pruebas terminadas. base=DQN; double=Double DQN.')
    if len(complete)==len(rows):
        for group in groups:
            selected=[r for r in complete if r['group']==group]
            print(f"{group}: {sum(r['threshold_met_both_tests'] for r in selected)}/3 semillas ≥200 en ambas pruebas.")
    else:
        print('Las puntuaciones de validación no sustituyen las pruebas con condiciones nuevas.')
    return rows


def figura_mejora(name,output=None):
    from IPython.display import Image,display
    output=Path(output) if output is not None else refinement_root()
    path=output/'figures'/f'{name}.png'
    if path.exists(): display(Image(filename=str(path)))
    else: print(f'Figura {name}: pendiente de completar y comprobar los modelos.')


def discusion_mejora(output=None):
    from IPython.display import Markdown,display
    output=Path(output) if output is not None else refinement_root()
    path=output/'report.md'
    if path.exists(): display(Markdown(path.read_text(encoding='utf-8')))
    else: print('La discusión final se calcula cuando termine la ronda; no se adelantan resultados.')
