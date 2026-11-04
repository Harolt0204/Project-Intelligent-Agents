"""Comparación emparejada de los modelos elegidos por validación, no por prueba."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def summarize(root):
    root=Path(root)
    config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    groups=[]
    for group in ('base','double'):
        paths=[root/group/f'seed_{seed}/summary.json' for seed in config['seeds']]
        if not all(path.exists() for path in paths):
            print(f'{group}: faltan modelos; no se calcula una media final.')
            continue
        runs=[json.loads(path.read_text(encoding='utf-8')) for path in paths]
        if not all(run.get('complete',False) for run in runs):
            print(f'{group}: pruebas independientes pendientes; no se calcula una media final.')
            continue
        groups.append({'group':group,'runs':runs,
            'test_mean':float(np.mean([r['test_mean'] for r in runs])),
            'test_std_across_seed_means':float(np.std([r['test_mean'] for r in runs])),
            'confirmation_mean':float(np.mean([r['confirmation_mean'] for r in runs])),
            'confirmation_std_across_seed_means':float(np.std([r['confirmation_mean'] for r in runs])),
            'successful_seeds_test':sum(r['test_mean']>=config['threshold'] for r in runs),
            'successful_seeds_confirmation':sum(r['confirmation_mean']>=config['threshold'] for r in runs),
            'successful_seeds_both':sum(r['threshold_met_both_tests'] for r in runs)})
    goal=len(groups)==2 and all(g['successful_seeds_both']==len(config['seeds']) for g in groups)
    data={'complete':len(groups)==2,'goal_met':goal,'groups':groups}
    (root/'summary.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
    lines=['# Mejora de Tier 2 y Tier 3','','## Protocolo y alcance','',
        f"Presupuesto completo: {config['steps']} pasos por modelo; semillas {config['seeds']}. Ambos algoritmos usan los mismos parámetros.",
        f"Parámetros: {config['base']}. Calendario de tasa: {config['learning_rate_schedule']}.",
        'Se guardan los pesos del final del entrenamiento y, por separado, el checkpoint con mayor media de validación confirmada en 100 episodios.',
        f"Validación: base {config['evaluation_base']}; controles de 10 cada {config['eval_every']} pasos. Se amplían a 100 cuando alcanzan 200 y siempre al final.",
        f"Prueba y confirmación: bases {config['heldout_evaluation_base']} y {config['confirmation_evaluation_base']}, 100 episodios por modelo en cada una. No intervienen en la selección.",
        'Los modelos seleccionados se cargan del disco y reproducen exactamente sus 100 retornos de validación antes de probarlos.',
        'Los estudios originales de 500000 pasos y sus resultados negativos se conservan. Esta ronda cambia conjuntamente el presupuesto y varios ajustes; no permite atribuir la mejora a uno solo.',
        'No se cambian recompensas, límite de 1000 pasos del entorno ni semillas de entrenamiento.',
        '',f'Grupos completos: {len(groups)}/2. Objetivo de ambas pruebas, tres semillas por método: {goal}.','',
        '| Método | Semilla | Paso seleccionado | Validación 100 | Prueba 100 | Confirmación 100 |',
        '|---|---:|---:|---:|---:|---:|']
    for group in groups:
        for run in group['runs']:
            lines.append(f"| {group['group']} | {run['seed']} | {run['selected_step']} | {run['selected_validation_mean']:.2f} | {run['test_mean']:.2f} | {run['confirmation_mean']:.2f} |")
    lines += ['','| Método | Media prueba ± DE | Media confirmación ± DE | Semillas ≥200 en ambas |',
              '|---|---:|---:|---:|']
    for group in groups:
        lines.append(f"| {group['group']} | {group['test_mean']:.2f} ± {group['test_std_across_seed_means']:.2f} | {group['confirmation_mean']:.2f} ± {group['confirmation_std_across_seed_means']:.2f} | {group['successful_seeds_both']}/3 |")
    if len(groups)==2:
        paired=[b['test_mean']-a['test_mean'] for a,b in zip(groups[0]['runs'],groups[1]['runs'])]
        lines += ['',f"Double menos DQN en prueba: {groups[1]['test_mean']-groups[0]['test_mean']:+.2f}; diferencias por semilla {paired}.",
                  'Alcanzar 200 en las tres semillas no garantiza éxito para cualquier inicialización futura. La DE es entre medias de semillas (ddof=0).']
        figures(root,config,groups)
    lines += ['','## Tiempo y reproducción','',
              'Tiempos por modelo en summary.json: entrenamiento excluye validación/prueba; total las incluye. Las semillas comparten CPU, un hilo PyTorch por proceso. No es un benchmark aislado.',
              'Reproducir esta ronda: python -m experiments.mejorar_tier23 --output results/reproduccion_mejora',
              'Regenerar figuras sin entrenar: python -m experiments.resumir_mejora --root results/tier23_mejora_v1',
              'Los modelos incluyen versiones de paquetes, hardware y huellas de código en config.json.',
              'Referencia de parámetros: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml','']
    (root/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Resumen generado; objetivo 3/3 de ambos métodos en ambas pruebas: {goal}',flush=True)
    return data


def figures(root,config,groups):
    folder=root/'figures'
    folder.mkdir(exist_ok=True)
    colors=['#2563eb','#d97706']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for group,color in zip(groups,colors):
        curves=[]
        training=[]
        for seed in config['seeds']:
            with (root/group['group']/f'seed_{seed}/validation.csv').open(encoding='utf-8',newline='') as f:
                rows=list(csv.DictReader(f))
            xs=[int(r['training_steps']) for r in rows]
            curves.append([np.mean(json.loads(r['returns'])[:10]) for r in rows])
            with (root/group['group']/f'seed_{seed}/training.csv').open(encoding='utf-8',newline='') as f:
                episodes=list(csv.DictReader(f))
            training.append([np.mean([float(r['return']) for r in episodes if start<int(r['training_steps'])<=end and r['budget_cut']=='False']) for start,end in zip(xs[:-1],xs[1:])])
        for ax,values,steps in [(axes[0],np.asarray(curves),xs),(axes[1],np.asarray(training),xs[1:])]:
            mean,spread=np.nanmean(values,axis=0),np.nanstd(values,axis=0)
            ax.plot(steps,mean,label=group['group'],color=color)
            ax.fill_between(steps,mean-spread,mean+spread,color=color,alpha=.15)
            ax.set_xlabel('Pasos del entorno')
            ax.set_ylabel('Retorno medio')
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis='x',style='plain')
    axes[0].set_title('Validación greedy: 10 episodios por control')
    axes[1].set_title('Entrenamiento: episodios completos por bloque')
    axes[0].axhline(config['threshold'],color='#64748b',ls='--',label='Referencia: 200')
    axes[0].legend()
    fig.suptitle('Nueva comparación DQN / Double DQN | media ± DE de tres semillas')
    for suffix in ('png','pdf'): fig.savefig(folder/f'learning.{suffix}',dpi=160)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    xs=np.arange(len(config['seeds']))
    for group,offset,color in zip(groups,(-.18,.18),colors):
        for ax,key in zip(axes,('test_mean','confirmation_mean')):
            ax.bar(xs+offset,[r[key] for r in group['runs']],width=.34,label=group['group'],color=color)
            ax.axhline(config['threshold'],color='#64748b',ls='--')
            ax.set_xticks(xs,config['seeds'])
            ax.set_xlabel('Semilla de entrenamiento')
            ax.set_ylabel('Retorno medio de 100 episodios')
            ax.grid(axis='y',alpha=.2)
    axes[0].set_title('Prueba: condiciones nuevas')
    axes[1].set_title('Confirmación: otra muestra nueva')
    axes[0].legend()
    for suffix in ('png','pdf'): fig.savefig(folder/f'tests.{suffix}',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/tier23_mejora_v1'))
    summarize(parser.parse_args().root)
