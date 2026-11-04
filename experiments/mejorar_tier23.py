"""Ejecutar seis modelos nuevos; los resultados originales no se sobrescriben."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from experiments.console_utils import utf8_console

ROOT = Path(__file__).resolve().parents[1]


def expected_hash(config, group, seed):
    job = {'variant': 'dqn' if group == 'base' else 'double',
           'parameters': dict(config['base']), 'role': 'refinement'}
    return hashlib.sha256(json.dumps({'config':config,'job':job,'seed':seed}, sort_keys=True).encode()).hexdigest()


def progress(output, config):
    records = []
    for group in ('base','double'):
        for seed in config['seeds']:
            folder = output/group/f'seed_{seed}'
            row = {'group':group,'seed':seed,'state':'pending','steps':0}
            try:
                if (folder/'summary.json').exists():
                    run = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
                    row.update(state='complete' if run['complete'] else 'validation_complete',
                               steps=run['training_steps'],best_validation_100=run['selected_validation_mean'])
                    if run['complete']:
                        row.update(test_mean=run['test_mean'],confirmation_mean=run['confirmation_mean'])
                elif (folder/'failure.json').exists():
                    row.update(state='failed', error=json.loads((folder/'failure.json').read_text(encoding='utf-8')))
                elif (folder/'validation.csv').exists():
                    with (folder/'validation.csv').open(encoding='utf-8',newline='') as f:
                        rows = list(csv.DictReader(f))
                    if rows:
                        row.update(state='training',steps=int(rows[-1]['training_steps']),
                                   validation_mean=float(rows[-1]['mean_return']))
                    if (folder/'selection.json').exists():
                        selected = json.loads((folder/'selection.json').read_text(encoding='utf-8'))
                        row['best_validation_100'] = selected['mean_return']
                elif folder.exists():
                    row['state'] = 'starting'
            except (OSError, ValueError, KeyError):
                row['state'] = 'updating'
            records.append(row)
    try:
        (output/'progress.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    except OSError:
        pass  # Un lector de Windows no debe interrumpir los entrenamientos.


def main():
    utf8_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/tier23_mejora_v1.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results/tier23_mejora_v1')
    parser.add_argument('--worker',choices=('base','double'))
    parser.add_argument('--seed',type=int)
    parser.add_argument('--test-only',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if args.test_only:
        from experiments.refinar_lunar import finalize_tests
        finalize_tests(output)
        return
    if args.worker:
        config=json.loads((output/'protocol.json').read_text(encoding='utf-8'))
        if args.seed not in config['seeds']: raise ValueError('Semilla fuera del protocolo')
        folder=output/args.worker/f'seed_{args.seed}'
        if (folder/'summary.json').exists():
            saved=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            if saved['input_hash']!=expected_hash(config,args.worker,args.seed): raise ValueError('Resultado incompatible')
            if not saved['training_complete']: raise ValueError('Entrenamiento incompleto')
            print(f'Reutilizado {args.worker} {args.seed}',flush=True)
            return
        if folder.exists(): raise FileExistsError(f'No se sobrescribe una ejecucion parcial: {folder}')
        from experiments.refinar_lunar import train
        train(config,args.worker,args.seed,folder)
        return
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if config['seeds'] != [42,43,44]: raise ValueError('Este ejecutor compara las tres semillas predefinidas')
    output.mkdir(parents=True,exist_ok=True)
    protocol=output/'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text(encoding='utf-8'))!=config:
        raise ValueError('Otro protocolo necesita otra carpeta')
    protocol.write_text(json.dumps(config,indent=2),encoding='utf-8')
    rounds=[[('double',43),('base',42),('double',44)], [('base',43),('base',44),('double',42)]]
    execution={'maximum_workers':3,'torch_threads_per_worker':1,'rounds':rounds,
               'note':'Casos debiles primero; todos usan el mismo presupuesto. Procesos comparten CPU, no benchmark aislado.'}
    (output/'execution.json').write_text(json.dumps(execution,indent=2),encoding='utf-8')
    (output/'logs').mkdir(exist_ok=True)
    failed=[]
    for jobs in rounds:
        active=[]
        try:
            for group,seed in jobs:
                log=output/'logs'/f'{group}_{seed}.log'
                handle=log.open('w',encoding='utf-8')
                process=subprocess.Popen([sys.executable,'-m','experiments.mejorar_tier23','--worker',group,
                    '--seed',str(seed),'--output',str(output)],cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'},
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
                active.append({'group':group,'seed':seed,'process':process,'handle':handle,'log':log,'offset':0})
            while True:
                for item in active:
                    with item['log'].open(encoding='utf-8',errors='replace') as reader:
                        reader.seek(item['offset'])
                        text=reader.read()
                        item['offset']=reader.tell()
                    if text: print(text,end='',flush=True)
                progress(output,config)
                if all(item['process'].poll() is not None for item in active): break
                time.sleep(5)
        finally:
            for item in active:
                if item['process'].poll() is None: item['process'].terminate()
                item['process'].wait(timeout=15)
                item['handle'].close()
        for item in active:
            if item['process'].returncode or not (output/item['group']/f"seed_{item['seed']}/summary.json").exists():
                failed.append({'group':item['group'],'seed':item['seed'],'log':str(item['log'])})
        if not failed:
            weak=[]
            for group,seed in jobs:
                run=json.loads((output/group/f'seed_{seed}/summary.json').read_text(encoding='utf-8'))
                if run['selected_validation_mean']<config['validation_gate']:
                    weak.append({'group':group,'seed':seed,'validation_mean':run['selected_validation_mean']})
            if weak:
                (output/'development_status.json').write_text(json.dumps({'ready_for_test':False,'weak':weak,
                    'test_samples_consulted':False,'reason':'Margen de validacion insuficiente; revisar antes de abrir prueba'},indent=2),encoding='utf-8')
                print('La ronda necesita ajustes en desarrollo. No se abrieron las muestras de prueba.',flush=True)
                progress(output,config)
                return
    if failed:
        (output/'failures.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
        raise RuntimeError('Hubo ejecuciones fallidas; se conserva su diagnostico')
    subprocess.run([sys.executable,'-m','experiments.mejorar_tier23','--test-only','--output',str(output)],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-m','experiments.resumir_mejora','--root',str(output)],cwd=ROOT,check=True)
    progress(output,config)
    print('Ronda terminada. Revisar goal_met en summary.json antes de afirmar 3/3.',flush=True)


if __name__=='__main__': main()
