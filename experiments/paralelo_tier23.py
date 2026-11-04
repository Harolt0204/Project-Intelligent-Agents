"""Tres semillas en procesos independientes, un tratamiento cada vez."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from experiments.lunarlander import ROOT, configuration, plan
from experiments.analizar_tier23 import analyze
from experiments.console_utils import utf8_console


def main():
    utf8_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results/tier23_lunarlander'))
    parser.add_argument('--groups',nargs='+')
    args=parser.parse_args()
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    config=configuration()
    path=output/'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=config: raise ValueError('Usa otra carpeta para otro protocolo')
    else:
        path.write_text(json.dumps(config,indent=2,ensure_ascii=False),encoding='utf-8')
    jobs=plan(config)
    groups=list(jobs) if args.groups is None else args.groups
    if any(name not in jobs for name in groups): raise ValueError('Grupo desconocido')
    (output/'plan.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    (output/'execution.json').write_text(json.dumps({'mode':'Three independent seed processes; treatments run in sequence','torch_threads_per_process':1,'maximum_workers':len(config['seeds']),'numerical_protocol_unchanged':True,'timing_note':'Wall-clock observado con procesos compartiendo CPU; no benchmark aislado. La primera semilla base comenzó antes de habilitar paralelismo.'},indent=2),encoding='utf-8')
    logs=output/'logs'
    logs.mkdir(exist_ok=True)
    failed=[]
    for name in groups:
        active=[]
        for seed in config['seeds']:
            log=logs/f'{name}_{seed}.log'
            handle=log.open('w',encoding='utf-8')
            process=subprocess.Popen([sys.executable,'-m','experiments.lunar_worker','--root',str(output),'--group',name,'--seed',str(seed)],
                cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONUTF8':'1'},
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
            active.append({'seed':seed,'process':process,'handle':handle,'log':log,'offset':0})
        try:
            while True:
                for item in active:
                    with item['log'].open(encoding='utf-8',errors='replace') as reader:
                        reader.seek(item['offset'])
                        text=reader.read()
                        item['offset']=reader.tell()
                    if text: print(text,end='',flush=True)
                if all(item['process'].poll() is not None for item in active): break
                time.sleep(5)
        finally:
            for item in active:
                if item['process'].poll() is None: item['process'].terminate()
                item['process'].wait(timeout=10)
                item['handle'].close()
        runs=[]
        for item in active:
            file=output/name/f"seed_{item['seed']}/summary.json"
            if item['process'].returncode!=0 or not file.exists():
                failed.append({'group':name,'seed':item['seed'],'log':str(item['log'])})
            else:
                runs.append(json.loads(file.read_text(encoding='utf-8')))
        if runs:
            values=[r['after'] for r in runs]
            data={'runs':runs,'mean_after':float(np.mean(values)),'std_across_seed_means':float(np.std(values)),
                  'successful_seeds':sum(v>=config['threshold'] for v in values),'complete':len(runs)==len(config['seeds'])}
            (output/name/'aggregate.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
            print(f"Grupo {name}: {data['mean_after']:.2f} ± {data['std_across_seed_means']:.2f}; completo={data['complete']}",flush=True)
        analyze(output)
    if failed:
        (output/'parallel_failures.json').write_text(json.dumps(failed,indent=2),encoding='utf-8')
        raise RuntimeError('Consultar parallel_failures.json; no se han inventado resultados de ejecuciones fallidas')


if __name__=='__main__': main()
