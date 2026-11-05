"""Continuar el trabajo autorizado: estudios, figuras y verificaciones, sin publicar."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from experiments.lunarlander import ROOT
from experiments.console_utils import utf8_console


def execute(*arguments):
    print('Etapa:', ' '.join(arguments),flush=True)
    subprocess.run([sys.executable,'-m',*arguments],cwd=ROOT,check=True)


def main():
    utf8_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-base',action='store_true')
    args=parser.parse_args()
    output=ROOT/'results/tier23_lunarlander'
    if args.wait_base:
        print('Esperando las tres semillas base que ya están en ejecución.',flush=True)
        start=time.monotonic()
        while not all((output/'base'/f'seed_{seed}/summary.json').exists() for seed in (42,43,44)):
            if time.monotonic()-start>10800:
                raise TimeoutError('La base no se completó; consultar los registros, sin sobrescribir resultados parciales')
            time.sleep(5)
    execute('experiments.paralelo_tier23')
    execute('experiments.analizar_tier23')
    if not (output/'verification/new_summary.json').exists():
        execute('experiments.verificar_tier23')
    execute('experiments.mostrar_lunarlander','--group','base','--root',str(output),'--gif')
    execute('experiments.mostrar_lunarlander','--group','double','--root',str(output),'--gif')
    execute('experiments.analizar_tier23')
    execute('experiments.verificar_notebooks_lunar')
    execute('experiments.verificar_notebooks_lunar','--animacion','2')
    execute('experiments.verificar_notebooks_lunar','--animacion','3')
    (output/'workflow_complete.json').write_text(json.dumps({'studies_executed':True,'saved_models_verified':True,
        'notebooks_executed':True,'animations_verified':True},indent=2),encoding='utf-8')
    print('Estudios y verificaciones terminados. Falta revisión y publicación.',flush=True)


if __name__=='__main__': main()
