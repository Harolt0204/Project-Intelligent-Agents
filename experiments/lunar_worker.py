"""Una semilla independiente; proceso numérico, no agente externo."""

import argparse
import json
from pathlib import Path

from experiments.lunarlander import plan, fingerprint, train
from experiments.console_utils import utf8_console


def main():
    utf8_console()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/tier23_lunarlander'))
    parser.add_argument('--group',required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    config=json.loads((args.root/'protocol.json').read_text(encoding='utf-8'))
    job=plan(config)[args.group]
    if args.seed not in config['seeds']: raise ValueError('Semilla fuera del protocolo')
    folder=args.root/args.group/f'seed_{args.seed}'
    if (folder/'summary.json').exists():
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary['input_hash']!=fingerprint(config,job,args.seed): raise ValueError('Resultado incompatible')
        print(f'Reutilizado {args.group}, semilla {args.seed}',flush=True)
    elif folder.exists():
        raise FileExistsError(f'No se sobrescribe una ejecución parcial: {folder}')
    else:
        train(config,job,args.seed,folder)


if __name__=='__main__': main()
