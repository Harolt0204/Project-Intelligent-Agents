"""Reproducir el primer episodio de un modelo LunarLander guardado."""

import argparse
from pathlib import Path

from experiments.cuaderno_lunar import demostrar


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group',default='base')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--root',type=Path,default=None,help='Omitir para usar los modelos vigentes revisados')
    parser.add_argument('--gif',action='store_true')
    args=parser.parse_args()
    demostrar(args.group,args.seed,args.root,inline=False,save_gif=args.gif)


if __name__=='__main__': main()
