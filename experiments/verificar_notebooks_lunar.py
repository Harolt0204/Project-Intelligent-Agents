"""Comprobar ambos notebooks con kernel real, sin tocar el notebook de Tier 1."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import sys

from jupyter_client import KernelManager
import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--animacion',choices=('2','3'))
    args=parser.parse_args()
    runtime=ROOT/'.jupyter_runtime'
    runtime.mkdir(exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    checks=[]
    for tier,name in [(2,'Tier2_LunarLander.ipynb'),(3,'Tier3_DoubleDQN.ipynb')]:
        if args.animacion and int(args.animacion)!=tier: continue
        path=ROOT/name
        original_hash=hashlib.sha256(path.read_bytes()).hexdigest()
        nb=nbformat.read(path,as_version=4)
        if any(c.cell_type=='code' and re.search(r'(?m)^\s*EJECUTAR_TIER[23]\s*=\s*True\b',c.source) for c in nb.cells):
            raise RuntimeError(f'{name} tiene activado un entrenamiento largo. Se conserva el archivo y no se relanza durante la verificación.')
        if args.animacion:
            setup=next(c.source for c in nb.cells if c.cell_type=='code' and 'from pathlib import Path' in c.source)
            demo=next(c.source for c in nb.cells if c.cell_type=='code' and 'VER_NAVE = False' in c.source)
            nb.cells=[nbformat.v4.new_code_cell(setup),nbformat.v4.new_code_cell(demo.replace('VER_NAVE = False','VER_NAVE = True'))]
        for i,c in enumerate(nb.cells):
            if c.cell_type=='code': compile(c.source,f'{name}_cell_{i}','exec')
        km=KernelManager(kernel_name='python3')
        km.kernel_spec.argv[0]=sys.executable
        def progress(cell,cell_index):
            print(f'{name}: celda {cell_index+1}/{len(nb.cells)}',flush=True)
        client=NotebookClient(nb,km=km,timeout=240,startup_timeout=60,on_cell_start=progress,
                              resources={'metadata':{'path':str(ROOT)}})
        try:
            client.execute()
        finally:
            if km.has_kernel: km.shutdown_kernel(now=True)
            km.cleanup_resources()
        if args.animacion:
            html=[o['data'].get('text/html','') for c in nb.cells for o in c.get('outputs',[]) if 'data' in o]
            assert any('data:image/png;base64' in h for h in html)
        else:
            if hashlib.sha256(path.read_bytes()).hexdigest()!=original_hash:
                raise RuntimeError(f'{name} cambió durante la verificación; se conserva tu versión sin sobrescribirla')
            nbformat.write(nb,path)
        checks.append({'notebook':name,'code_cells_executed':sum(c.cell_type=='code' for c in nb.cells),'animation':bool(args.animacion),'errors':0})
    file='tier23_notebooks_check.json' if not args.animacion else f'tier{args.animacion}_animation_check.json'
    (ROOT/'results'/file).write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print('Verificación terminada sin errores.',flush=True)


if __name__=='__main__': main()
