"""Verificar las tablas de IC y ejecutar únicamente el cuaderno de análisis."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,help='Manifest local previo de archivos protegidos')
    args=parser.parse_args()
    data=json.loads((ROOT/'results/analisis_intervalos/summary.json').read_text(encoding='utf-8'))
    assert data['runs_retrained']==0
    for r in data['rows']+data['paired_differences']:
        assert len(r['values'])==3 and r['n_training_seeds']==3
        mean=statistics.mean(r['values'])
        half=4.302652729696142*statistics.stdev(r['values'])/math.sqrt(3)
        for key,expected in [('mean',mean),('ci95_low',mean-half),('ci95_high',mean+half)]:
            assert math.isclose(r[key],expected,rel_tol=1e-12,abs_tol=1e-12), (r,key)
    assert {r['method'] for r in data['rows'] if r['tier']==1}=={
        'Q-learning (2/2/6/6 intervalos)','Q-learning (2/2/10/10 intervalos)',
        'Q-learning (2/2/20/20 intervalos)','DQN ajustado'}
    protected=0
    if args.before:
        manifest=json.loads(args.before.read_text(encoding='utf-8'))
        for name,digest in manifest.items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f'Archivo protegido cambiado: {name}'
        protected=len(manifest)
    runtime=ROOT/'.jupyter_runtime'; runtime.mkdir(exist_ok=True)
    os.environ['JUPYTER_RUNTIME_DIR']=str(runtime)
    path=ROOT/'Analisis_Teorico_e_Intervalos.ipynb'
    original=hashlib.sha256(path.read_bytes()).hexdigest()
    nb=nbformat.read(path,as_version=4); nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code': compile(c.source,'analysis_cell','exec')
    km=KernelManager(kernel_name='python3'); km.kernel_spec.argv[0]=sys.executable
    client=NotebookClient(nb,km=km,timeout=120,startup_timeout=60,resources={'metadata':{'path':str(ROOT)}})
    try: client.execute()
    finally:
        if km.has_kernel: km.shutdown_kernel(now=True)
        km.cleanup_resources()
    assert hashlib.sha256(path.read_bytes()).hexdigest()==original, 'El cuaderno cambió durante la verificación'
    nbformat.write(nb,path)
    images=sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
    assert images==8
    result={'complete':True,'new_training_runs':0,'ci_records_checked':len(data['rows'])+len(data['paired_differences']),
        'protected_files_unchanged':protected,'notebook_code_cells_executed':sum(c.cell_type=='code' for c in nb.cells),
        'embedded_figures':images,'notebook_errors':0,'original_three_notebooks_preserved':bool(args.before)}
    (ROOT/'results/analisis_intervalos/audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
