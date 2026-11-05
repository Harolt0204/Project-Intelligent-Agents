"""Auditar el presupuesto, selección congelada y pruebas de los seis modelos."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from experiments.refinar_lunar import ROOT, job_for, load_selected, evaluate, fingerprint


def audit(root, training_commit='8a5d3096b66ce95ca6902ea8281e7941cdfbb7b3'):
    root=Path(root)
    config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    frozen=json.loads((root/'frozen_selection.json').read_text(encoding='utf-8'))
    if len(frozen)!=2*len(config['seeds']): raise ValueError('Faltan modelos congelados')
    frozen={(r['group'],r['seed']):r for r in frozen}
    sources={}
    initial_policies={}
    records=[]
    for group in ('base','double'):
        for seed in config['seeds']:
            folder=root/group/f'seed_{seed}'
            model,run=load_selected(folder)
            summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            selected=json.loads((folder/'selection.json').read_text(encoding='utf-8'))
            tests=json.loads((folder/'test_evaluations.json').read_text(encoding='utf-8'))
            if (not summary['complete'] or run['protocol']!=config or run['job']!=job_for(config,group)
                    or run['seed']!=seed or summary['seed']!=seed):
                raise ValueError(f'Protocolo o modelo incompleto: {folder}')
            if summary['input_hash']!=fingerprint(config,job_for(config,group),seed) or run['input_hash']!=summary['input_hash']:
                raise ValueError('Huella de entrada incorrecta')
            p=config['base']
            first=((max(p['learning_starts'],p['batch_size'])+p['train_every']-1)//p['train_every'])*p['train_every']
            expected_updates=((config['steps']-first)//p['train_every']+1)*p['gradient_steps']
            if (summary['training_steps']!=config['steps'] or summary['updates']!=expected_updates or
                    summary['target_syncs']!=config['steps']//p['target_sync_every'] or
                    summary['stored_experiences']!=min(p['buffer_size'],config['steps'])):
                raise ValueError(f'Presupuesto o contadores incorrectos: {folder}')
            for name,digest in run['source_hashes'].items():
                if name not in sources:
                    sources[name]=hashlib.sha256(subprocess.check_output(['git','show',training_commit+':'+name],cwd=ROOT)).hexdigest()
                if sources[name]!=digest: raise ValueError(f'Código distinto de la versión de entrenamiento: {name}')
            digest=hashlib.sha256((folder/run['model_file']).read_bytes()).hexdigest()
            if digest!=summary['selected_model_sha256'] or digest!=frozen[(group,seed)]['sha256']:
                raise ValueError('Los pesos cambiaron desde la selección')
            with (folder/'validation.csv').open(encoding='utf-8',newline='') as f:
                controls=list(csv.DictReader(f))
            initial_policies[(group,seed)]=json.loads(controls[0]['returns'])
            confirmed=[float(r['confirmed_mean']) for r in controls if r['confirmed_mean']]
            if selected['mean_return']!=max(confirmed) or selected['training_steps']!=summary['selected_step']:
                raise ValueError('La selección no coincide con el mejor control confirmado')
            if abs(np.mean(selected['returns'])-selected['mean_return'])>1e-9 or not summary['selected_validation_reproduced']:
                raise ValueError('Validación guardada incompatible')
            if len(tests)!=2: raise ValueError('Falta una prueba')
            scores=[]
            for row,key,base in zip(tests,('test_mean','confirmation_mean'),
                    (config['heldout_evaluation_base'],config['confirmation_evaluation_base'])):
                if row['evaluation_base']!=base or row['episodes']!=100 or len(row['returns'])!=100:
                    raise ValueError('La prueba no contiene las condiciones predefinidas')
                mean=float(np.mean(row['returns']))
                if not np.isfinite(row['returns']).all() or abs(mean-summary[key])>1e-9 or abs(mean-row['mean_return'])>1e-9:
                    raise ValueError('La media no coincide con los cien retornos')
                replayed=evaluate(model,config,seed,1,evaluation_base=base)
                if replayed[0]!=row['returns'][0]: raise ValueError('El primer episodio de prueba no se reproduce')
                scores.append(mean)
            passed=all(value>=config['threshold'] for value in scores)
            if summary['threshold_met_both_tests']!=passed: raise ValueError('Indicador de éxito incorrecto')
            records.append({'group':group,'seed':seed,'selected_step':summary['selected_step'],
                'test_mean':scores[0],'confirmation_mean':scores[1],'goal_met':passed,
                'weights_frozen':True,'training_code_matches':True,'configuration_and_budget_verified':True,
                'validation_reproduced_episodes':100,'test_prefix_reproduced_episodes':2})
    if any(initial_policies[('base',seed)]!=initial_policies[('double',seed)] for seed in config['seeds']):
        raise ValueError('Las políticas iniciales emparejadas no coinciden')
    result={'model_count':len(records),'training_code_commit':training_commit,'models':records,
            'paired_initial_policies_verified':True,
            'validation_returns_reproduced':600,'test_episodes':1200,'test_prefix_episodes_reproduced':12,
            'goal_met':all(row['goal_met'] for row in records),
            'note':'Las pruebas completas se ejecutaron desde los pesos cargados tras congelar su elección; esta auditoría repite el primer episodio de cada muestra.'}
    (root/'audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'results/tier23_mejora_v1')
    parser.add_argument('--training-commit',default='8a5d3096b66ce95ca6902ea8281e7941cdfbb7b3')
    args=parser.parse_args()
    audit(args.root,args.training_commit)
