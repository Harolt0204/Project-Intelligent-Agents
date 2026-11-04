"""Reproducir evaluaciones guardadas y comprobar otros inicios sin entrenar."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from experiments.lunarlander import load_model, evaluate, make_env, plan


def outcomes(model,config,seed,base):
    """La misma política greedy, guardando causas observables de finalización."""
    env=make_env(config)
    rng=np.random.default_rng(base+500_000+1000*seed)
    records=[]
    try:
        for episode in range(config['final_eval_episodes']):
            obs,_=env.reset(seed=base+1000*seed+episode)
            total=0.
            length=0
            while True:
                with torch.inference_mode():
                    q=model(torch.as_tensor(obs,dtype=torch.float32)).numpy()
                action=int(rng.choice(np.flatnonzero(q==q.max())))
                obs,reward,terminated,truncated,_=env.step(action)
                total+=reward
                length+=1
                if terminated or truncated:
                    raw=env.unwrapped
                    if truncated:
                        reason='time_limit'
                    elif getattr(raw,'game_over',False):
                        reason='body_contact'
                    elif abs(float(obs[0]))>=1:
                        reason='outside_viewport'
                    elif not raw.lander.awake:
                        reason='resting'
                    else:
                        reason='other_termination'
                    records.append({'return':float(total),'steps':length,'ending':reason,'final_observation':obs.tolist()})
                    break
    finally: env.close()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('results/tier23_lunarlander'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    dest=args.output or args.root/'verification'
    dest.mkdir(parents=True,exist_ok=False)
    config=json.loads((args.root/'protocol.json').read_text(encoding='utf-8'))
    checks=[]
    for name in plan(config):
        for seed in config['seeds']:
            folder=args.root/name/f'seed_{seed}'
            model,run_config=load_model(folder)
            with (folder/'evaluation.csv').open(encoding='utf-8',newline='') as f:
                final=list(csv.DictReader(f))[-1]
            assert int(final['episodes'])==config['final_eval_episodes']
            expected=json.loads(final['returns'])
            episodes=outcomes(model,config,seed,config['evaluation_base'])
            actual=[episode['return'] for episode in episodes]
            if actual!=expected: raise AssertionError(f'No se reprodujeron exactamente los retornos: {folder}')
            checks.append({'group':name,'seed':seed,'exact_returns':True,'mean_return':float(np.mean(actual)),
                           'endings':dict(Counter(e['ending'] for e in episodes)),'episodes':episodes})
            print(f'Verificado {name}, semilla {seed}: {np.mean(actual):.2f}',flush=True)
            (dest/'reproduction.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    # La selección usa únicamente desarrollo; las condiciones nuevas siguen sin consultar.
    dqn_groups = [name for name, job in plan(config).items() if job['variant'] == 'dqn']
    best_tier2 = max(dqn_groups, key=lambda name: np.mean([r['mean_return'] for r in checks if r['group'] == name]))
    groups = list(dict.fromkeys(['base', 'double', best_tier2]))
    protocol={'evaluation_base':config['heldout_evaluation_base'],'groups':groups,
              'episodes_per_model':config['final_eval_episodes'],
              'tier2_selected_group':best_tier2,
              'selection_note':'Base y Double son la comparación predefinida. Se añade el DQN de mayor media de desarrollo en Tier 2; esa selección se decidió después de observar desarrollo. Las condiciones iniciales adicionales estaban fijadas antes de entrenar y no se usan para reajustar.',
              'purpose':'Comprobar modelos con otras condiciones iniciales sin volver a entrenar'}
    (dest/'new_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[]
    summary=[]
    for name in groups:
        means=[]
        for seed in config['seeds']:
            model,_=load_model(args.root/name/f'seed_{seed}')
            episodes=outcomes(model,config,seed,config['heldout_evaluation_base'])
            values=[e['return'] for e in episodes]
            average=float(np.mean(values))
            means.append(average)
            records.append({'group':name,'training_seed':seed,'mean_return':average,
                            'endings':dict(Counter(e['ending'] for e in episodes)),'episodes':episodes})
            print(f'Nuevos inicios {name}, semilla {seed}: {average:.2f}',flush=True)
            (dest/'new_records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
        summary.append({'group':name,'mean_return':float(np.mean(means)),'std_across_seed_means':float(np.std(means)),
                        'means_by_seed':dict(zip(map(str,config['seeds']),means)),
                        'successful_seeds':sum(v>=config['threshold'] for v in means)})
    (dest/'new_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')


if __name__=='__main__': main()
