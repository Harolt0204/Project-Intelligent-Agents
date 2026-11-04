"""Confirmación de DQN y Double: seleccionar por validación y probar aparte."""

import csv
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch

from agents.replay import ReplayBuffer
from experiments.lunarlander import CLASSES, ROOT, QNetwork, evaluate, make_env, fingerprint


def job_for(config, group):
    if group not in ('base', 'double'):
        raise ValueError('Solo se comparan DQN base y Double DQN')
    return {'variant': 'dqn' if group == 'base' else 'double',
            'parameters': dict(config['base']), 'role': 'refinement'}


def learning_rate(step, config):
    schedule = config['learning_rate_schedule']
    fraction = np.clip((step - schedule['decay_start']) /
                       (schedule['decay_end'] - schedule['decay_start']), 0., 1.)
    return float(config['base']['learning_rate'] + fraction *
                 (schedule['final'] - config['base']['learning_rate']))


def accept_validation(candidate, previous):
    """Solo recibe puntuaciones de validación; nunca recibe una prueba final."""
    return previous is None or candidate > previous


def load_selected(folder):
    folder = Path(folder)
    run = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    architecture = run['architecture']
    torch.set_num_threads(1)
    model = QNetwork(architecture[0], architecture[-1], architecture[1:-1]).eval()
    model.load_state_dict(torch.load(folder/run['model_file'], map_location='cpu', weights_only=True))
    return model, run


def train(config, group, seed, folder):
    """Ambos métodos comparten este bucle y la misma regla de selección."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    job = job_for(config, group)
    p = job['parameters']
    env = make_env(config)
    obs_dim, actions = int(env.observation_space.shape[0]), int(env.action_space.n)
    agent = CLASSES[job['variant']](obs_dim=obs_dim, n_actions=actions,
        hidden_width=p['hidden_widths'], learning_rate=p['learning_rate'],
        gamma=p['gamma'], seed=seed, loss=p['loss'])
    replay = ReplayBuffer(p['buffer_size'], observation_shape=(obs_dim,), seed=4_000_000+seed)
    sources = ['agents/dqn.py', 'agents/double_dqn.py', 'agents/replay.py',
               'experiments/lunarlander.py', 'experiments/refinar_lunar.py']
    run = {'protocol': config, 'job': job, 'seed': seed,
        'input_hash': fingerprint(config, job, seed), 'architecture': [obs_dim, *p['hidden_widths'], actions],
        'model_file': 'online_selected.pt', 'last_model_file': 'online_final.pt',
        'environment_max_steps': env.spec.max_episode_steps,
        'python': platform.python_version(), 'platform': platform.platform(),
        'processor': platform.processor(), 'logical_cpus': os.cpu_count(), 'device': 'cpu', 'torch_threads': 1,
        'packages': {name: metadata.version(name) for name in ('torch','numpy','gymnasium','Box2D')},
        'source_hashes': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources}}
    (folder/'config.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    started = time.perf_counter()
    train_seconds = 0.
    validations = []
    best = None
    updates = syncs = 0

    def checkpoint(step):
        nonlocal best
        final = step == config['steps']
        episodes = config['final_eval_episodes'] if final else config['periodic_eval_episodes']
        returns = evaluate(agent.online, config, seed, episodes)
        row = {'training_steps': step, 'episodes': episodes, 'mean_return': float(np.mean(returns)),
               'returns': json.dumps(returns), 'training_seconds': train_seconds,
               'confirmed_mean': None, 'confirmed_returns': None, 'selected': False}
        if final or np.mean(returns) >= config['threshold']:
            confirmed = returns if final else evaluate(agent.online, config, seed, config['final_eval_episodes'])
            score = float(np.mean(confirmed))
            row.update(confirmed_mean=score, confirmed_returns=json.dumps(confirmed))
            if accept_validation(score, None if best is None else best['mean_return']):
                torch.save(agent.online.state_dict(), folder/'online_selected.pt')
                torch.save(agent.target.state_dict(), folder/'target_selected.pt')
                best = {'training_steps': step, 'mean_return': score, 'returns': confirmed}
                (folder/'selection.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
                row['selected'] = True
        validations.append(row)
        with (folder/'validation.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(validations)
        print(f'{group} semilla {seed} | paso {step:>7} | validacion {episodes}: {row["mean_return"]:.2f}'
              + (f' | confirmacion{config["final_eval_episodes"]}: {row["confirmed_mean"]:.2f}' if row['confirmed_mean'] is not None else ''), flush=True)

    step = 0
    try:
        checkpoint(0)
        obs, _ = env.reset(seed=seed)
        episode = episode_steps = 0
        total = 0.
        with ((folder/'training.csv').open('w', newline='', encoding='utf-8') as tf,
              (folder/'loss.csv').open('w', newline='', encoding='utf-8') as lf):
            tw = csv.DictWriter(tf, fieldnames=['training_steps','episode','episode_steps','return',
                                             'epsilon','terminated','truncated','budget_cut'])
            tw.writeheader()
            lw = csv.writer(lf)
            lw.writerow(['training_steps','updates','mean_loss','learning_rate'])
            timer = time.perf_counter()
            for step in range(1, config['steps']+1):
                epsilon = 1 + min((step-1)/p['epsilon_decay_steps'], 1)*(p['epsilon_end']-1)
                action = agent.choose_action(obs, epsilon)
                following, reward, terminated, truncated, _ = env.step(action)
                replay.add(obs, action, reward, following, terminated, truncated)
                if step >= p['learning_starts'] and len(replay) >= p['batch_size'] and step % p['train_every'] == 0:
                    lr = learning_rate(step, config)
                    for param_group in agent.optimizer.param_groups:
                        param_group['lr'] = lr
                    losses = []
                    for _ in range(p['gradient_steps']):
                        result = agent.learn(replay.sample(p['batch_size']))
                        if not np.isfinite(result['loss']):
                            raise FloatingPointError('Perdida no finita')
                        updates += 1
                        losses.append(result['loss'])
                    lw.writerow([step, updates, float(np.mean(losses)), lr])
                if step % p['target_sync_every'] == 0:
                    agent.sync_target()
                    syncs += 1
                obs = following
                total += reward
                episode_steps += 1
                if terminated or truncated or step == config['steps']:
                    episode += 1
                    tw.writerow({'training_steps': step, 'episode': episode, 'episode_steps': episode_steps,
                        'return': total, 'epsilon': epsilon, 'terminated': terminated, 'truncated': truncated,
                        'budget_cut': not (terminated or truncated)})
                    if terminated or truncated:
                        obs, _ = env.reset()
                    total = 0.
                    episode_steps = 0
                if step % config['eval_every'] == 0 or step == config['steps']:
                    train_seconds += time.perf_counter()-timer
                    tf.flush()
                    lf.flush()
                    checkpoint(step)
                    timer = time.perf_counter()
        torch.save(agent.online.state_dict(), folder/'online_final.pt')
        torch.save(agent.target.state_dict(), folder/'target_final.pt')
        # Comprobar el archivo seleccionado sin abrir todavía las muestras de prueba.
        selected, _ = load_selected(folder)
        reproduced = evaluate(selected, config, seed, config['final_eval_episodes'])
        if reproduced != best['returns']:
            raise AssertionError('La validacion del checkpoint guardado no se reproduce')
        reached = [r['training_steps'] for r in validations
                   if r['confirmed_mean'] is not None and r['confirmed_mean'] >= config['threshold']]
        summary = {'seed': seed, 'group': group, 'input_hash': run['input_hash'],
            'training_complete': True, 'complete': False,
            'training_steps': config['steps'], 'selected_step': best['training_steps'],
            'selected_validation_mean': best['mean_return'], 'last_validation_mean': validations[-1]['confirmed_mean'],
            'first_confirmed_validation_threshold_step': min(reached) if reached else None,
            'training_seconds': train_seconds, 'total_seconds': time.perf_counter()-started,
            'updates': updates, 'target_syncs': syncs, 'stored_experiences': len(replay),
            'selected_validation_reproduced': True,
            'selected_model_sha256': hashlib.sha256((folder/'online_selected.pt').read_bytes()).hexdigest()}
        (folder/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return summary
    except Exception as error:
        (folder/'failure.json').write_text(json.dumps({'training_steps': step, 'error': repr(error), 'complete': False}, indent=2), encoding='utf-8')
        raise
    finally:
        env.close()


def finalize_tests(root):
    """Congelar los seis modelos antes de abrir las dos muestras de prueba."""
    root=Path(root)
    config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    frozen=[]
    for group in ('base','double'):
        for seed in config['seeds']:
            folder=root/group/f'seed_{seed}'
            summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
            expected=fingerprint(config,job_for(config,group),seed)
            digest=hashlib.sha256((folder/'online_selected.pt').read_bytes()).hexdigest()
            if not summary['training_complete'] or summary['input_hash']!=expected or digest!=summary['selected_model_sha256']:
                raise ValueError(f'Modelo o protocolo incompatible: {folder}')
            if summary['selected_validation_mean']<config['validation_gate']:
                raise ValueError('No se abre la prueba: algun modelo no alcanzo el margen de validacion')
            frozen.append({'group':group,'seed':seed,'sha256':digest,'selected_step':summary['selected_step']})
    frozen_path=root/'frozen_selection.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8'))!=frozen:
        raise ValueError('Los pesos seleccionados cambiaron; se conserva la prueba anterior')
    frozen_path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    for item in frozen:
        group,seed=item['group'],item['seed']
        folder=root/group/f'seed_{seed}'
        summary=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if summary['complete']: continue
        selected,_=load_selected(folder)
        tests=[]
        timer=time.perf_counter()
        for label,base in [('test',config['heldout_evaluation_base']),('confirmation',config['confirmation_evaluation_base'])]:
            returns=evaluate(selected,config,seed,config['final_eval_episodes'],evaluation_base=base)
            tests.append({'sample':label,'evaluation_base':base,'episodes':len(returns),
                'mean_return':float(np.mean(returns)),'std_return':float(np.std(returns)),'returns':returns})
            (folder/'test_evaluations.json').write_text(json.dumps(tests,indent=2),encoding='utf-8')
            print(f'{group} semilla {seed} | {label} {len(returns)}: {np.mean(returns):.2f}',flush=True)
        if hashlib.sha256((folder/'online_selected.pt').read_bytes()).hexdigest()!=item['sha256']:
            raise AssertionError('Los pesos cambiaron durante la evaluacion final')
        summary.update(complete=True,test_mean=tests[0]['mean_return'],confirmation_mean=tests[1]['mean_return'],
            threshold_met_both_tests=bool(all(row['mean_return']>=config['threshold'] for row in tests)),
            test_seconds=time.perf_counter()-timer)
        summary['total_seconds']+=summary['test_seconds']
        (folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
