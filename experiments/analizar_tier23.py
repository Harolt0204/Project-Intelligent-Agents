"""Figuras y discusión Tier 2/3 desde los CSV reales; no entrena modelos."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.lunarlander import configuration, plan
from experiments.intervalos import mean_ci95, ci_record


def available(output, config):
    rows = []
    for name, job in plan(config).items():
        files = [output / name / f"seed_{seed}/summary.json" for seed in config["seeds"]]
        if not all(path.exists() for path in files):
            continue
        runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
        values = [r["after"] for r in runs]
        rows.append({"group": name, "variant": job["variant"], "role": job["role"],
            "final_ci95": ci_record(values),
            "parameters": job["parameters"], "final_mean": float(np.mean(values)),
            "std_across_seed_means": float(np.std(values)), "means_by_seed": dict(zip(map(str,config["seeds"]),values)),
            "successful_seeds": sum(v >= config["threshold"] for v in values),
            "training_seconds_mean": float(np.mean([r["training_seconds"] for r in runs])),
            "total_seconds_mean": float(np.mean([r["total_seconds"] for r in runs])),
            "first_confirmed_threshold_steps": [r["first_confirmed_threshold_step"] for r in runs]})
    return rows


def curves(output, name, config):
    eval_curves, train_curves = [], []
    expected = list(range(0, config["steps"]+1, config["eval_every"]))
    if expected[-1] != config["steps"]:
        expected.append(config["steps"])
    for seed in config["seeds"]:
        folder = output / name / f"seed_{seed}"
        with (folder / "evaluation.csv").open(encoding="utf-8",newline="") as file:
            eval_rows = list(csv.DictReader(file))
        xs = [int(r["training_steps"]) for r in eval_rows]
        if xs != expected:
            raise ValueError(f"Controles incompatibles: {folder}")
        eval_curves.append([np.mean(json.loads(r["returns"])[:config["periodic_eval_episodes"]]) for r in eval_rows])
        with (folder / "training.csv").open(encoding="utf-8",newline="") as file:
            train_rows = list(csv.DictReader(file))
        values = []
        for start, end in zip(xs[:-1],xs[1:]):
            returns = [float(r["return"]) for r in train_rows if start<int(r["training_steps"])<=end and r["budget_cut"]=="False"]
            values.append(np.mean(returns) if returns else np.nan)
        train_curves.append(values)
    return xs, np.asarray(eval_curves), np.asarray(train_curves)


def study_groups(config, key):
    groups = []
    for value in config["studies"][key]:
        name = "base" if value==config["base"][key] else f"{key}_{value:g}".replace(".","p")
        groups.append((name,f"{key} = {value:g}"))
    return groups


def diagnostics(output, config, rows):
    """Describir los registros medidos sin equiparar una Q alta con sesgo real."""
    records = []
    for group in rows:
        for seed in config['seeds']:
            folder = output / group['group'] / f'seed_{seed}'
            checks = json.loads((folder / 'threshold_checks.json').read_text(encoding='utf-8'))
            best = max(checks, key=lambda row: row['mean_return'])
            with (folder / 'loss.csv').open(encoding='utf-8', newline='') as file:
                losses = [float(row['mean_loss']) for row in csv.DictReader(file)
                          if int(row['training_steps']) > config['steps'] - config['eval_every']]
            with (folder / 'evaluation.csv').open(encoding='utf-8', newline='') as file:
                controls = list(csv.DictReader(file))
            final = float(controls[-1]['mean_return'])
            records.append({'group': group['group'], 'seed': seed,
                'best_observed_100_episode_mean': best['mean_return'],
                'best_observed_100_episode_step': best['training_steps'],
                'final_100_episode_mean': final,
                'change_from_best_observed_100': final - best['mean_return'],
                'last_block_loss_median': float(np.median(losses)),
                'last_block_loss_p95': float(np.percentile(losses, 95)),
                'final_q_probe_min': float(controls[-1]['q_probe_min']),
                'final_q_probe_max': float(controls[-1]['q_probe_max'])})
    (output / 'diagnostics.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    return records


def plot(output, config, groups, filename, title):
    fig, axes = plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    interval_rows = []
    colors = ["#2563eb","#d97706","#15803d"]
    for (name,label), color in zip(groups,colors):
        xs, eval_values, train_values = curves(output,name,config)
        for ax,values,steps in [(axes[0],eval_values,xs),(axes[1],train_values,xs[1:])]:
            mean, low, high = mean_ci95(values)
            ax.plot(steps,mean,label=label,color=color,linewidth=2)
            ax.fill_between(steps,low,high,color=color,alpha=.15)
            for step, average, lower, upper in zip(steps, mean, low, high):
                interval_rows.append({'group':name,'kind':'evaluation' if ax is axes[0] else 'training',
                    'training_steps':step,'mean':float(average),'ci95_low':float(lower),'ci95_high':float(upper)})
            ax.set_xlabel("Pasos de entrenamiento")
            ax.grid(alpha=.2)
            ax.ticklabel_format(axis="x",style="plain")
    axes[0].axhline(config["threshold"],ls="--",color="#64748b",label="Objetivo: 200 en 100 episodios")
    axes[0].set_title("Evaluación greedy (10 episodios por control)")
    axes[1].set_title("Entrenamiento con exploración")
    axes[0].set_ylabel("Retorno medio")
    axes[1].set_ylabel("Retorno por episodio completo, media por bloque")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="outside lower center",ncol=min(3,len(labels)),fontsize=9)
    fig.suptitle(title+" | media e IC 95% puntual (t, 3 semillas)",fontsize=11)
    for suffix in ("png","pdf"):
        fig.savefig(output / "figures" / f"{filename}.{suffix}",dpi=160)
    plt.close(fig)
    with (output/'figures'/f'{filename}_intervals.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(interval_rows[0]))
        writer.writeheader()
        writer.writerows(interval_rows)


def report(output, config, rows, tier=None):
    if tier == 2:
        rows = [r for r in rows if r["group"] != "double"]
    if tier == 3:
        rows = [r for r in rows if r["group"] in ("base", "double")]
    by_name = {r["group"]:r for r in rows}
    title = f"Tier {tier}" if tier else "Tier 2 y Tier 3"
    expected = 6 if tier == 2 else (2 if tier == 3 else len(plan(config)))
    lines = [f"# {title} — LunarLander-v3", "",
        "## Protocolo", "",
        f"Entorno {config['environment']}, acciones discretas, parámetros {config['environment_kwargs']}.",
        f"Presupuesto {config['steps']} pasos, semillas {config['seeds']}, evaluación greedy cada {config['eval_every']} pasos.",
        f"Base: red 8 → {' → '.join(map(str, config['base']['hidden_widths']))} → 4, Adam lr={config['base']['learning_rate']}, pérdida {config['base']['loss']}, gamma={config['base']['gamma']}.",
        f"Replay {config['base']['buffer_size']}, lote {config['base']['batch_size']}, ajustes desde {config['base']['learning_starts']} pasos: {config['base']['gradient_steps']} ajuste cada {config['base']['train_every']} pasos; copia target cada {config['base']['target_sync_every']}.",
        f"Exploración lineal de 1 a {config['base']['epsilon_end']} en {config['base']['epsilon_decay_steps']} pasos, después constante. Evaluación sin exploración.",
        "Controles de 10 episodios; inicio/final de 100 y confirmación de 100 cuando un control alcanza 200.",
        "Se entregan modelos finales. No se elige el mejor checkpoint ni se descarta una semilla débil.",
        "Las curvas usan los primeros 10 episodios al inicio/final; las tablas finales usan los 100.",
        "Las bandas son IC t del 95% puntuales entre tres medias de semillas (DE muestral, ddof=1). Las tablas conservan media ± DE descriptiva (ddof=0).",
        "Con tres réplicas, la normalidad no se verifica bien; los IC son orientativos, condicionales a esta configuración y no bandas simultáneas de toda la curva. Véase report/intervalos_confianza.md.",
        "DQN base se reutiliza en los dos estudios, la ablación y Tier 3. No son entrenamientos adicionales independientes.",
        "Todos los agentes son propios; la referencia SB3 Zoo solo orientó parámetros. No se importó un agente entrenado.",
        "Errata del protocolo congelado: el texto descriptivo selection menciona 200000 por arrastre de un piloto. El campo steps, los bucles y los CSV usan 500000; no se cambia el protocolo histórico.",
        "", "## Estado y resultados completos", "",
        f"Grupos completos: {len(rows)}/{expected}. No se informan ejecuciones parciales como resultados finales.", "",
        "| Grupo | Media final ± DE | Retornos por semilla | Semillas ≥200 | Entrenamiento medio | Total medio con evaluación |",
        "|---|---:|---|---:|---:|---:|"]
    for r in rows:
        values=", ".join(f"{v:.2f}" for v in r["means_by_seed"].values())
        lines.append(f"| {r['group']} | {r['final_mean']:.2f} ± {r['std_across_seed_means']:.2f} | {values} | {r['successful_seeds']}/3 | {r['training_seconds_mean']:.1f} s | {r['total_seconds_mean']:.1f} s |")
    lines += ["", "La columna entrenamiento excluye evaluación; total la incluye. El total va desde la evaluación inicial hasta guardar los modelos, excluyendo arranque de Python y preparación inicial.",
              "Los tratamientos se ejecutan en secuencia; sus semillas pueden correr en procesos paralelos, un hilo PyTorch por proceso.",
              "Consultar execution.json: la primera semilla base comenzó con ejecución secuencial. Los tiempos no se interpretan como comparaciones aisladas de hardware.",
              "Durante el trabajo se realizan verificaciones y mediciones de coste; estos tiempos de reloj describen esta máquina y no son benchmarks aislados."]
    if tier != 3:
        lines += ["", "## Hiperparámetros", ""]
    for key in (config["studies"] if tier != 3 else []):
        names=study_groups(config,key)
        if all(name in by_name for name,_ in names):
            best=max(names,key=lambda pair:by_name[pair[0]]["final_mean"])
            delta = by_name[best[0]]['final_mean'] - by_name['base']['final_mean']
            lines.append(f"- {key}: valores {config['studies'][key]}; mayor media observada: {best[1]} ({by_name[best[0]]['final_mean']:.2f}; diferencia frente a base: {delta:+.2f}).")
            lines.append("  Se varía solo ese parámetro respecto a base. La variación entre semillas limita la generalización del ranking.")
            if key == 'learning_rate':
                slow = min(names, key=lambda pair: by_name[pair[0]]['parameters'][key])[0]
                gap = by_name[slow]['final_mean'] - by_name['base']['final_mean']
                lines.append(f"  La menor tasa terminó {gap:+.2f} puntos frente a base con el mismo número de ajustes. Adam escala los cambios por learning_rate; una tasa pequeña puede ralentizar la adquisición de una política útil con presupuesto fijo. Esto no demuestra que bastaría aumentar los pasos para resolverla.")
            elif key == 'target_sync_every':
                intervals = config['studies'][key]
                lines.append(f"  Las copias cada {intervals} pasos equivalen aproximadamente a {[round(value/config['base']['train_every'], 1) for value in intervals]} ajustes de online por copia. Un intervalo largo conserva objetivos más antiguos; uno corto los hace seguir más de cerca a online. El ranking mide esa compensación en esta configuración, no una frecuencia óptima universal.")
    if "base" in by_name and "no_target" in by_name:
        delta=by_name["no_target"]["final_mean"]-by_name["base"]["final_mean"]
        paired = [by_name['no_target']['means_by_seed'][str(seed)] - by_name['base']['means_by_seed'][str(seed)] for seed in config['seeds']]
        lines += ["", "## Ablación de red objetivo", "",
            f"Sin target menos DQN base: {delta:+.2f} puntos de media final.",
            f"Diferencias emparejadas por semilla {config['seeds']}: {', '.join(f'{value:+.2f}' for value in paired)}; mejora en {sum(value > 0 for value in paired)}/3.",
            f"DE sin target: {by_name['no_target']['std_across_seed_means']:.2f}; base: {by_name['base']['std_across_seed_means']:.2f}.",
            "Replay, red online, optimizador, presupuesto y demás parámetros permanecen iguales; no hay red target independiente.",
            "Los objetivos se calculan con online dentro de no_grad. Cambia la referencia del bootstrap, no se añade un gradiente por ella.",
            ("La caída observada es compatible con el papel estabilizador de una referencia que permanece fija entre copias." if delta < 0 else
             "En esta configuración la ablación no reduce la media final: el resultado no respalda una mejora de retorno atribuible a target aquí."),
            "Al variar los objetivos de bootstrap cambia también la trayectoria de aprendizaje. Tres semillas no permiten concluir que target sea necesaria o innecesaria universalmente."]
        if min(paired) < 0 < max(paired):
            worst_seed = config['seeds'][int(np.argmin(paired))]
            lines.append(f"El efecto es mixto: mejora en algunas semillas y empeora en otras. La mayor caída es {min(paired):.2f} en la semilla {worst_seed}; no se interpreta la diferencia media como una mejora o caída uniforme.")
    if "base" in by_name and "double" in by_name:
        delta=by_name["double"]["final_mean"]-by_name["base"]["final_mean"]
        paired = [by_name['double']['means_by_seed'][str(seed)] - by_name['base']['means_by_seed'][str(seed)] for seed in config['seeds']]
        lines += ["", "## Tier 3 — Double DQN", "",
            f"Double DQN menos DQN base: {delta:+.2f} puntos de media final.",
            f"Diferencias emparejadas por semilla {config['seeds']}: {', '.join(f'{value:+.2f}' for value in paired)}; mejora en {sum(value > 0 for value in paired)}/3.",
            "La red online elige argmax; target valora esa acción. En DQN estándar target selecciona y valora.",
            "Se mantienen parámetros, inicializaciones por semilla, presupuesto y evaluación del baseline.",
            ("Double DQN aumenta la media observada; la magnitud se debe leer junto con las tres semillas y la evaluación adicional." if delta > 0 else
             "Double DQN no aumenta la media final observada. Separar selección y valoración no garantiza una política mejor con este presupuesto."),
            "La propuesta busca reducir sesgo de sobreestimación; no garantiza mejorar cada entorno o cada semilla."]
    lines += ["", "## Diagnóstico y límites", "",
        "LunarLander tiene recompensas distintas de CartPole. No se usa el límite Q≤100 del análisis anterior.",
        "Los CSV registran pérdidas, extremos Q en 256 estados y una brecha de selección entre online y target.",
        "Esa brecha no es el error Q frente al valor verdadero ni demuestra sobreestimación por sí sola.",
        "Las semillas se usan durante el desarrollo; tres repeticiones son el mínimo, no una garantía universal.",
        "La comparación causal de hiperparámetros depende de mantener el resto del protocolo; revisar config.json por modelo.",
        ("Tier 3 requiere comparar base y Double DQN con tres semillas y parámetros iguales." if tier==3 else
         "Tier 2 requiere ambos estudios (dos parámetros, tres valores cada uno) y la ablación; Tier 3 se documenta por separado."),
        "La brecha entre redes se registra después de las copias programadas y puede ser cero por esa coincidencia temporal.",
        "", "## Pasos hasta el umbral", ""]
    for r in rows:
        lines.append(f"- {r['group']}: "+", ".join(f"{seed}: {step if step is not None else 'not reached'}" for seed,step in zip(config["seeds"],r["first_confirmed_threshold_steps"])))
    diagnostic_file = output / 'diagnostics.json'
    if diagnostic_file.exists():
        measured = [r for r in json.loads(diagnostic_file.read_text(encoding='utf-8')) if r['group'] in by_name]
        declines = [r for r in measured if r['best_observed_100_episode_mean'] >= config['threshold']
                    and r['final_100_episode_mean'] < config['threshold']]
        lines += ['', '## Estabilidad y pérdidas medidas', '',
            'diagnostics.json resume la pérdida del último bloque y los valores Q de diagnóstico por modelo.',
            'La pérdida mide ajuste al objetivo de bootstrap, no calidad de aterrizaje. Una pérdida menor no implica mayor retorno.',
            'El mejor control de 100 solo se usa para describir la trayectoria, nunca para elegir los pesos entregados. No todos los controles intermedios se ampliaron a 100.',
            f'{len(declines)} ejecuciones confirmaron ≥200 en un control anterior y terminaron por debajo de 200.']
        for r in declines:
            lines.append(f"- {r['group']}, semilla {r['seed']}: mejor confirmación observada {r['best_observed_100_episode_mean']:.2f} a {r['best_observed_100_episode_step']} pasos; final {r['final_100_episode_mean']:.2f}.")
        lines += ['La regresión es compatible con el aprendizaje sobre objetivos cambiantes y trayectorias distintas de replay/exploración; los registros por sí solos no aíslan una causa única.']
        lines += ['', f"Valores por semilla, en orden {config['seeds']}. Pérdida: mediana del último bloque de entrenamiento; Q: máximo final en el banco de 256 estados aleatorios de cada semilla.",
                  'Ese banco no estima el valor verdadero de la política ni permite demostrar sobreestimación.', '',
                  '| Grupo | Medianas de pérdida | Máximos Q en el banco |', '|---|---|---|']
        for group in by_name:
            selected = sorted((r for r in measured if r['group'] == group), key=lambda r: config['seeds'].index(r['seed']))
            loss_values = ', '.join(f"{r['last_block_loss_median']:.3f}" for r in selected)
            q_values = ', '.join(f"{r['final_q_probe_max']:.2f}" for r in selected)
            lines.append(f'| {group} | {loss_values} | {q_values} |')
    if rows:
        first=output/rows[0]["group"]/f"seed_{config['seeds'][0]}/config.json"
        machine=json.loads(first.read_text(encoding="utf-8"))
        lines += ["", "## Equipo y reproducción", "",f"Python {machine['python']}; paquetes {machine['packages']}; {machine['platform']}.",
            f"CPU {machine['processor']}, {machine['logical_cpus']} procesadores lógicos; un hilo, sin GPU.",
            "Parámetros: configs/tier23_lunarlander.json. Notebooks: Tier2_LunarLander.ipynb y Tier3_DoubleDQN.ipynb."]
    verified=output/'verification/new_summary.json'
    if verified.exists():
        held=json.loads(verified.read_text(encoding='utf-8'))
        held=[r for r in held if r['group'] in by_name]
        lines += ['', '## Evaluación adicional con nuevos inicios', '',
            'Las condiciones iniciales se fijaron en el protocolo antes de entrenar. No se utilizan para reajustar parámetros ni escoger una semilla.',
            'Base y Double pertenecen a la comparación predefinida. También se evalúa el DQN de mayor media de desarrollo de Tier 2: la elección de esa variante es posterior a desarrollo y está registrada en new_protocol.json.',
            'Son nuevas condiciones iniciales de la misma distribución, no nuevos entrenamientos.', '',
            '| Grupo | Media nueva ± DE | Semillas ≥200 |', '|---|---:|---:|']
        for r in held:
            lines.append(f"| {r['group']} | {r['mean_return']:.2f} ± {r['std_across_seed_means']:.2f} | {r['successful_seeds']}/3 |")
        reproduction=output/'verification/reproduction.json'
        if reproduction.exists():
            verified_rows=json.loads(reproduction.read_text(encoding='utf-8'))
            lines += ['', '## Finalización de los episodios de desarrollo', '',
                'Resting indica que el cuerpo quedó en reposo; body_contact indica contacto del cuerpo; time_limit es corte por tiempo.',
                'Estos conteos describen comportamiento. El criterio del proyecto sigue siendo la recompensa media, no solo el contacto con el suelo.', '',
                'La clasificación prioriza corte por tiempo, contacto del cuerpo, fuera del área y reposo; algunos eventos físicos pueden coincidir.', '',
                '| Grupo | Cuerpo/contacto | Fuera del área | Reposo | Límite de tiempo | Otros |', '|---|---:|---:|---:|---:|---:|']
            for group in by_name:
                selected=[r for r in verified_rows if r['group']==group]
                counts={k:sum(r.get('endings',{}).get(k,0) for r in selected) for k in ['body_contact','outside_viewport','resting','time_limit','other_termination']}
                lines.append(f"| {group} | {counts['body_contact']} | {counts['outside_viewport']} | {counts['resting']} | {counts['time_limit']} | {counts['other_termination']} |")
            lines += ['', 'Para los grupos cuya media final queda por debajo de 200, los finales observados ayudan a concretar el fallo:']
            labels = {'body_contact':'contacto del cuerpo con el suelo', 'outside_viewport':'salida del área',
                      'resting':'reposo del cuerpo', 'time_limit':'agotamiento del límite de tiempo', 'other_termination':'otra terminación'}
            for group, row in by_name.items():
                if row['final_mean'] >= config['threshold']: continue
                selected = [r for r in verified_rows if r['group'] == group]
                counts = {key: sum(r.get('endings',{}).get(key,0) for r in selected) for key in labels}
                dominant = max(counts, key=counts.get)
                total = sum(counts.values())
                lines.append(f"- {group}: {labels[dominant]} en {counts[dominant]}/{total} episodios, la finalización más frecuente. Debe leerse junto con el retorno: reposar no basta para obtener ≥200, y alcanzar el límite no confirma un aterrizaje.")
            lines.append('Son síntomas observados de las políticas finales; no identifican por sí solos una causa única del proceso de aprendizaje.')
    lines += ["", "## Referencias", "",
        "- Gymnasium LunarLander: https://gymnasium.farama.org/environments/box2d/lunar_lander/",
        "- Referencia de parámetros: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/dqn.yml",
        "- Double DQN: van Hasselt, Guez y Silver, https://arxiv.org/abs/1509.06461", ""]
    filename = f"report_tier{tier}.md" if tier else "report.md"
    (output / filename).write_text("\n".join(lines),encoding="utf-8")


def analyze(output):
    output=Path(output)
    config=json.loads((output/"protocol.json").read_text(encoding="utf-8"))
    rows=available(output,config)
    complete={r["group"] for r in rows}
    (output/"figures").mkdir(exist_ok=True)
    studies=[(study_groups(config,k),k,f"LunarLander: estudio de {k}") for k in config["studies"]]
    studies += [([("base","DQN base"),("no_target","Sin red objetivo")],"ablation","Tier 2: ablación de target"),
                ([("base","DQN base"),("double","Double DQN")],"double_dqn","Tier 3: Double DQN")]
    for groups,filename,title in studies:
        if all(name in complete for name,_ in groups):
            plot(output,config,groups,filename,title)
    (output/"summary.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    diagnostics(output, config, rows)
    report(output,config,rows)
    report(output,config,rows,tier=2)
    report(output,config,rows,tier=3)
    print(f"Resumen y figuras generados: {len(rows)}/{len(plan(config))} grupos completos.")
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("results/tier23_lunarlander"))
    analyze(parser.parse_args().output)


if __name__=='__main__':
    main()
