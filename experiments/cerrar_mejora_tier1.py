"""Verificar modelos finales y evaluar nuevos inicios, sin volver a entrenarlos."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.cuaderno_tier1 import evaluar_guardado
from envs.discretizers import discretization_label


def report(args, fresh_summary):
    comparison_root = getattr(args, "comparison", None) or args.dqn_root / "comparacion"
    comparison = json.loads((comparison_root / "summary.json").read_text(encoding="utf-8"))
    dqn_config = json.loads((args.dqn_root / "seed_42/config.json").read_text(encoding="utf-8"))
    tab_config = json.loads((args.tabular_root / "bins_10/seed_42/config.json").read_text(encoding="utf-8"))
    lines = ["# Tier 1: notebook y mejora de CartPole", "",
        "## Protocolo", "",
        "Implementaciones propias de Q-learning y DQN estándar. Cada agente entrena 200000 pasos",
        "con semillas 42, 43 y 44. Evaluación greedy separada cada 10000 pasos (10 episodios),",
        "evaluación final de 100 episodios y confirmaciones de 100 si un control da ≥475.",
        "Se entrega el modelo final; no se elige el mejor checkpoint ni se descartan semillas.",
        "Las bandas son IC t del 95% puntuales entre tres medias por semilla. Las tablas conservan media ± DE descriptiva; método y límites en intervalos_confianza.md.",
        "Las curvas usan los primeros 10 episodios también al inicio/final; las tablas finales usan 100.",
        "El objetivo del enunciado es ≥475; se registra por separado el umbral declarado por Gymnasium.",
        "", "## Resultados finales", "",
        "| Método | Media ± DE | Retornos por semilla (42, 43, 44) | Semillas ≥475 | Tiempo de entrenamiento |",
        "|---|---:|---|---:|---:|"]
    for row in comparison:
        values = ", ".join(f"{v:.2f}" for v in row["final_returns_by_seed"].values())
        lines.append(f"| {row['method']} | {row['final_mean_100_episodes']:.2f} ± {row['std_across_seed_means']:.2f} | "
                     f"{values} | {row['final_successful_seeds']}/3 | {row['mean_training_seconds']:.2f} s |")
    lines += ["", "Los tiempos excluyen evaluación y son mediciones de esta computadora; no son benchmarks universales.",
              "No se ejecutaron entrenamientos simultáneos. Se realizaron tareas de edición y comprobación durante el trabajo.",
              "", "## Verificación y nuevos inicios", "",
              "Los 100 retornos finales de cada uno de los 12 modelos se reprodujeron exactamente al cargarlo.",
              "Después se evaluaron 100 inicios nuevos por modelo, sin entrenar ni ajustar parámetros.",
              f"La muestra se fijó antes de observarla: semilla de evaluación {getattr(args, 'evaluation_base', 3_000_000)}+1000*semilla_entrenada.",
              "Son nuevos inicios de la misma distribución, no nuevas semillas de entrenamiento.", "",
              "| Método | Media nueva ± DE | Semillas ≥475 |", "|---|---:|---:|"]
    for row in fresh_summary:
        lines.append(f"| {row['method']} | {row['mean_return']:.2f} ± {row['std_across_training_seed_means']:.2f} | {row['models_meeting_475']}/3 |")
    best = max(comparison[:3], key=lambda row: row["final_mean_100_episodes"])
    lines += ["", f"La mayor media tabular entre estas tres granularidades fue {best['method']}: "
              f"{best['final_mean_100_episodes']:.2f}. Esa selección se hace con la media de las tres semillas, no con un episodio.",
              "Si una granularidad no alcanza 475, se informa como resultado negativo; no se reemplaza por otra semilla."]
    lines += ["", "## Qué cambió y qué podemos concluir", "",
        "- DQN conserva replay, target, epsilon-greedy, recompensa original y corrección de cortes temporales.",
        f"- Red: {dqn_config['architecture']}; tasa {dqn_config['learning_rate']}; memoria {dqn_config['buffer_size']}; "
        f"lote {dqn_config['batch_size']}; inicio {dqn_config['learning_starts']}; ajuste cada {dqn_config['train_every']} pasos.",
        f"- Target cada {dqn_config['target_sync_every_environment_steps']}; pérdida {dqn_config['loss']}; "
        f"epsilon hasta {dqn_config['epsilon_end']} en {dqn_config['epsilon_decay_steps']} pasos.",
        "- Configuración DQN inspirada en CleanRL, pero ejecutada con nuestro propio agente. Se mantienen 200000 pasos,",
        "  frente a los 500000 del default de referencia. También difieren generadores aleatorios y clipping de gradiente.",
        f"- Tabular: alpha={tab_config['alpha']}, gamma={tab_config['gamma']}, epsilon final={tab_config['epsilon_end']}; "
        f"límites inferiores={tab_config['low']}; superiores={tab_config['high']}.",
        f"- Inicialización de la tabla Q: {tab_config.get('initial_q', 0.0)} en todas las casillas y acciones; no cambia recompensa ni fórmula Q-learning.",
        f"- Alpha final: {tab_config.get('alpha_end', tab_config['alpha'])}; espaciado de intervalos: potencia {tab_config.get('spacing_power', 1.0)} (1=uniforme).",
        f"- Ejemplo de resolución (configuración angular 10): {tab_config.get('bins_per_dimension', [10]*4)} intervalos por variable; "
        f"{int(np.prod(tab_config.get('bins_per_dimension', [10]*4)))} estados tabulares.",
        "- Se modificaron varios parámetros para buscar una configuración funcional. Esto no es una ablación controlada",
        "  ni permite atribuir toda la mejora a un único cambio. El estudio controlado de Tier 2 sigue pendiente.",
        "- Las semillas 42/43/44 se usaron durante el desarrollo: no constituyen un test independiente de selección.",
        "  La evaluación nueva comprueba otros inicios, sin demostrar éxito universal ni eliminar esa limitación.",
        "- Más intervalos no garantizan mejor aprendizaje: cambian la resolución y el número de casillas por visitar.",
        "- Las granularidades que no alcancen 475 se conservan e informan; la mejor se selecciona por su media entre semillas.",
        "- Los experimentos anteriores permanecen íntegros: DQN MSE anterior=234.10±188.02; mejor tabular anterior=359.70±102.57.",
        "", "## Pasos hasta el objetivo", "",
        "Primer control confirmado de 100 episodios; no identifica el instante exacto de aprendizaje:", ""]
    for row in comparison:
        lines.append(f"- {row['method']}: " + ", ".join(f"{seed}: {value if value is not None else 'not reached'}" for seed, value in zip((42,43,44), row['first_confirmed_threshold_step_by_seed'])))
    import csv
    qmax = []
    for seed in (42, 43, 44):
        with (args.dqn_root / f"seed_{seed}/evaluation.csv").open(encoding="utf-8", newline="") as file:
            qmax.append(float(list(csv.DictReader(file))[-1]["q_probe_max"]))
    lines += ["", "## Diagnóstico de valores Q", "",
              f"Máximos finales en los 256 estados de diagnóstico de cada semilla: {qmax}.",
              "Con gamma=0.99 y recompensa ≤1 el retorno descontado verdadero no supera 100.",
              "Persisten pequeñas sobreestimaciones incluso cuando la política alcanza el objetivo.",
              "La muestra diagnóstica no entrena al agente ni demuestra corrección en todos los estados.",
              "Este valor descontado es distinto del retorno de hasta 500 puntos por episodio."]
    lines += ["", "## Reproducibilidad", "",
        f"Python {dqn_config['python']}; paquetes {dqn_config['packages']}; plataforma {dqn_config['platform']}.",
        f"CPU {dqn_config['processor']}; {dqn_config['logical_cpus']} procesadores lógicos; PyTorch CPU, un hilo.",
        "Parámetros editables en configs/tier1_mejorado.json y notebook Tier1_Explicado.ipynb.",
        "Los comandos están en README.md. El notebook muestra el código real y los archivos guardados.",
        "", "## Referencias y alcance", "",
        "- CleanRL DQN: https://docs.cleanrl.dev/rl-algorithms/dqn/",
        "- Configuración de referencia: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/dqn.py",
        "- Gymnasium CartPole: https://gymnasium.farama.org/environments/classic_control/cart_pole/",
        "", "Este bloque corresponde a Tier 1. Faltan Tier 2, Tier 3 para la pareja, análisis teórico,",
        "integración del artículo final y preparación de defensa. El notebook no sustituye el paper.", ""]
    report_file = args.report or args.output / "report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqn-root", type=Path, required=True)
    parser.add_argument("--tabular-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, help="Opcional: ruta del informe derivado; por defecto se guarda dentro de output")
    parser.add_argument("--comparison", type=Path, help="Carpeta de figuras y summary.json, si difiere de dqn-root/comparacion")
    parser.add_argument("--evaluation-base", type=int, default=3_000_000,
                        help="Base fijada antes de evaluar; no cambiar después para buscar una muestra favorable")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "training_seeds": [42, 43, 44], "episodes_per_model": 100,
        "new_evaluation_function_seed": f"{args.evaluation_base} + 1000 * training_seed",
        "new_environment_reset_seed": f"{args.evaluation_base + 10000} + 1000 * training_seed + episode",
        "purpose": "Verificar modelos finales y medir nuevos inicios tras fijar configuración",
        "model_selection": "Modelo final de 200000 pasos; no mejor checkpoint ni mejor semilla",
        "selection_history": "Configuración elegida tras pruebas exploratorias guardadas; estas semillas de entrenamiento no son un test independiente",
        "new_sample_policy": "Muestra fijada antes de evaluar. No ajustar estos modelos ni seleccionar otra muestra después de verla.",
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    groups = []
    for bins in (6, 10, 20):
        root = args.tabular_root / f"bins_{bins}"
        config = json.loads((root / "seed_42/config.json").read_text(encoding="utf-8"))
        groups.append((discretization_label(config), root))
    groups.append(("DQN ajustado", args.dqn_root))
    records, summary = [], []
    for label, root in groups:
        means = []
        for seed in (42, 43, 44):
            folder = root / f"seed_{seed}"
            expected = json.loads((folder / "summary.json").read_text(encoding="utf-8"))["after"]
            reproduced = evaluar_guardado(folder, episodes=100)
            with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as file:
                import csv
                saved_returns = json.loads(list(csv.DictReader(file))[-1]["returns"])
            if reproduced != saved_returns:
                raise AssertionError(f"No se reprodujeron exactamente los retornos de {folder}")
            assert abs(float(np.mean(reproduced)) - expected) < 1e-9
            fresh = evaluar_guardado(folder, evaluation_seed=args.evaluation_base + 1000 * seed, episodes=100)
            mean = float(np.mean(fresh))
            means.append(mean)
            records.append({"method": label, "training_seed": seed, "source_run": folder.as_posix(),
                            "reproduced_development_mean": expected, "development_returns_exact": True,
                            "new_mean_return": mean, "new_returns": fresh})
            (args.output / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
            print(f"{label}, semilla {seed}: guardado={expected:.2f}, nuevos inicios={mean:.2f}", flush=True)
        summary.append({"method": label, "mean_return": float(np.mean(means)),
                        "std_across_training_seed_means": float(np.std(means)),
                        "means_by_training_seed": dict(zip(["42", "43", "44"], means)),
                        "models_meeting_475": sum(mean >= 475 for mean in means)})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report(args, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
