"""Comparación de granularidades con iguales pasos, parámetros y semillas."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.entrenar_tabular import train


def summarize_and_plot(output, bins_list, seeds):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_protocol = {"curve_eval_episodes": 10, "final_summary_eval_episodes": 100,
                       "seeds": seeds, "band": "standard deviation, ddof=0",
                       "smoothing": False, "matplotlib": matplotlib.__version__}
    (output / "figure_protocol.json").write_text(json.dumps(figure_protocol, indent=2), encoding="utf-8")

    summaries = []
    curve_rows = []
    fig, ax = plt.subplots(figsize=(9, 5.3), layout="constrained")
    colors = ["#2563eb", "#d97706", "#15803d"]
    for bins, color in zip(bins_list, colors):
        curves = []
        finals = []
        seconds = []
        visited = []
        for seed in seeds:
            folder = output / f"bins_{bins}" / f"seed_{seed}"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            steps = np.array([int(row["training_steps"]) for row in rows])
            if curves and not np.array_equal(steps, reference_steps):
                raise ValueError("Las curvas no tienen los mismos checkpoints")
            reference_steps = steps
            # Usar siempre 10 episodios en las curvas, incluso cuando el log
            # inicial/final contiene 100. El resumen final sí utiliza los 100.
            curves.append([float(np.mean(json.loads(row["returns"])[:10])) for row in rows])
            finals.append(summary["after"])
            seconds.append(summary["training_seconds"])
            visited.append(summary["visited_states"])
        values = np.asarray(curves)
        mean = values.mean(axis=0)
        std = values.std(axis=0, ddof=0)
        ax.plot(steps, mean, label=f"{bins} intervalos por variable", color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        for step, value, spread in zip(steps, mean, std):
            curve_rows.append({"bins": bins, "training_steps": int(step),
                               "mean_return": float(value), "std_across_seeds": float(spread)})
        summaries.append({"bins": bins, "states": bins ** 4,
                          "final_mean": float(np.mean(finals)),
                          "final_std_across_seeds": float(np.std(finals)),
                          "final_return_by_seed": dict(zip(map(str, seeds), finals)),
                          "mean_training_seconds": float(np.mean(seconds)),
                          "mean_visited_states": float(np.mean(visited)),
                          "seeds_meeting_final_threshold": sum(value >= 475 for value in finals)})
    ax.axhline(475, color="#64748b", linestyle="--", label="Objetivo del proyecto: 475")
    ax.set(title="CartPole: Q-learning con tres discretizaciones",
           xlabel="Pasos de entrenamiento", ylabel="Retorno medio de evaluación (10 episodios)",
           xlim=(0, max(steps)))
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Media ± desviación estándar entre 3 semillas; sin suavizado", fontsize=10)
    fig.savefig(output / "curvas_discretizacion.png", dpi=160)
    fig.savefig(output / "curvas_discretizacion.pdf")
    plt.close(fig)
    for name, rows in (("summary.csv", summaries), ("curves.csv", curve_rows)):
        with (output / name).open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print("\nRESULTADOS FINALES", flush=True)
    for row in summaries:
        print(f"{row['bins']} intervalos: {row['final_mean']:.2f} ± "
              f"{row['final_std_across_seeds']:.2f}; "
              f"tiempo medio de entrenamiento: {row['mean_training_seconds']:.1f} s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/parte_06_bins"))
    parser.add_argument("--plot-only", action="store_true", help="Regenerar figuras sin entrenar")
    args = parser.parse_args()
    bins_list = [6, 10, 20]
    seeds = [42, 43, 44]
    # Protocolo fijado antes de mirar los resultados: solo cambia la granularidad.
    steps = 200_000
    eval_every = 10_000
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for bins in bins_list:
            for seed in seeds:
                run = args.output / f"bins_{bins}" / f"seed_{seed}"
                if run.exists():
                    parser.error(f"Ya existe {run}; usa otro --output o --plot-only")
        protocol = {"bins": bins_list, "seeds": seeds, "training_steps_per_run": steps,
                    "eval_every": eval_every, "alpha": 0.1, "gamma": 0.99,
                    "epsilon": "1 -> 0.05 durante el primer 80% de pasos",
                    "periodic_eval_episodes": 10, "initial_final_eval_episodes": 100,
                    "band": "Desviación estándar entre medias de semillas, ddof=0",
                    "curve_eval_episodes": "Primeros 10 en todos los checkpoints; resumen final usa 100",
                    "training_order": "6, 10, 20 bins; semillas 42, 43, 44; secuencial",
                    "note": "Comparación tabular; DQN y análisis final de Tier 1 pendientes"}
        (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        for bins in bins_list:
            print(f"\n=== {bins} INTERVALOS POR VARIABLE ===", flush=True)
            config = SimpleNamespace(output=args.output / f"bins_{bins}", bins=bins,
                                     steps=steps, eval_every=eval_every,
                                     purpose="Tier 1: comparación tabular de granularidades")
            for seed in seeds:
                train(config, seed)
    summarize_and_plot(args.output, bins_list, seeds)


if __name__ == "__main__":
    main()
