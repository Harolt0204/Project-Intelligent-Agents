"""Graficar DQN y las tres discretizaciones tabulares a igual presupuesto."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dqn-root", type=Path, default=Path("results/parte_10_dqn"))
    parser.add_argument("--tabular-root", type=Path, default=Path("results/parte_06_bins"))
    parser.add_argument("--dqn-label", default="DQN")
    parser.add_argument("--extra-dqn-root", type=Path)
    parser.add_argument("--extra-dqn-label", default="DQN adicional")
    args = parser.parse_args()
    output = args.dqn_root / "comparacion"
    output.mkdir(parents=True, exist_ok=True)
    groups = [(f"Q-learning ({bins} intervalos)", args.tabular_root / f"bins_{bins}")
              for bins in (6, 10, 20)] + [(args.dqn_label, args.dqn_root)]
    if args.extra_dqn_root is not None:
        groups.append((args.extra_dqn_label, args.extra_dqn_root))
    colors = ["#2563eb", "#d97706", "#15803d", "#a21caf", "#64748b"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    summaries, curve_rows = [], []
    for (label, root), color in zip(groups, colors):
        curves, training_curves, finals, times, first_confirmed = [], [], [], [], []
        for seed in (42, 43, 44):
            folder = root / f"seed_{seed}"
            config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            if config["training_steps"] != 200_000 or config["seed"] != seed:
                raise ValueError("Presupuestos o semillas incompatibles")
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            with (folder / "evaluation.csv").open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            steps = [int(row["training_steps"]) for row in rows]
            if steps != list(range(0, 200_001, 10_000)):
                raise ValueError("Los checkpoints no coinciden")
            curves.append([float(np.mean(json.loads(row["returns"])[:10])) for row in rows])
            with (folder / "training.csv").open(newline="", encoding="utf-8") as file:
                train_rows = list(csv.DictReader(file))
            train_curve = []
            for end in range(10_000, 200_001, 10_000):
                values = [float(row["return"]) for row in train_rows
                          if end - 10_000 < int(row["training_steps"]) <= end
                          and row["budget_cut"] == "False"]
                train_curve.append(float(np.mean(values)) if values else float("nan"))
            training_curves.append(train_curve)
            finals.append(summary["after"])
            times.append(summary["training_seconds"])
            first_confirmed.append(summary.get("first_confirmed_threshold_step"))
        for ax, values, xs in ((axes[0], np.asarray(curves), steps),
                               (axes[1], np.asarray(training_curves), list(range(10_000, 200_001, 10_000)))):
            mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0)
            ax.plot(xs, mean, color=color, linewidth=2, label=label)
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.12)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Pasos de entrenamiento")
            ax.set_xlim(0, 200_000)
            for step, average, spread in zip(xs, mean, std):
                curve_rows.append({"method": label, "kind": "evaluation" if ax is axes[0] else "training",
                                   "training_steps": step, "mean_return": float(average),
                                   "std_across_seeds": float(spread)})
        summaries.append({"method": label, "final_mean_100_episodes": float(np.mean(finals)),
                          "std_across_seed_means": float(np.std(finals)),
                          "final_returns_by_seed": dict(zip(["42", "43", "44"], finals)),
                          "mean_training_seconds": float(np.mean(times)),
                          "final_successful_seeds": sum(value >= 475 for value in finals),
                          "first_confirmed_threshold_step_by_seed": first_confirmed,
                          "threshold_note": "Tabular: solo se comprobó con 100 episodios al inicio/final; DQN: confirmaciones adicionales"})
    axes[0].axhline(475, color="#64748b", linestyle="--", label="Objetivo: 475 en 100 episodios")
    axes[0].set_title("Evaluación sin exploración")
    axes[0].set_ylabel("Retorno medio (10 episodios por checkpoint)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="outside lower center", ncol=3)
    axes[1].set_title("Entrenamiento con exploración")
    axes[1].set_ylabel("Retorno por episodio completo; bloques de 10.000 pasos")
    fig.suptitle("CartPole: mismo presupuesto de 200.000 pasos | media ± DE entre 3 semillas", fontsize=11)
    fig.savefig(output / "tier1_comparacion.png", dpi=160)
    fig.savefig(output / "tier1_comparacion.pdf")
    plt.close(fig)
    for filename, rows in (("summary.csv", summaries), ("curves.csv", curve_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
