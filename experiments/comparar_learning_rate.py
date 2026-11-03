"""Comparar la primera tasa DQN con una tasa cinco veces menor."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    baseline = Path("results/parte_10_dqn")
    reduced = Path("results/parte_11_lr")
    output = reduced / "comparacion_lr"
    output.mkdir(parents=True, exist_ok=True)
    seeds = (42, 43, 44)
    differences = []
    for seed in seeds:
        before = json.loads((baseline / f"seed_{seed}/config.json").read_text(encoding="utf-8"))
        after = json.loads((reduced / f"seed_{seed}/config.json").read_text(encoding="utf-8"))
        changed = sorted(key for key in before.keys() | after.keys() if before.get(key) != after.get(key))
        if changed != ["learning_rate"] or before["learning_rate"] != 0.0005 or after["learning_rate"] != 0.0001:
            raise ValueError(f"La comparación no cambia exclusivamente learning_rate: {changed}")
        differences.append({"seed": seed, "changed_fields": changed})

    fig, ax = plt.subplots(figsize=(9, 5.5), layout="constrained")
    summaries = []
    curve_rows = []
    for root, label, color in ((baseline, "DQN: tasa 0,0005", "#a21caf"),
                               (reduced, "DQN: tasa 0,0001", "#0891b2"),
                               (Path("results/parte_06_bins/bins_10"), "Q-learning: 10 intervalos (referencia)", "#d97706")):
        curves, finals, diagnostics, times, successes = [], [], [], [], []
        for seed in seeds:
            folder = root / f"seed_{seed}"
            with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            steps = [int(row["training_steps"]) for row in rows]
            if steps != list(range(0, 200_001, 10_000)):
                raise ValueError("Checkpoints incompatibles")
            curves.append([float(np.mean(json.loads(row["returns"])[:10])) for row in rows])
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            finals.append(summary["after"])
            times.append(summary["training_seconds"])
            successes.append(summary.get("first_confirmed_threshold_step"))
            if (folder / "q_diagnostic.json").exists():
                diagnostics.append(json.loads((folder / "q_diagnostic.json").read_text(encoding="utf-8")))
        values = np.asarray(curves)
        mean, std = values.mean(axis=0), values.std(axis=0)
        ax.plot(steps, mean, label=label, color=color, linewidth=2)
        ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.14)
        for step, value, spread in zip(steps, mean, std):
            curve_rows.append({"method": label, "training_steps": step,
                               "mean_return_10_episodes": float(value), "std_across_seeds": float(spread)})
        summaries.append({"method": label, "final_mean_100_episodes": float(np.mean(finals)),
                          "final_std_across_seed_means": float(np.std(finals)),
                          "final_returns_by_seed": dict(zip(map(str, seeds), finals)),
                          "final_successful_seeds": sum(value >= 475 for value in finals),
                          "first_confirmed_threshold_step_by_seed": successes,
                          "mean_training_seconds": float(np.mean(times)),
                          "q_diagnostics": diagnostics})
    ax.axhline(475, color="#64748b", linestyle="--", label="Objetivo: 475 en 100 episodios")
    ax.set(title="CartPole: efecto de reducir la tasa de aprendizaje de DQN",
           xlabel="Pasos de entrenamiento", ylabel="Retorno medio de evaluación (10 episodios)",
           xlim=(0, 200_000))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Solo cambia la tasa DQN | media ± DE entre 3 semillas | sin suavizado", fontsize=10)
    fig.savefig(output / "learning_rate.png", dpi=160)
    fig.savefig(output / "learning_rate.pdf")
    plt.close(fig)
    delta = {str(seed): summaries[1]["final_returns_by_seed"][str(seed)] -
                       summaries[0]["final_returns_by_seed"][str(seed)] for seed in seeds}
    result = {"config_differences": differences, "summaries": summaries,
              "paired_final_return_differences_reduced_minus_baseline": delta,
              "note": "Dos tasas elegidas secuencialmente; no es un barrido completo de Tier 2 ni prueba de generalización"}
    (output / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (output / "curves.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    for summary in summaries:
        print(summary["method"], f"{summary['final_mean_100_episodes']:.2f} ± {summary['final_std_across_seed_means']:.2f}")
    print("Diferencias por semilla (tasa menor - original):", delta)


if __name__ == "__main__":
    main()
