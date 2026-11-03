"""Comparar MSE con los controles Smooth L1 ya registrados, a 50k pasos."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.entrenar_dqn import train


BASELINE = Path("results/parte_12_target_corto/target_500")
SEEDS = (42, 43, 44)


def plot_and_check(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config_checks = []
    for seed in SEEDS:
        first = json.loads((BASELINE / f"seed_{seed}/config.json").read_text(encoding="utf-8"))
        second = json.loads((root / f"mse/seed_{seed}/config.json").read_text(encoding="utf-8"))
        changed = sorted(key for key in first.keys() | second.keys() if first.get(key) != second.get(key))
        if changed != ["loss"] or first["loss"] != "Smooth L1" or second["loss"] != "MSE":
            raise ValueError(f"La comparación cambia otros parámetros: {changed}")
        config_checks.append({"seed": seed, "changed_fields": changed})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    summaries, curve_rows = [], []
    for label, folder_root, color in (("Smooth L1", BASELINE, "#a21caf"),
                                       ("MSE", root / "mse", "#0891b2")):
        curves, q_curves, finals, times, successes = [], [], [], [], []
        for seed in SEEDS:
            folder = folder_root / f"seed_{seed}"
            with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            steps = [int(row["training_steps"]) for row in rows]
            if steps != list(range(0, 50_001, 10_000)):
                raise ValueError("Los presupuestos o checkpoints no coinciden")
            curves.append([float(np.mean(json.loads(row["returns"])[:10])) for row in rows])
            q_curves.append([float(row["q_probe_max"]) for row in rows])
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            finals.append(summary["after"])
            times.append(summary["training_seconds"])
            successes.append(summary["first_confirmed_threshold_step"])
        values = np.asarray(curves)
        mean, std = values.mean(axis=0), values.std(axis=0)
        axes[0].plot(steps, mean, color=color, label=label, linewidth=2)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        for i, values_q in enumerate(q_curves):
            axes[1].plot(steps, values_q, color=color, alpha=0.65, marker=".", label=label if i == 0 else None)
        for step, average, spread in zip(steps, mean, std):
            curve_rows.append({"loss": label, "training_steps": step,
                               "mean_return_10_episodes": float(average), "std_across_seeds": float(spread)})
        summaries.append({"loss": label, "source_root": str(folder_root),
                          "final_mean_100_episodes": float(np.mean(finals)),
                          "std_across_seed_means": float(np.std(finals)),
                          "final_returns_by_seed": dict(zip(map(str, SEEDS), finals)),
                          "final_max_q_on_256_probe_states_by_seed": dict(zip(map(str, SEEDS), [q[-1] for q in q_curves])),
                          "final_successful_seeds": sum(value >= 475 for value in finals),
                          "first_confirmed_success_steps": dict(zip(map(str, SEEDS), successes)),
                          "mean_training_seconds": float(np.mean(times))})
    axes[0].set(title="Rendimiento durante la prueba corta", ylabel="Retorno medio (10 episodios)")
    axes[0].axhline(475, color="#64748b", linestyle="--", label="Objetivo: 475 en 100 episodios")
    axes[1].set(title="Magnitud de las estimaciones por semilla", ylabel="Mayor Q en 256 estados diagnósticos")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].axhline(100, color="#64748b", linestyle="--", label="Máximo Q teórico: 100")
    for ax in axes:
        ax.set(xlabel="Pasos de entrenamiento", xlim=(0, 50_000))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Solo cambia la pérdida | tasa 0,0005 | target cada 500 | 3 semillas", fontsize=10)
    fig.savefig(root / "loss_prueba_corta.png", dpi=160)
    fig.savefig(root / "loss_prueba_corta.pdf")
    plt.close(fig)
    differences = {str(seed): summaries[1]["final_returns_by_seed"][str(seed)] -
                             summaries[0]["final_returns_by_seed"][str(seed)] for seed in SEEDS}
    result = {"summaries": summaries, "config_checks": config_checks,
              "paired_return_differences_mse_minus_smooth_l1": differences,
              "note": "Prueba de 50k; no demuestra estabilidad a 200k. Las pérdidas numéricas MSE y Smooth L1 no se comparan como si tuvieran la misma escala."}
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (root / "curves.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/parte_14_loss"))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for seed in SEEDS:
            if not (BASELINE / f"seed_{seed}/summary.json").exists():
                parser.error("Falta un control Smooth L1 verificado")
            if (root / f"mse/seed_{seed}").exists():
                parser.error("Ya existen resultados: usar --plot-only u otro --output")
        protocol = {"training_steps": 50_000, "losses": ["Smooth L1", "MSE"],
                    "baseline_root": str(BASELINE), "reuse_existing_verified_controls": True,
                    "seeds": list(SEEDS), "learning_rate": 0.0005, "target_sync_every": 500,
                    "epsilon_decay_steps": 160_000, "q_probe_states": 256,
                    "hypothesis": "MSE podría cambiar la respuesta a errores grandes y mejorar rendimiento",
                    "data_note": "Mismas reglas y semillas de recolección; las experiencias pueden divergir cuando cambian las acciones aprendidas",
                    "scope": "Prueba corta; no selección del mejor checkpoint ni entrenamiento largo"}
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        config = SimpleNamespace(output=root / "mse", steps=50_000, learning_rate=0.0005,
                                 target_sync_every=500, epsilon_decay_steps=160_000, loss="mse")
        for seed in SEEDS:
            train(config, seed)
    plot_and_check(root)


if __name__ == "__main__":
    main()
