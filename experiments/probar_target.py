"""Prueba corta controlada de copia target cada 500 frente a 2000 pasos."""

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.entrenar_dqn import train


def plot_and_check(root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seeds = (42, 43, 44)
    checks, summaries = [], []
    for seed in seeds:
        first = json.loads((root / f"target_500/seed_{seed}/config.json").read_text(encoding="utf-8"))
        second = json.loads((root / f"target_2000/seed_{seed}/config.json").read_text(encoding="utf-8"))
        changed = sorted(key for key in first.keys() | second.keys() if first.get(key) != second.get(key))
        if changed != ["target_sync_every_environment_steps"]:
            raise ValueError(f"Cambios no controlados: {changed}")
        checks.append({"seed": seed, "changed_fields": changed})
        # La nueva instrumentación no debe alterar la trayectoria original.
        previous = Path("results/parte_10_dqn") / f"seed_{seed}/evaluation.csv"
        with previous.open(encoding="utf-8", newline="") as file:
            old_rows = {int(row["training_steps"]): row for row in csv.DictReader(file)}
        with (root / f"target_500/seed_{seed}/evaluation.csv").open(encoding="utf-8", newline="") as file:
            short_rows = list(csv.DictReader(file))
        for row in short_rows:
            step = int(row["training_steps"])
            if json.loads(row["returns"])[:10] != json.loads(old_rows[step]["returns"])[:10]:
                raise ValueError(f"La referencia corta no reproduce el original en {step} pasos")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    curve_rows = []
    for interval, color in ((500, "#a21caf"), (2000, "#0891b2")):
        curves, q_curves, final_returns, max_qs, times = [], [], [], [], []
        for seed in seeds:
            folder = root / f"target_{interval}/seed_{seed}"
            with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            steps = [int(row["training_steps"]) for row in rows]
            if steps != list(range(0, 50_001, 10_000)):
                raise ValueError("Checkpoints inesperados en la prueba corta")
            curves.append([float(np.mean(json.loads(row["returns"])[:10])) for row in rows])
            q_curves.append([float(row["q_probe_max"]) for row in rows])
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            final_returns.append(summary["after"])
            max_qs.append(float(rows[-1]["q_probe_max"]))
            times.append(summary["training_seconds"])
        values = np.asarray(curves)
        mean, std = values.mean(axis=0), values.std(axis=0)
        label = f"Copia cada {interval:,} pasos".replace(",", ".")
        axes[0].plot(steps, mean, label=label, color=color, linewidth=2)
        axes[0].fill_between(steps, mean - std, mean + std, color=color, alpha=0.15)
        # Cada línea fina corresponde al máximo del banco diagnóstico de una semilla.
        for i, q_values in enumerate(q_curves):
            axes[1].plot(steps, q_values, color=color, alpha=0.65,
                         label=label if i == 0 else None, marker=".")
        for step, average, spread in zip(steps, mean, std):
            curve_rows.append({"target_interval": interval, "training_steps": step,
                               "mean_return_10_episodes": float(average), "std_across_seeds": float(spread)})
        summaries.append({"target_interval": interval,
                          "final_mean_100_episodes": float(np.mean(final_returns)),
                          "std_across_seed_means": float(np.std(final_returns)),
                          "final_returns_by_seed": dict(zip(map(str, seeds), final_returns)),
                          "final_max_q_on_256_probe_states_by_seed": dict(zip(map(str, seeds), max_qs)),
                          "mean_training_seconds": float(np.mean(times))})
    axes[0].set(title="Rendimiento durante la prueba corta",
                ylabel="Retorno medio de evaluación (10 episodios)")
    axes[0].legend(fontsize=9)
    axes[1].axhline(100, linestyle="--", color="#64748b", label="Máximo Q teórico: 100")
    axes[1].set(title="Magnitud de las estimaciones por semilla",
                ylabel="Mayor Q en 256 estados diagnósticos")
    axes[1].set_yscale("symlog", linthresh=1)
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xlabel("Pasos de entrenamiento")
        ax.set_xlim(0, 50_000)
        ax.grid(alpha=0.2)
    fig.suptitle("Solo cambia la copia de target | tasa 0,0005 | exploración decae en 160.000 pasos", fontsize=10)
    fig.savefig(root / "target_prueba_corta.png", dpi=160)
    fig.savefig(root / "target_prueba_corta.pdf")
    plt.close(fig)
    result = {"summaries": summaries, "config_checks": checks,
              "baseline_prefix_reproduces_original": True,
              "note": "50k pasos; no equivale a entrenar 200k ni confirma estabilidad posterior. Curvas de 10 episodios, resumen final de 100."}
    (root / "comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (root / "curves.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/parte_12_target_corto"))
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for interval in (500, 2000):
            for seed in (42, 43, 44):
                if (root / f"target_{interval}/seed_{seed}").exists():
                    parser.error("Ya existen resultados: usar --plot-only u otro --output")
        protocol = {"training_steps": 50_000, "target_intervals": [500, 2000],
                    "learning_rate": 0.0005, "epsilon_decay_steps": 160_000,
                    "seeds": [42, 43, 44], "q_probe_states": 256,
                    "hypothesis": "Copias menos frecuentes podrían frenar el crecimiento de estimaciones Q",
                    "reference": "Tasa original 0.0005; no se usa la configuración fallida de tasa menor",
                    "scope": "Prueba corta antes de decidir sobre entrenamientos completos",
                    "order": "Secuencial: target500 semillas42/43/44, luego target2000 semillas42/43/44"}
        (root / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        for interval in (500, 2000):
            config = SimpleNamespace(output=root / f"target_{interval}", steps=50_000,
                                     learning_rate=0.0005, target_sync_every=interval,
                                     epsilon_decay_steps=160_000)
            for seed in (42, 43, 44):
                train(config, seed)
    plot_and_check(root)


if __name__ == "__main__":
    main()
