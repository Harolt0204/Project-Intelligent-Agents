"""Ejecutar estudios Tier 2 y Double DQN Tier 3 sin repetir el baseline."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from experiments.lunarlander import ROOT, configuration, plan, fingerprint, train
from experiments.console_utils import utf8_console


def run(config, output, groups=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    planned = plan(config)
    groups = list(planned) if groups is None else groups
    if not groups or any(name not in planned for name in groups):
        raise ValueError(f"Grupos válidos: {list(planned)}")
    if config["steps"] < 1000 or not config["seeds"] or len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Revisar presupuesto y semillas")
    protocol = output / "protocol.json"
    if protocol.exists() and json.loads(protocol.read_text(encoding="utf-8")) != config:
        raise ValueError("La configuración no coincide con esta carpeta; conserva el historial y usa otra")
    protocol.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "plan.json").write_text(json.dumps(planned, indent=2, ensure_ascii=False), encoding="utf-8")
    failures = []
    for name in groups:
        summaries = []
        job = planned[name]
        for seed in config["seeds"]:
            folder = output / name / f"seed_{seed}"
            summary_file = folder / "summary.json"
            if summary_file.exists():
                summary = json.loads(summary_file.read_text(encoding="utf-8"))
                if summary["input_hash"] != fingerprint(config, job, seed):
                    raise ValueError(f"Configuración incompatible: {folder}")
                print(f"Reutilizando resultado completo: {name}, semilla {seed}", flush=True)
            else:
                if folder.exists():
                    raise FileExistsError(f"Hay una ejecución parcial; no se sobrescribe: {folder}")
                try:
                    summary = train(config, job, seed, folder)
                except Exception as error:
                    failures.append({"group": name, "seed": seed, "error": repr(error)})
                    print(f"Fallo documentado en {folder}: {error}", flush=True)
                    continue
            summaries.append(summary)
        if summaries:
            values = [row["after"] for row in summaries]
            aggregate = {"runs": summaries, "mean_after": float(np.mean(values)),
                "std_across_seed_means": float(np.std(values)), "successful_seeds": sum(v >= config["threshold"] for v in values),
                "complete": len(summaries) == len(config["seeds"])}
            (output / name / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
            print(f"{name}: {np.mean(values):.2f} ± {np.std(values):.2f} | {aggregate['successful_seeds']}/{len(values)} semillas ≥200", flush=True)
    if failures:
        (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError("Hubo fallos documentados; consultar failures.json")


def main():
    utf8_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tier23_lunarlander.json")
    parser.add_argument("--output", type=Path, default=Path("results/tier23_lunarlander"))
    parser.add_argument("--groups", nargs="+", help="Omitir para ejecutar los siete grupos; base se reutiliza si está completo")
    parser.add_argument("--tier", choices=("2", "3", "all"), default="all")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    groups = args.groups
    if groups is None and args.tier != "all":
        groups = [name for name, job in plan(config).items() if (args.tier == "2" and name != "double") or (args.tier == "3" and name in ("base", "double"))]
    run(config, args.output, groups)


if __name__ == '__main__':
    main()
