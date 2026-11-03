"""Comprobar dependencias y modelos incluidos, sin entrenar ni abrir ventanas."""

import csv
from importlib import metadata
import json
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    print(f"Python: {sys.version.split()[0]}", flush=True)
    if sys.version_info[:2] != (3, 12):
        print("AVISO: este proyecto se verificó con Python 3.12.", flush=True)
    from packaging.requirements import Requirement

    for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        requirement = Requirement(line)
        installed = metadata.version(requirement.name)
        if installed not in requirement.specifier:
            raise SystemExit(f"Versión diferente: {requirement.name}={installed}; se necesita {requirement.specifier}")
    print("Dependencias: versiones correctas.", flush=True)

    from experiments.mostrar_tier1 import show

    examples = [("tabular", "results/parte_06_bins/bins_10/seed_44"),
                ("dqn", "results/parte_15_mse_completo/seed_44")]
    for kind, relative in examples:
        with (root / relative / "evaluation.csv").open(newline="", encoding="utf-8") as file:
            final = list(csv.DictReader(file))[-1]
        expected = json.loads(final["returns"])[0]
        actual = show(kind, seed=44, episodes=1, headless=True)[0]
        if actual != expected:
            raise SystemExit(f"La demostración {kind} no reprodujo el retorno guardado: {actual} vs {expected}")
    required = ["report/cierre_tier1.md", "results/parte_15_mse_completo/comparacion/tier1_comparacion.png",
                "results/parte_15_mse_completo/evaluacion_nueva/records.json"]
    for relative in required:
        if not (root / relative).is_file():
            raise SystemExit(f"Falta un archivo del Tier 1: {relative}")
    print("\nInstalación comprobada: ambos agentes cargan, reproducen su intento y los resultados están incluidos.")
    print("Puedes abrir Mostrar_Tier1.bat. No fue necesario entrenar.")


if __name__ == "__main__":
    main()
