"""Resumir los intentos tabulares reales, incluidos los que no mejoraron."""

import json
from pathlib import Path

import numpy as np


def main():
    groups = [("Anterior", Path("results/parte_17_tabular_ajustado"))]
    groups += [(name, Path("results/parte_18_qlearning") / name)
               for name in ("optimista", "centrado", "resolucion", "resolucion_suave")]
    lines = ["# Búsqueda de mejora para Q-learning", "",
        "Cada fila incluye los tres entrenamientos 42/43/44, 200000 pasos por agente y",
        "evaluación final de 100 episodios. Se conserva el modelo final, sin elegir el mejor checkpoint.",
        "No se modificaron las recompensas de CartPole ni la fórmula Q-learning.", "",
        "| Diseño | Intervalos por variable | Q inicial | Alpha inicio → final | Retorno medio ± DE | Semillas ≥475 |",
        "|---|---|---:|---|---:|---:|"]
    for name, root in groups:
        for bins in (6, 10, 20):
            folder = root / f"bins_{bins}"
            paths = [folder / f"seed_{seed}/summary.json" for seed in (42, 43, 44)]
            if not all(path.exists() for path in paths):
                continue
            runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
            config = json.loads((folder / "seed_42/config.json").read_text(encoding="utf-8"))
            values = [r["after"] for r in runs]
            counts = config.get("bins_per_dimension", [bins] * 4)
            alpha = f"{config['alpha']} → {config.get('alpha_end', config['alpha'])}"
            lines.append(f"| {name} | {'/'.join(map(str, counts))} | {config.get('initial_q', 0):g} | "
                         f"{alpha} | {np.mean(values):.2f} ± {np.std(values):.2f} | {sum(v >= 475 for v in values)}/3 |")
    lines += ["", "## Qué se probó", "",
        "1. **Optimista:** cambia solamente Q inicial, de 0 a 100, frente al diseño anterior.",
        "   Con gamma=0.99, 100 es el límite superior del retorno descontado con recompensa ≤1.",
        "   Es una estimación inicial, no una recompensa añadida ni una política correcta precargada.",
        "2. **Centrado:** combina intervalos más finos cerca de cero (potencia 2) y alpha decreciente.",
        "   Esta combinación exploratoria no permite atribuir el resultado a un único factor.",
        "3. **Resolución por variable:** respecto al control optimista, conserva sus parámetros y",
        "   cambia la resolución de posición y velocidad del carrito a 2 intervalos cada una.",
        "   Mantiene 6, 10 o 20 para ángulo y velocidad angular. No se elimina ninguna observación.",
        "4. **Resolución suave:** conserva la tabla compacta e inicialización optimista y reduce alpha",
        "   linealmente de 0.1 a 0.01 durante los 200000 pasos. Frente al control resolución,",
        "   este ensayo modifica el calendario de alpha para investigar estabilidad al final.",
        "", "## Por qué reducir casillas puede ayudar", "",
        "Una tabla uniforme de 20 intervalos por variable contiene 20⁴=160000 estados.",
        "La tabla 2×2×20×20 contiene 1600: cien veces menos, conservando detalle angular.",
        "Esto permite reutilizar más experiencias por casilla. A cambio, pierde detalle de posición",
        "y velocidad: el resultado no garantiza que esa simplificación sea adecuada en cualquier entorno.",
        "El enunciado exige tres granularidades; también se conservan los estudios anteriores con",
        "6/10/20 intervalos en cada una de las cuatro variables.",
        "", "## Diagnóstico y límites", "",
        "La evaluación guardada de las semillas débiles del diseño anterior terminó por inclinación",
        "de la vara. Los detalles y retornos reproducidos están en diagnostico_fallos.json.",
        "Esto orienta la búsqueda, sin probar una causa única de los fallos.",
        "Todas estas semillas se usaron en el desarrollo: no son un test independiente de selección.",
        "La evaluación adicional se fija antes de medirla y no se usa para ajustar la configuración elegida.",
        "Los resultados parciales que subieron a 500 y luego bajaron se conservan en los CSV;",
        "el desempeño final se informa aunque sea peor que un control anterior.",
        "", "## Reproducir los intentos", "",
        "Cada comando usa una carpeta que no debe existir; no reemplaza los resultados entregados.", "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_optimista.json --metodo tabular --output results/repetir_optimista",
        ".\\.venv\\Scripts\\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_centrado.json --metodo tabular --output results/repetir_centrado",
        ".\\.venv\\Scripts\\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_resolucion.json --metodo tabular --output results/repetir_resolucion",
        ".\\.venv\\Scripts\\python.exe -m experiments.cuaderno_tier1 --config configs/qlearning_resolucion_suave.json --metodo tabular --output results/repetir_resolucion_suave",
        ".\\.venv\\Scripts\\python.exe -m experiments.resumir_busqueda_qlearning",
        "```", ""]
    path = Path("report/busqueda_qlearning.md")
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Resumen generado desde resultados completos: {path}")


if __name__ == "__main__":
    main()
