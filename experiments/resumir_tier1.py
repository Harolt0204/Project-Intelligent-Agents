"""Construir el resumen de Tier 1 a partir de resultados reales ya guardados."""

import json
from pathlib import Path


def main():
    root = Path("results/parte_15_mse_completo")
    comparison = json.loads((root / "comparacion/summary.json").read_text(encoding="utf-8"))
    new_evaluation = json.loads((root / "evaluacion_nueva/summary.json").read_text(encoding="utf-8"))
    held = {row["method"]: row for row in new_evaluation}
    config = json.loads((root / "seed_42/config.json").read_text(encoding="utf-8"))
    diagnostics = [json.loads((root / f"seed_{seed}/q_diagnostic.json").read_text(encoding="utf-8"))
                   for seed in (42, 43, 44)]
    lines = [
        "# Cierre experimental de Tier 1 — CartPole",
        "",
        "Este documento consolida los experimentos; no es el artículo final ni completa Tier 2/Tier 3.",
        "",
        "## Protocolo y alcance",
        "",
        "Q-learning propio con 6, 10 y 20 intervalos por variable frente a DQN propio.",
        "Cada ejecución usa 200 000 pasos y semillas independientes 42, 43 y 44.",
        "Las curvas usan pasos de entrenamiento y evaluación greedy separada de 10 episodios por control.",
        "Los resúmenes finales usan 100 episodios por agente. Las bandas son una desviación estándar",
        "entre las tres medias por semilla (ddof=0), sin suavizado; no son intervalos de confianza.",
        "",
        "El DQN MSE se eligió después de pruebas exploratorias. Conserva el calendario original:",
        "red 4-64-64-2, Adam 0.0005, gamma 0.99, replay 50 000, lote 64, ajuste cada cuatro pasos",
        "desde el paso 1 000, copia target cada 500 y epsilon de 1 a 0.05 en 160 000 pasos.",
        "Los controles previos, incluidos los fallos, permanecen guardados. Se entrega el modelo final,",
        "sin sustituirlo por el mejor checkpoint. MSE es una variante de pérdida, no un agente prefabricado.",
        "",
        "## Resultados con las condiciones de evaluación del desarrollo",
        "",
        "| Método | Retorno final medio ± DE | Entrenamiento medio | Modelos finales con media ≥475 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(f"| {row['method']} | {row['final_mean_100_episodes']:.2f} ± {row['std_across_seed_means']:.2f} | "
                     f"{row['mean_training_seconds']:.2f} s | {row['final_successful_seeds']}/3 |")
    lines += ["", "Los tiempos excluyen las evaluaciones y dependen de esta implementación y del equipo.",
              "DQN usa un hilo PyTorch en CPU; no se afirma una diferencia universal de velocidad.",
              "", "### Pasos hasta el umbral", ""]
    for row in comparison:
        steps = row["first_confirmed_threshold_step_by_seed"]
        rendered = "; ".join(f"{seed}: {value if value is not None else 'not reached en los controles de 100 episodios'}"
                             for seed, value in zip((42, 43, 44), steps))
        lines.append(f"- {row['method']}: {rendered}.")
    lines += [
        "", "La detección no localiza el instante exacto de aprendizaje. Tabular comprobó 100 episodios",
        "al inicio y al final; DQN también confirma con 100 cualquier control intermedio de 10 que dé ≥475.",
        "Por esa diferencia no se comparan los tiempos de primer éxito como si su detección fuera idéntica.",
        "", "## Evaluación adicional con condiciones iniciales nuevas", "",
        "Se fijaron antes de ver sus resultados 100 episodios nuevos por modelo, con semillas de reset",
        "1010000 + 1000*semilla_entrenada + episodio. Los métodos comparten condiciones emparejadas,",
        "sin solapamiento de esos intervalos entre las tres semillas entrenadas. No se actualizan pesos",
        "ni tablas ni se usan estos resultados para ajustar los modelos aquí evaluados.",
        "Son nuevos inicios de la misma distribución, no nuevos entrenamientos ni prueba universal.",
        "", "| Método | Retorno nuevo medio ± DE | Modelos con media ≥475 |",
        "|---|---:|---:|",
    ]
    for row in comparison:
        value = held[row["method"]]
        lines.append(f"| {value['method']} | {value['mean_return']:.2f} ± {value['std_across_training_seed_means']:.2f} | "
                     f"{value['models_meeting_475']}/3 |")
    lines += [
        "", "## Interpretación", "",
        "- DQN MSE obtuvo mayor retorno temprano en el primer control de 10 000 pasos que los tabulares;",
        "  eso no demuestra aprendizaje estable ni cumplimiento del umbral. Las curvas muestran pérdida",
        "  posterior de rendimiento. La comparación de eficiencia muestral debe distinguir el inicio del final.",
        "- En tabular, la mejor media final entre las granularidades probadas fue 10 intervalos.",
        "  Más intervalos aportan detalle, pero reparten experiencia entre más casillas; menos intervalos",
        "  agrupan situaciones que pueden requerir acciones distintas. No se afirma que 10 sea óptimo universal.",
        "- El cambio a MSE mejoró la prueba corta de 50 000 pasos en las tres semillas, pero esa mejora",
        "  no permitía anticipar el resultado de 200 000. Se conservan ambos presupuestos sin mezclarlos.",
        "- La sobreestimación persiste en los modelos finales. No se identificó una causa única; las",
        "  auditorías de transiciones, cálculos independientes y un MDP con solución conocida pasaron.",
        "  Esto respalda componentes concretos, sin demostrar corrección total ni estabilidad general.",
        "", "### Diagnóstico de las estimaciones MSE finales", "",
        "Se consultaron 1 000 estados legales de una política aleatoria por semilla. Con recompensa ≤1",
        "y gamma=0.99, el verdadero retorno descontado no supera 100. No es el retorno sin descuento",
        "de un episodio, cuyo máximo es 500. La muestra diagnóstica no se utiliza para entrenar.",
        "", "| Semilla | Menor Q | Mayor Q | Valores Q >100 |",
        "|---|---:|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(f"| {row['seed']} | {row['q_min']:.2f} | {row['q_max']:.2f} | "
                     f"{100 * row['fraction_entries_above_bound']:.2f} % |")
    lines += [
        "", "## Reproducibilidad y evidencia", "",
        f"Python {config['python']}; PyTorch {config['packages']['torch']}; Gymnasium {config['packages']['gymnasium']}.",
        f"Sistema: {config['platform']}. CPU: {config['processor']}; {config['logical_cpus']} procesadores lógicos.",
        "Los archivos config.json registran dependencias, semillas, rangos y parámetros por ejecución.",
        "Los CSV separan entrenamiento y evaluación; se conservan pérdidas, tablas y modelos finales.",
        "La verificación carga los modelos y reproduce sus evaluaciones finales. El video usa la primera",
        "semilla y el primer episodio de evaluación, no el mejor intento ni un promedio.",
        "", "Figuras y métricas: results/parte_15_mse_completo/comparacion.",
        "Evaluación nueva: results/parte_15_mse_completo/evaluacion_nueva.",
        "Diagnósticos y pesos: results/parte_15_mse_completo/seed_*.",
        "Historial de pruebas: report/notas_discretizacion.md, notas_dqn_cartpole.md, notas_learning_rate.md,",
        "notas_target_corto.md, auditoria_dqn.md y notas_loss.md. El uso de IA está en report/uso_ia.md.",
        "", "## Estado del proyecto", "",
        "La comparación experimental de Tier 1 queda consolidada con sus resultados y limitaciones.",
        "Faltan Tier 2 (LunarLander, estudios y ablación), Tier 3 para la pareja (Double DQN), análisis",
        "teórico general, integración del artículo PDF de 6–8 páginas y preparación de la defensa.",
        "Este cierre no afirma que todos los agentes resolvieran el entorno ni que el proyecto completo esté terminado.",
    ]
    report = Path("report/cierre_tier1.md")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Resumen generado desde los resultados reales: {report}")


if __name__ == "__main__":
    main()
