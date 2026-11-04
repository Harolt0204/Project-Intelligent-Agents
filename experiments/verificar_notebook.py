"""Ejecutar todas las celdas del notebook con el Python del proyecto."""

import os
import argparse
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--animacion", action="store_true", help="Comprobar la celda opcional del carrito en un notebook temporal")
    parser.add_argument("--agente", choices=("dqn", "tabular"), default="dqn")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    runtime = root / ".jupyter_runtime"
    runtime.mkdir(exist_ok=True)
    os.environ["JUPYTER_RUNTIME_DIR"] = str(runtime)
    path = root / "Tier1_Explicado.ipynb"
    nb = nbformat.read(path, as_version=4)
    if args.animacion:
        setup = next(c.source for c in nb.cells if c.cell_type == "code" and "from pathlib import Path" in c.source)
        demo = next(c.source for c in nb.cells if c.cell_type == "code" and "VER_CARRITO = False" in c.source)
        demo = demo.replace("VER_CARRITO = False", "VER_CARRITO = True")
        demo = demo.replace("AGENTE = 'dqn'", f"AGENTE = '{args.agente}'")
        nb.cells = [nbformat.v4.new_code_cell(setup), nbformat.v4.new_code_cell("ACTUAL = actual()\n" + demo)]
    nbformat.validate(nb)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"cell_{i}", "exec")
    km = KernelManager(kernel_name="python3")
    km.kernel_spec.argv[0] = sys.executable
    def progress(cell, cell_index):
        print(f"Ejecutando celda {cell_index + 1}/{len(nb.cells)}", flush=True)
    client = NotebookClient(nb, km=km, timeout=180, startup_timeout=60,
                            on_cell_start=progress, resources={"metadata": {"path": str(root)}})
    try:
        client.execute()
    finally:
        if km.has_kernel:
            km.shutdown_kernel(now=True)
        km.cleanup_resources()
    if args.animacion:
        html = [out["data"]["text/html"] for cell in nb.cells for out in cell.get("outputs", [])
                if "data" in out and "text/html" in out["data"]]
        assert any("data:image/png;base64" in item for item in html), "No se encontró la animación HTML"
        check = {"kernel_executable": sys.executable, "agent": args.agente, "animation_html_verified": True,
                 "html_characters": [len(item) for item in html]}
        filename = "notebook_animation_check.json" if args.agente == "dqn" else "notebook_animation_check_tabular.json"
        (root / "results" / filename).write_text(json.dumps(check, indent=2), encoding="utf-8")
    else:
        nbformat.write(nb, path)
    print(f"Notebook verificado: {sum(c.cell_type == 'code' for c in nb.cells)} celdas ejecutadas sin errores")


if __name__ == "__main__":
    main()
