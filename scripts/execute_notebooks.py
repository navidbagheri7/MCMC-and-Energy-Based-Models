"""Execute every authored notebook in a fresh kernel, preserving real outputs."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


class CurrentInterpreterClient(NotebookClient):
    """Keep notebook dependencies identical to the interpreter running this script."""

    def create_kernel_manager(self):
        manager = super().create_kernel_manager()
        manager.kernel_spec.argv[0] = sys.executable
        return manager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    records = []
    for path in sorted((root / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        CurrentInterpreterClient(
            notebook,
            timeout=args.timeout,
            kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        ).execute()
        nbformat.write(notebook, path)
        code = [cell for cell in notebook.cells if cell.cell_type == "code"]
        records.append(
            {
                "notebook": str(path.relative_to(root)),
                "executed_code_cells": len(code),
                "all_executed": all(cell.execution_count is not None for cell in code),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "visual_outputs": sum(
                    any(key.startswith("image/") for key in output.get("data", {}))
                    for cell in code
                    for output in cell.outputs
                ),
            }
        )
        print("Executed", path.name, flush=True)
    (root / "docs/notebook_execution.json").write_text(
        json.dumps({"status": "passed", "notebooks": records}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
