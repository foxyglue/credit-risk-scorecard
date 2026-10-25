"""Run 01–04 sequentially, each with a fresh kernel; persist outputs even on error."""
import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager


def main():
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, choices=range(1, 5), default=1)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    # A workspace-local kernel uses exactly the interpreter that runs this script.
    kernel_root = root / ".venv" / "notebook-kernels"
    spec_dir = kernel_root / "credit-risk"
    spec_dir.mkdir(parents=True, exist_ok=True)
    (spec_dir / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Credit Risk (execution environment)", "language": "python",
    }), encoding="utf-8")
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    paths = sorted((root / "notebooks").glob("0[1-4]_*.ipynb"))
    if len(paths) != 4 or {int(path.name[:2]) for path in paths} != {1, 2, 3, 4}:
        raise FileNotFoundError("Expected exactly notebooks 01–04 before execution.")
    for path in paths:
        if int(path.name[:2]) < args.start:
            continue
        print(f"START {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        client = NotebookClient(
            notebook, timeout=None, kernel_name="credit-risk",
            resources={"metadata": {"path": str(root / "notebooks")}},
        )
        client.km = client.create_kernel_manager()
        client.km.kernel_spec_manager = KernelSpecManager(kernel_dirs=[str(kernel_root)])
        try:
            client.execute()
        finally:
            notebook.metadata["kernelspec"] = {
                "display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3",
            }
            nbformat.write(notebook, path)
        print(f"DONE {path.name}", flush=True)
    packages = ["pandas", "numpy", "scikit-learn", "lightgbm", "optbinning", "optuna",
                "pyarrow", "matplotlib", "seaborn", "joblib", "nbformat", "nbclient", "ipykernel", "pyzmq"]
    versions = {name: importlib.metadata.version(name) for name in packages}
    (root / "requirements-tested.txt").write_text(
        "# Tested on Python " + sys.version.split()[0] + "\n" +
        "\n".join(f"{name}=={version}" for name, version in versions.items()) + "\n",
        encoding="utf-8",
    )
    print("ALL NOTEBOOKS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
