"""Portable evidence, atomic JSON and reproducible source identities."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")

    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))

    temp.write_text(json.dumps(value, default=convert, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def source_identity():
    folder = Path(__file__).parent
    return {
        p.name: sha256(p)
        for p in sorted(folder.glob("*.py"))
        if p.name not in ["cli.py", "reporting.py", "__init__.py"]
    }


def environment():
    import importlib.metadata
    import platform

    return dict(
        python=platform.python_version(),
        packages={
            name: importlib.metadata.version(name)
            for name in ["numpy", "scipy", "torch", "arviz", "pyro-ppl", "scikit-learn"]
        },
        device="cpu",
        precision="float64 sampling and Pyro potential; float32 neural energies with float64 quadrature weights",
        threads=1,
    )
