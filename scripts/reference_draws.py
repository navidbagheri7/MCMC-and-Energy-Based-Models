"""Load hash-bound original reference draws without CPU-dependent SVD regeneration."""

import json
from pathlib import Path

import numpy as np

from sampling_lab.artifacts import sha256


def load_reference(name, root=Path(".")):
    root = Path(root)
    manifest = json.loads((root / "assets/reference_draws/manifest.json").read_text())
    assert manifest["status"] == "passed" and manifest["seed"] == 800000
    for file, digest in manifest["source_sha256"].items():
        assert sha256(root / file) == digest, file
    record = manifest["targets"][name]
    path = root / record["file"]
    assert sha256(path) == record["sha256"]
    with np.load(path, allow_pickle=False) as data:
        samples = data["observables"].copy()
    assert list(samples.shape) == record["shape"] and samples.dtype == np.float64
    assert len(samples) == manifest["draws_per_target"] and np.isfinite(samples).all()
    return samples
