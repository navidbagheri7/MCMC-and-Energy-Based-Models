"""Archive original exact reference draws only after matching every recorded diagnostic.

Already-published draws are included in the source distribution. Reconstructing
old seeded multivariate normals can vary across LAPACK builds: in that case this
command refuses to replace the archived evidence, rather than relaxing checks.
"""

import json
from pathlib import Path

import numpy as np

from sampling_lab.artifacts import sha256, write_json
from sampling_lab.targets import load_target

NAMES = ["gaussian", "perturbed", "ring", "banana", "funnel", "funnel_noncentered", "regression"]


def main():
    draws = {}
    protocols = {}
    for name in NAMES:
        target = load_target(name)
        sample = target.observable(target.reference(100000, np.random.default_rng(800000)))
        assert sample.shape == (100000, target.dim) and np.isfinite(sample).all()
        matched = []
        for path in sorted(Path("results/reference/sampling").glob("*/metrics.json")):
            metric = json.loads(path.read_text())
            if metric["target"] != name:
                continue
            d = metric["diagnostics"]
            source = Path("assets/source_data") / (path.parent.name + ".npz")
            with np.load(source, allow_pickle=False) as data:
                states = data["observables"]
                np.testing.assert_allclose(
                    sample.mean(0), d["reference_mean"], rtol=1e-10, atol=1e-11
                )
                np.testing.assert_allclose(
                    (states.mean((0, 1)) - sample.mean(0)) / sample.std(0, ddof=1),
                    d["standardized_mean_error"],
                    rtol=1e-10,
                    atol=1e-11,
                )
                covariance = float(
                    np.linalg.norm(np.cov(states.reshape(-1, target.dim).T) - np.cov(sample.T))
                    / np.linalg.norm(np.cov(sample.T))
                )
                np.testing.assert_allclose(
                    covariance, d["covariance_relative_error"], rtol=1e-9, atol=1e-11
                )
            matched.append(path.parent.name)
        assert matched, name
        draws[name] = sample
        protocols[name] = matched
    # All matches must pass before any public file is written.
    output = Path("assets/reference_draws")
    output.mkdir(exist_ok=True)
    records = {}
    for name, sample in draws.items():
        path = output / (name + ".npz")
        np.savez_compressed(path, observables=sample)
        records[name] = {
            "file": str(path),
            "sha256": sha256(path),
            "shape": list(sample.shape),
            "dtype": str(sample.dtype),
            "matched_original_protocols": protocols[name],
        }
    write_json(
        output / "manifest.json",
        {
            "status": "passed",
            "seed": 800000,
            "draws_per_target": 100000,
            "source_sha256": {
                "src/sampling_lab/" + name: sha256(Path("src/sampling_lab") / name)
                for name in ["targets.py", "experiments.py", "diagnostics.py"]
            },
            "targets": records,
            "scope": "Original independent exact draws in observable coordinates, reconstructed and matched to all archived reference means, standardized mean errors and covariance errors before export. These preserve finite Monte Carlo evidence across LAPACK/SVD sign conventions; they are not fitted chains or analytic population moments. Logistic has no invented exact draws. Existing chains, weights and recorded scores are unchanged.",
        },
    )
    print(
        "Archived seven exact-reference arrays after matching",
        sum(map(len, protocols.values())),
        "original protocols",
    )


if __name__ == "__main__":
    main()
