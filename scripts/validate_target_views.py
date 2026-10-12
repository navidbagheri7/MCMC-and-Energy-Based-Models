"""Independently verify target contours, focus bounds and exact-draw marginals."""

import json
import re
from pathlib import Path

import numpy as np
from reference_draws import load_reference
from scipy.special import logsumexp
from scipy.stats import multivariate_normal, norm

from sampling_lab.artifacts import write_json
from sampling_lab.targets import load_target


def main():
    views = json.loads(Path("assets/explorer/target_views.json").read_text())
    evidence = json.loads(
        re.search(
            r'<script[^>]*id="evidence"[^>]*>(.*?)</script>',
            Path("docs/explorer.html").read_text(),
            re.S,
        ).group(1)
    )
    assert evidence["targets"] == views
    checked = []
    for name, view in views.items():
        target = load_target(name)
        if name == "logistic":
            assert view["density"] is None and view["marginal_reference"] is None
            expected = np.stack(
                [
                    target.mean[:2] - 4 * np.sqrt(np.diag(target.cov)[:2]),
                    target.mean[:2] + 4 * np.sqrt(np.diag(target.cov)[:2]),
                ],
                axis=1,
            )
            np.testing.assert_allclose(view["bounds"], expected, rtol=0, atol=1e-12)
            checked.append(
                {
                    "target": name,
                    "scope": "Approximate viewing window only; no exact reference invented",
                }
            )
            continue
        samples = load_reference(name)[:, :2]
        bounds = np.quantile(samples, [0.001, 0.999], axis=0).T
        np.testing.assert_array_equal(view["bounds"], bounds)
        x, y = np.meshgrid(*(np.linspace(*b, 91) for b in bounds))
        points = np.stack([x.ravel(), y.ravel()], axis=1)
        if name in ["gaussian", "regression", "perturbed"]:
            logp = multivariate_normal.logpdf(points, mean=target.mean[:2], cov=target.cov[:2, :2])
            if name == "perturbed":
                logp += np.cos(points[:, 0]) + np.sin(points[:, 1])
        elif name == "ring":
            logp = logsumexp(
                [
                    multivariate_normal.logpdf(points, mean=c, cov=np.eye(2) * 0.35**2)
                    for c in target.centers
                ],
                axis=0,
            ) - np.log(8)
        elif name == "banana":
            logp = norm.logpdf(points[:, 0], scale=2) + norm.logpdf(
                points[:, 1] - 0.25 * (points[:, 0] ** 2 - 4)
            )
        else:
            logp = norm.logpdf(points[:, 0], scale=3) + norm.logpdf(
                points[:, 1], scale=np.exp(points[:, 0] / 2)
            )
        np.testing.assert_allclose(
            view["density"], np.exp(logp - logp.max()).reshape(x.shape), rtol=0, atol=5.01e-10
        )
        for axis, marginal in enumerate(view["marginal_reference"]):
            edges = np.linspace(*bounds[axis], 41)
            counts, _ = np.histogram(samples[:, axis], edges)
            np.testing.assert_array_equal(marginal["edges"], edges)
            np.testing.assert_array_equal(
                marginal["density"], counts / (len(samples) * np.diff(edges))
            )
        checked.append(
            {
                "target": name,
                "density_grid_values": 91**2,
                "exact_reference_draws": len(samples),
                "reference_seed": 800000,
            }
        )
    write_json(
        "docs/target_views_validation.json",
        {
            "status": "passed",
            "targets": checked,
            "scope": "Embedded target evidence equals its public JSON. Every contour-grid value matches an independent SciPy density expression; focus bounds and both reference marginals match hash-bound original exact draws. Regression uses a covariance submatrix for a true marginal; transformed funnels use original-space density. Logistic has no invented exact marginal.",
        },
    )
    print("Verified", len(checked), "target views and", 7 * 91**2, "density-grid values")


if __name__ == "__main__":
    main()
