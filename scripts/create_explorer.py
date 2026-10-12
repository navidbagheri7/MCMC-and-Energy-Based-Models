"""Offline sampling/energy explorer, with real chain states and live mechanics."""

import base64
import json
from pathlib import Path

import numpy as np

from sampling_lab.artifacts import sha256, write_json
from sampling_lab.targets import Target, load_target


def target_views():
    """Independent target geometry and exact-draw marginals, never fitted chains."""
    views = {}
    for name in [
        "gaussian",
        "perturbed",
        "ring",
        "banana",
        "funnel",
        "funnel_noncentered",
        "regression",
        "logistic",
    ]:
        target = load_target(name)
        if name == "logistic":
            limits = np.stack(
                [
                    target.mean[:2] - 4 * np.sqrt(np.diag(target.cov)[:2]),
                    target.mean[:2] + 4 * np.sqrt(np.diag(target.cov)[:2]),
                ],
                axis=1,
            )
            views[name] = dict(
                bounds=limits,
                density=None,
                marginal_reference=None,
                scope="First two posterior coefficients. Focus window uses training-only MAP ± four Laplace standard deviations as a viewing aid; no exact 2D posterior density or reference sample is available.",
            )
            continue
        reference = target.observable(target.reference(100000, np.random.default_rng(800000)))[
            :, :2
        ]
        limits = np.quantile(reference, [0.001, 0.999], axis=0).T
        axis_x, axis_y = [np.linspace(*bounds, 91) for bounds in limits]
        xx, yy = np.meshgrid(axis_x, axis_y)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1)
        density_target = (
            Target("funnel")
            if name == "funnel_noncentered"
            else Target("regression", 2, target.mean[:2], target.cov[:2, :2])
            if name == "regression"
            else target
        )
        log_density = density_target.log_prob(points)
        relative = np.exp(log_density - log_density.max()).reshape(xx.shape)
        marginals = []
        for axis in range(2):
            edges = np.linspace(*limits[axis], 41)
            count, _ = np.histogram(reference[:, axis], edges)
            marginals.append(dict(edges=edges, density=count / (len(reference) * np.diff(edges))))
        views[name] = dict(
            bounds=limits,
            density=np.round(relative, 9),
            marginal_reference=marginals,
            reference_count=len(reference),
            reference_seed=800000,
            scope="Contour levels show relative target density, not credible-region probabilities. Focus bounds use the 0.1% and 99.9% quantiles of 100,000 independent exact reference draws. Reference marginal curves are finite-sample density histograms. States outside this view are counted explicitly; Full extent shows every saved state.",
        )
    return views


def main():
    rows, bindings = [], {}
    for family in ["sampling", "ebm"]:
        for path in sorted((Path("results/reference") / family).glob("*/metrics.json")):
            m = json.loads(path.read_text())
            run = json.loads(path.with_name("run.json").read_text())
            cfg = m["config"]
            source = Path(run["numeric_source"])
            name = path.parent.name
            with np.load(source, allow_pickle=False) as data:
                if family == "sampling":
                    target = cfg["target"]
                    group = (
                        "Warm starts"
                        if cfg["profile"] == "warm"
                        else "Bayesian regression"
                        if target == "regression"
                        else "Bayesian classification"
                        if target == "logistic"
                        else "Multimodality and tempering"
                        if target == "ring"
                        else "Geometry and reparameterization"
                        if target in ["banana", "funnel", "funnel_noncentered"]
                        else "Chain diagnostics"
                    )
                    chain = data["observables"]
                    rows.append(
                        dict(
                            group=group,
                            task=target,
                            preset=f"{cfg['method']} · {cfg['profile']} · preset {cfg['step_size']:g}",
                            name=name,
                            metrics=m,
                            points_b64=base64.b64encode(
                                np.ascontiguousarray(chain[:, :, :2], dtype="<f8").tobytes()
                            ).decode(),
                            shape=list(chain[:, :, :2].shape),
                            image="../assets/figures/" + name + ".png",
                            source="../" + str(path),
                            numeric="../" + str(source),
                            kind="sampling",
                        )
                    )
                else:
                    rows.append(
                        dict(
                            group="Energy-based learning",
                            task=cfg["task"],
                            preset=f"{cfg['method']} · fit seed {cfg['seed']}",
                            name=name,
                            metrics=m,
                            points=[],
                            image="../assets/figures/" + name + ".png",
                            source="../" + str(path),
                            numeric="../" + str(source),
                            kind="energy",
                        )
                    )
            for p in [
                path,
                path.with_name("run.json"),
                source,
                Path("assets/figures") / (name + ".png"),
            ]:
                bindings[str(p)] = sha256(p)
    assert len(rows) == 120
    views = target_views()
    view_path = Path("assets/explorer/target_views.json")
    write_json(view_path, views)
    views = json.loads(view_path.read_text())
    bindings[str(view_path)] = sha256(view_path)
    bindings["src/sampling_lab/targets.py"] = sha256("src/sampling_lab/targets.py")
    evidence = dict(rows=rows, targets=views)
    template = Path("scripts/templates/explorer.html")
    script = Path("scripts/templates/explorer.js")
    output = Path("docs/explorer.html")
    output.write_text(
        template.read_text()
        .replace("__EVIDENCE__", json.dumps(evidence, allow_nan=False).replace("<", "\\u003c"))
        .replace("__SCRIPT__", script.read_text())
    )
    for p in [template, script, Path("scripts") / Path(__file__).name]:
        bindings[str(p)] = sha256(p)
    write_json(
        "docs/explorer_manifest.json",
        dict(
            status="complete",
            html_sha256=sha256(output),
            presets=len(rows),
            groups=8,
            bindings=bindings,
            scope="All recorded sampler and energy protocols are selectable. Four unthinned original-space chain trajectories, full extent, replay retained draws only; changes to controls do not refit or invent an intermediate policy. Live linear Gaussian/HMC mechanics are separate analytic examples.",
        ),
    )
    print("Built", len(rows), "measured presets; eight concepts")


if __name__ == "__main__":
    main()
