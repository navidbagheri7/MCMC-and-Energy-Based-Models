"""Offline sampling/energy explorer, with real chain states and live mechanics."""

import base64
import json
from pathlib import Path

import numpy as np

from sampling_lab.artifacts import sha256, write_json


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
    evidence = dict(rows=rows)
    template = Path("scripts/templates/explorer.html")
    script = Path("scripts/templates/explorer.js")
    output = Path("docs/explorer.html")
    output.write_text(
        template.read_text()
        .replace("__EVIDENCE__", json.dumps(evidence, allow_nan=False).replace("<", "\\u003c"))
        .replace("__SCRIPT__", script.read_text())
    )
    for p in [template, script, Path(__file__)]:
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
