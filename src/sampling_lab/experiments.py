"""Four-chain benchmark evidence with declared budgets and no test-based tuning."""

import json
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score

from .artifacts import environment, sha256, source_identity, write_json
from .diagnostics import summarize
from .reference import run_nuts
from .samplers import run_chain
from .targets import load_target


def evaluate(config, output, source_root=Path("assets/source_data")):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    name = f"{config['target']}_{config['method']}_{config.get('profile', 'dispersed')}_{config['step_size']:g}"
    folder = output / name
    source = source_identity()
    if (folder / "metrics.json").exists():
        old = json.loads((folder / "run.json").read_text())
        if old["config"] != config or old["source_sha256"] != source:
            raise ValueError("Existing evidence belongs to different source/protocol")
        assert sha256(source_root / (name + ".npz")) == old["numeric_sha256"]
        print("Verified", name, flush=True)
        return
    if (folder / "run.json").exists():
        raise ValueError("Incomplete run retained; use a new directory")
    write_json(
        folder / "run.json",
        dict(status="running", config=config, source_sha256=source, environment=environment()),
    )
    target = load_target(config["target"])
    starts = [
        target.start(i, config["seed"], config.get("profile", "dispersed"))
        for i in range(config["chains"])
    ]
    if config["method"] == "nuts":
        results = run_nuts(target, starts, config["draws"], config["warmup"], config["seed"])
    else:
        results = [
            run_chain(
                target,
                config["method"],
                start,
                config["draws"],
                config["warmup"],
                config["step_size"],
                config["seed"] + i * 10007,
                config.get("leapfrog_steps", 12),
                config.get("preconditioned", False),
            )
            for i, start in enumerate(starts)
        ]
    samples = np.asarray([c["samples"] for c in results])
    observables = target.observable(samples)
    reference = (
        None
        if target.name == "logistic"
        else target.observable(target.reference(100000, np.random.default_rng(800000)))
    )
    diagnostics = summarize(observables, results, reference)
    metrics = dict(
        target=target.name,
        method=config["method"],
        config=config,
        diagnostics=diagnostics,
        acceptance_per_chain=[
            float(c["accepted"].mean()) if c["accepted"] is not None else c["acceptance_rate"]
            for c in results
        ],
        final_step_size=[float(c["step_size"][-1]) for c in results],
        swap_acceptance=[
            c["swap_accepts"] / c["swap_attempts"] if c["swap_attempts"] else None for c in results
        ],
        scope="Fixed presets declared before fitting. Four independent chains; warmup excluded, repeated states retained. Approximate ULA/SGLD/SGHMC bias is not removed by convergence diagnostics. No thinning. Comparisons have unequal gradient/replica work; both work and wall time reported.",
    )
    arrays = dict(
        samples=samples,
        observables=observables,
        starts=np.asarray(starts),
        divergent=np.asarray([c["divergent"] for c in results]),
        step_size=np.asarray([c["step_size"] for c in results]),
    )
    if results[0]["energy"] is not None:
        arrays.update(
            energy=np.asarray([c["energy"] for c in results]),
            accepted=np.asarray([c["accepted"] for c in results]),
            warmup=np.asarray([c["warmup"] for c in results]),
            preconditioner=np.asarray([c["final_preconditioner"] for c in results]),
        )
    if target.name in ["logistic", "regression"]:
        arrays.update(target.data)
        flat = samples.reshape(-1, target.dim)
        predictive = {}
        for split in ["validation", "test"]:
            X, y = target.data["X_" + split], target.data["y_" + split]
            logits = X @ flat.T
            if target.name == "logistic":
                probability = expit(logits).mean(1)
                predictive[split] = dict(
                    log_loss=log_loss(y, probability, labels=[0, 1]),
                    brier_score=brier_score_loss(y, probability),
                    roc_auc=roc_auc_score(y, probability),
                )
                arrays["probability_" + split] = probability
            else:
                mean = logits.mean(1)
                variance = logits.var(1) + target.data["noise_sigma"] ** 2
                lo, hi = mean - 1.96 * np.sqrt(variance), mean + 1.96 * np.sqrt(variance)
                predictive[split] = dict(
                    rmse_standardized=np.sqrt(mean_squared_error(y, mean)),
                    predictive_interval_95_coverage=np.mean((y >= lo) & (y <= hi)),
                    noise_sigma=target.data["noise_sigma"],
                )
                arrays.update({f"mean_{split}": mean, f"lower_{split}": lo, f"upper_{split}": hi})
        metrics["predictive"] = predictive
        metrics["data_scope"] = (
            "Built-in sklearn Breast Cancer Wisconsin Diagnostic (30 features) or Diabetes (10 features). Seeded 70/15/15 splits; scaling, MAP and all posterior likelihood use training only; noise sigma .75 for regression fixed in advance. No test tuning; probabilities are posterior Monte Carlo averages, intervals use Gaussian predictive moment approximation."
        )
    source_root.mkdir(parents=True, exist_ok=True)
    path = source_root / (name + ".npz")
    np.savez_compressed(path, **arrays)
    write_json(folder / "metrics.json", metrics)
    write_json(
        folder / "run.json",
        dict(
            status="complete",
            environment=environment(),
            config=config,
            source_sha256=source,
            numeric_source=str(path),
            numeric_sha256=sha256(path),
        ),
    )
    print(
        name,
        "Rhat",
        diagnostics["rank_rhat"][:2],
        "ESS",
        diagnostics["bulk_ess"][:2],
        "pass",
        diagnostics["convergence_flag"],
        flush=True,
    )
