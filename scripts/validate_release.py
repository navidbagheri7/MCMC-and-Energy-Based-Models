"""Recompute public chain diagnostics, predictions, energy likelihoods and bindings."""

import json
import subprocess
import sys
from pathlib import Path

import arviz as az
import numpy as np
import torch
from scipy.special import expit
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error, roc_auc_score

from sampling_lab.artifacts import sha256, write_json
from sampling_lab.ebm import dataset, load, log_partition
from sampling_lab.targets import load_target


def main():
    torch.set_num_threads(1)
    checked = []
    targets = {}
    references = {}
    commands = []
    cfg = json.loads(Path("configs/sampling.json").read_text())["experiments"]
    expected = {f"{c['target']}_{c['method']}_{c['profile']}_{c['step_size']:g}": c for c in cfg}
    files = list(Path("results/reference/sampling").glob("*/run.json"))
    assert len(files) == len(expected) == 84
    for path in sorted(files):
        r = json.loads(path.read_text())
        m = json.loads(path.with_name("metrics.json").read_text())
        name = path.parent.name
        assert (
            r["status"] == "complete"
            and r["config"] == expected[name]
            and m["config"] == r["config"]
        )
        for f, digest in r["source_sha256"].items():
            assert sha256(Path("src/sampling_lab") / f) == digest
        source = Path(r["numeric_source"])
        assert sha256(source) == r["numeric_sha256"]
        with np.load(source, allow_pickle=False) as data:
            values = dict(data)
        for array in values.values():
            if array.dtype.kind in "fci":
                assert np.isfinite(array).all(), source
        x = values["samples"]
        observed = values["observables"]
        c = r["config"]
        assert x.shape == (4, 2000, load_target(c["target"]).dim)
        target = targets.setdefault(c["target"], load_target(c["target"]))
        # Recomputed float64 transforms can differ by a few CPU rounding units.
        np.testing.assert_allclose(observed, target.observable(x), rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(
            values["starts"], [target.start(i, c["seed"], c["profile"]) for i in range(4)]
        )
        truth = references.setdefault(
            c["target"],
            None
            if c["target"] == "logistic"
            else target.observable(target.reference(100000, np.random.default_rng(800000))),
        )
        posterior = az.from_dict(posterior={"position": observed})
        d = m["diagnostics"]
        for key, func, method in [
            ("rank_rhat", az.rhat, "rank"),
            ("bulk_ess", az.ess, "bulk"),
            ("tail_ess", az.ess, "tail"),
            ("mean_mcse", az.mcse, "mean"),
        ]:
            actual = np.asarray(func(posterior, method=method)["position"])
            record = np.asarray([np.nan if v is None else v for v in d[key]])
            np.testing.assert_allclose(actual, record, rtol=1e-9, atol=1e-10, equal_nan=True)
        np.testing.assert_allclose(observed.mean((0, 1)), d["mean"], rtol=1e-12, atol=1e-12)
        if truth is not None:
            np.testing.assert_allclose(
                d["standardized_mean_error"],
                (observed.mean((0, 1)) - truth.mean(0)) / truth.std(0, ddof=1),
                rtol=1e-10,
                atol=1e-11,
            )
        if c["method"] not in ["sgld", "sghmc"]:
            assert all(np.unique(s).size == 1 for s in values["step_size"])
        assert int(values["divergent"].sum()) == d["divergences"]
        if c["target"] in ["regression", "logistic"]:
            np.testing.assert_array_equal(values["train_indices"], target.data["train_indices"])
            assert set(values["train_indices"]).isdisjoint(values["test_indices"])
            for split in ["validation", "test"]:
                X, y = values["X_" + split], values["y_" + split]
                pred = X @ x.reshape(-1, target.dim).T
                if c["target"] == "logistic":
                    probability = expit(pred).mean(1)
                    np.testing.assert_allclose(
                        probability, values["probability_" + split], rtol=1e-12, atol=1e-12
                    )
                    scores = dict(
                        log_loss=log_loss(y, probability, labels=[0, 1]),
                        brier_score=brier_score_loss(y, probability),
                        roc_auc=roc_auc_score(y, probability),
                    )
                else:
                    mean = pred.mean(1)
                    variance = pred.var(1) + 0.75**2
                    lo, hi = mean - 1.96 * np.sqrt(variance), mean + 1.96 * np.sqrt(variance)
                    np.testing.assert_allclose(
                        mean, values["mean_" + split], rtol=1e-12, atol=1e-12
                    )
                    scores = dict(
                        rmse_standardized=np.sqrt(mean_squared_error(y, mean)),
                        predictive_interval_95_coverage=np.mean((y >= lo) & (y <= hi)),
                        noise_sigma=0.75,
                    )
                for key, value in scores.items():
                    np.testing.assert_allclose(
                        value, m["predictive"][split][key], rtol=1e-12, atol=1e-12
                    )
        checked.append(
            dict(protocol=name, numeric_sha256=r["numeric_sha256"], diagnostics_recomputed=True)
        )
    for path in sorted(Path("results/reference/ebm").glob("*/run.json")):
        r = json.loads(path.read_text())
        m = json.loads(path.with_name("metrics.json").read_text())
        c = r["config"]
        name = path.parent.name
        assert r["status"] == "complete"
        for f, digest in r["source_sha256"].items():
            assert sha256(Path("src/sampling_lab") / f) == digest
        source = Path(r["numeric_source"])
        assert sha256(source) == r["numeric_sha256"]
        model = load(path.parent / "energy.npz")
        assert sha256(path.parent / "energy.npz") == m["model_sha256"]
        with np.load(source, allow_pickle=False) as data:
            train, val, test = dataset(c["task"])
            for key, value in [("train", train), ("validation", val), ("test", test)]:
                np.testing.assert_array_equal(data[key], value)
            for arr in data.values():
                assert np.isfinite(arr).all()
            with torch.no_grad():
                partition = float(log_partition(model, 401, 18))
                nll = float((model(torch.from_numpy(test)) + partition).mean())
            np.testing.assert_allclose(partition, m["log_partition"], rtol=1e-9, atol=1e-9)
            np.testing.assert_allclose(nll, m["test_nll"], rtol=1e-9, atol=1e-9)
            index = int(np.argmin(data["history_nll"]))
            assert int(data["history_steps"][index]) == m["selected_step"]
            np.testing.assert_allclose(
                data["history_nll"][index], m["validation_nll"], rtol=1e-9, atol=1e-9
            )
        checked.append(
            dict(
                protocol=name,
                numeric_sha256=r["numeric_sha256"],
                test_likelihood_recomputed=True,
                model_sha256=m["model_sha256"],
            )
        )
    assert len(checked) == 120
    for family, loader in [("logistic", load_breast_cancer), ("regression", load_diabetes)]:
        name = f"{family}_nuts_dispersed_0.1"
        data = np.load(Path("assets/source_data") / (name + ".npz"), allow_pickle=False)
        raw = loader()
        idx = data["test_indices"]
        folder = Path(".local/inference_commands")
        folder.mkdir(parents=True, exist_ok=True)
        csv = folder / (family + ".csv")
        out = folder / (family + ".json")
        np.savetxt(
            csv, raw.data[idx], delimiter=",", header=",".join(data["feature_names"]), comments=""
        )
        subprocess.run(
            [
                sys.executable,
                "scripts/predict_posterior.py",
                "--protocol",
                name,
                "--input",
                str(csv),
                "--output",
                str(out),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        result = json.loads(out.read_text())
        actual = np.asarray(result["probability" if family == "logistic" else "mean"])
        truth = (
            data["probability_test"]
            if family == "logistic"
            else data["mean_test"] * float(data["y_std"]) + float(data["y_mean"])
        )
        np.testing.assert_allclose(actual, truth, rtol=1e-12, atol=1e-12)
        commands.append(dict(protocol=name, predictions=len(idx)))
    notebook = json.loads(Path("docs/notebook_execution.json").read_text())
    assert notebook["status"] == "passed" and len(notebook["notebooks"]) == 10
    import nbformat

    for record in notebook["notebooks"]:
        assert sha256(record["notebook"]) == record["sha256"]
        nb = nbformat.read(record["notebook"], as_version=4)
        code = [c for c in nb.cells if c.cell_type == "code"]
        assert len(code) == record["executed_code_cells"] and all(
            c.execution_count is not None for c in code
        )
        assert all(o.output_type != "error" for c in code for o in c.outputs)
        assert record["visual_outputs"] > 0
    browser = json.loads(Path("docs/explorer_browser_validation.json").read_text())
    manifest = json.loads(Path("docs/explorer_manifest.json").read_text())
    assert (
        browser["status"] == "passed"
        and sha256("docs/explorer.html") == browser["html_sha256"] == manifest["html_sha256"]
    )
    for f, digest in manifest["bindings"].items():
        assert sha256(f) == digest
    for config in Path("configs").glob("*.json"):
        assert (
            config.read_bytes()
            == (Path("src/sampling_lab/default_configs") / config.name).read_bytes()
        )
    write_json(
        "docs/validation.json",
        dict(
            status="passed",
            protocols=checked,
            posterior_commands=commands,
            reference_protocols=120,
            independent_chains=336,
            energy_fits=36,
            notebooks=10,
            notebook_code_cells=sum(r["executed_code_cells"] for r in notebook["notebooks"]),
            browser_presets=120,
            live_browser_checks=28,
            exact_replay_checks=252,
            scope="Public numeric/source/model integrity, independent ArviZ diagnostics and target/reference moments, all real-data scores, energy test likelihoods and two fresh full-heldout posterior inference commands. This is not proof that every chain converged or every approximation is unbiased.",
        ),
    )
    print(
        "Validated all 120 public protocols, predictive/likelihood scores and executable evidence"
    )


if __name__ == "__main__":
    main()
