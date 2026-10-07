"""Rebuild figures and comparative tables from immutable numerical records."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import write_json


def records(family):
    return [
        (p, json.loads(p.read_text()))
        for p in sorted((Path("results/reference") / family).glob("*/metrics.json"))
    ]


def figure(fig, name):
    Path("assets/figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    for extension in ["png", "pdf"]:
        fig.savefig(f"assets/figures/{name}.{extension}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    sampler_rows = []
    for path, m in records("sampling"):
        d = m["diagnostics"]
        cfg = m["config"]
        name = path.parent.name

        def finite(xs):
            return [x for x in xs if x is not None]

        sampler_rows.append(
            dict(
                target=m["target"],
                method=m["method"],
                profile=cfg["profile"],
                step_size=cfg["step_size"],
                max_rhat=max(finite(d["rank_rhat"]), default=np.nan),
                min_bulk_ess=min(finite(d["bulk_ess"]), default=np.nan),
                min_tail_ess=min(finite(d["tail_ess"]), default=np.nan),
                divergences=d["divergences"],
                seconds=d["total_sampling_and_warmup_seconds"],
                gradient_evaluations=d["gradient_evaluations_includes_warmup"],
                convergence_flag=d["convergence_flag"],
                protocol=name,
                metrics=str(path),
            )
        )
        with np.load(Path("assets/source_data") / (name + ".npz"), allow_pickle=False) as data:
            x = data["observables"]
            fig, axes = plt.subplots(2, 2, figsize=(10, 7))
            for chain, values in enumerate(x):
                axes[0, 0].plot(
                    np.arange(len(values)), values[:, 0], lw=0.5, alpha=0.6, label=f"Chain {chain}"
                )
                axes[1, 0].hist(
                    values[:, 0], bins=45, density=True, histtype="step", label=f"Chain {chain}"
                )
                axes[0, 1].scatter(values[::4, 0], values[::4, 1], s=2, alpha=0.25)
            axes[0, 0].set(
                xlabel="Retained draw; warmup excluded",
                ylabel="First observable",
                title="Trace, every retained state",
            )
            axes[0, 0].legend(fontsize=8)
            axes[1, 0].set(
                xlabel="First observable",
                ylabel="Marginal density",
                title="Each chain separately; no thinning",
            )
            axes[0, 1].set(
                xlabel="Observable 1",
                ylabel="Observable 2",
                title="Every fourth point shown; full extent",
            )
            axes[1, 1].axis("off")
            axes[1, 1].text(
                0.02,
                0.98,
                f"{m['method']} · {cfg['profile']}\n{m['target']} · preset {cfg['step_size']:g}\nRank Rhat max {sampler_rows[-1]['max_rhat']:.3f}\nBulk ESS min {sampler_rows[-1]['min_bulk_ess']:.1f}\nTail ESS min {sampler_rows[-1]['min_tail_ess']:.1f}\nDivergences {d['divergences']}\nDiagnostic flag {d['convergence_flag']}\nApproximate-kernel bias needs a separate check",
                va="top",
                fontsize=11,
            )
            figure(fig, name)
    table = pd.DataFrame(sampler_rows)
    assert len(table) == 84
    table.to_csv("docs/sampling_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in ["funnel", "funnel_noncentered"]:
        rows = table[(table.target == name) & table.method.isin(["hmc", "adaptive_hmc", "nuts"])]
        for ax, column in zip(axes, ["max_rhat", "min_bulk_ess"], strict=False):
            ax.plot(rows.method, rows[column], "o-", label=name)
    axes[0].axhline(1.01, color="black", ls="--", lw=1)
    axes[0].set(ylabel="Maximum rank Rhat", title="Centered versus non-centered geometry")
    axes[1].set(ylabel="Minimum bulk ESS", title="Same 4×2000 retained draws")
    axes[1].legend()
    figure(fig, "funnel_geometry")
    ebm_rows = []
    for path, m in records("ebm"):
        c = m["config"]
        ebm_rows.append(
            dict(
                task=c["task"],
                method=c["method"],
                seed=c["seed"],
                selected=m["selected_step"],
                validation_nll=m["validation_nll"],
                test_nll=m["test_nll"],
                partition_refinement_change=m["partition_max_change"],
                seconds=m["fitting_seconds"],
                negative_gradient_evaluations=m["negative_gradient_evaluations"],
                metrics=str(path),
            )
        )
        name = path.parent.name
        with np.load(Path("assets/source_data") / (name + ".npz"), allow_pickle=False) as data:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            grid = data["grid"]
            size = int(np.sqrt(len(grid)))
            density = data["density"].reshape(size, size)
            axes[0].imshow(
                density, origin="lower", extent=(-10, 10, -10, 10), cmap="magma", aspect="equal"
            )
            axes[0].set(xlim=(-5, 5), ylim=(-5, 5), title="Normalized fitted energy density")
            generated = data["generated"]
            axes[1].scatter(*data["test"].T, s=2, alpha=0.3, label="Independent test data")
            axes[1].scatter(*generated.T, s=2, alpha=0.3, label="Long-run ULA negatives")
            axes[1].set(aspect="equal", title="Held-out data and approximate negatives")
            axes[1].legend(fontsize=7)
            axes[2].plot(data["history_steps"], data["history_nll"])
            axes[2].axvline(m["selected_step"], ls="--", color="gray")
            axes[2].set(
                xlabel="Optimizer update",
                ylabel="Validation NLL",
                title="Validation-only selection",
            )
            figure(fig, name)
    energy = pd.DataFrame(ebm_rows)
    assert len(energy) == 36
    energy.to_csv("docs/energy_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, task in zip(axes, ["mixture", "moons"], strict=False):
        rows = energy[energy.task == task]
        stats = rows.groupby("method").test_nll.agg(["mean", "std"])
        ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], fmt="o", capsize=4)
        ax.set(title=task, ylabel="Test NLL: mean ± fit-seed SD")
        ax.tick_params(axis="x", rotation=25)
    figure(fig, "energy_objectives")
    lines = [
        "# Measured results",
        "",
        f"{len(table)} sampler protocols with four independent chains each; {len(energy)} neural-energy fits with three fitting seeds per task/method. All preset results, including failures, appear in the CSV tables.",
        "",
        f"{int(table.convergence_flag.sum())}/{len(table)} protocols meet the stated rank-Rhat/ESS/divergence flag. This flag is a diagnostic screen, not a proof of convergence or absence of finite-step bias.",
        "",
        "| Task | Negative phase | Fits | Mean test NLL | Fitting SD |",
        "|---|---|---:|---:|---:|",
    ]
    for (task, method), g in energy.groupby(["task", "method"]):
        lines.append(
            f"| {task} | {method} | {len(g)} | {g.test_nll.mean():.3f} | {g.test_nll.std():.3f} |"
        )
    lines += [
        "",
        "NLL uses a bounded-residual Gaussian-base model and explicit quadrature refinement. No image-generation or exact high-dimensional normalization claim is made. CD-1 is visibly weaker on the mixture; persistent/replay methods do not uniformly outperform CD-10.",
        "",
        "The centered funnel exposes divergences and poor mixing. Non-centering is an invertible change of sampling coordinates; diagnostics use the same original-space observables.",
        "",
        "[Sampling CSV](sampling_comparison.csv) · [Energy CSV](energy_comparison.csv) · [Interactive explorer](explorer.html)",
    ]
    Path("docs/results.md").write_text("\n".join(lines) + "\n")
    write_json("docs/experiment_index.json", dict(sampling=sampler_rows, energy=ebm_rows))
    print("Rebuilt", len(table) + len(energy), "figure pairs and two aggregate comparisons")


if __name__ == "__main__":
    main()
