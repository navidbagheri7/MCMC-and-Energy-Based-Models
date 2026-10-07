"""Rank-normalized multichain diagnostics; finite chains never imply convergence."""

import warnings

import arviz as az
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def summarize(samples, chains, reference=None):
    samples = np.asarray(samples)
    if (
        samples.ndim != 3
        or samples.shape[0] < 2
        or samples.shape[1] < 20
        or not np.isfinite(samples).all()
    ):
        raise ValueError("Diagnostics require finite [>=2 chains, >=20 draws, dimensions]")
    posterior = az.from_dict(posterior={"position": samples})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rhat = np.asarray(az.rhat(posterior, method="rank")["position"])
        ess_bulk = np.asarray(az.ess(posterior, method="bulk")["position"])
        ess_tail = np.asarray(az.ess(posterior, method="tail")["position"])
        mcse = np.asarray(az.mcse(posterior, method="mean")["position"])

    def finite(a):
        return [float(x) if np.isfinite(x) else None for x in a]

    seconds = sum(c["seconds"] for c in chains)
    cost = sum(c["gradient_evaluations"] for c in chains)
    diagnostic = dict(
        rank_rhat=finite(rhat),
        bulk_ess=finite(ess_bulk),
        tail_ess=finite(ess_tail),
        mean_mcse=finite(mcse),
        mean=samples.mean((0, 1)).tolist(),
        standard_deviation=samples.reshape(-1, samples.shape[-1]).std(0, ddof=1).tolist(),
        chain_means=samples.mean(1).tolist(),
        draws_per_chain=samples.shape[1],
        chains=samples.shape[0],
        total_sampling_and_warmup_seconds=seconds,
        gradient_evaluations_includes_warmup=cost,
        min_bulk_ess_per_second=float(np.nanmin(ess_bulk) / seconds)
        if np.isfinite(ess_bulk).any()
        else None,
        min_bulk_ess_per_gradient=float(np.nanmin(ess_bulk) / cost)
        if cost and np.isfinite(ess_bulk).any()
        else None,
        divergences=sum(int(c["divergent"].sum()) for c in chains),
        nonfinite_proposals=sum(c["nonfinite_proposals"] for c in chains),
        convergence_flag=bool(
            np.isfinite(rhat).all()
            and (rhat < 1.01).all()
            and (ess_bulk > 400).all()
            and (ess_tail > 400).all()
            and not any(c["divergent"].any() for c in chains)
        ),
    )
    if all(c["energy"] is not None for c in chains):
        energy = np.asarray([c["energy"] for c in chains])
        diagnostic["energy_bfmi"] = [float(v) if np.isfinite(v) else None for v in az.bfmi(energy)]
        diagnostic["bfmi_scope"] = (
            "Meaningful for HMC with refreshed momentum; for other kernels stored potential is a trace diagnostic only"
        )
    else:
        diagnostic["energy_bfmi"] = None
        diagnostic["bfmi_scope"] = (
            "Pyro public interface does not expose the paired momentum energy; no fabricated energy diagnostic"
        )
    if reference is not None:
        reference = np.asarray(reference)
        scale = reference.std(0, ddof=1)
        diagnostic.update(
            reference_mean=reference.mean(0),
            standardized_mean_error=(samples.mean((0, 1)) - reference.mean(0)) / scale,
            covariance_relative_error=float(
                np.linalg.norm(
                    np.cov(samples.reshape(-1, samples.shape[-1]).T) - np.cov(reference.T)
                )
                / np.linalg.norm(np.cov(reference.T))
            ),
        )
    return diagnostic


def split_rank_rhat_oracle(samples):
    """Independent rank/folded split-Rhat for tests against the diagnostic library."""
    samples = np.asarray(samples)
    n = samples.shape[1] // 2
    split = np.concatenate([samples[:, :n], samples[:, -n:]], axis=0)

    def calculate(values):
        ranks = rankdata(values.ravel()).reshape(values.shape)
        normalized = ndtri((ranks - 3 / 8) / (values.size + 1 / 4))
        W = np.mean(np.var(normalized, axis=1, ddof=1))
        B = n * np.var(normalized.mean(1), ddof=1)
        return np.sqrt(((n - 1) / n * W + B / n) / W) if W else np.nan

    return max(calculate(split), calculate(np.abs(split - np.median(split))))
