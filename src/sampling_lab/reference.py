"""Maintained Pyro NUTS reference; independent sequential chains and warmup only."""

import time

import numpy as np
import pyro
import torch
from pyro.infer.mcmc import MCMC, NUTS


def run_nuts(target, starts, draws, warmup, seed):
    torch.set_num_threads(1)
    results = []
    for chain, start in enumerate(starts):
        pyro.set_rng_seed(seed + chain * 10007)
        calls = [0]

        def potential(values, calls=calls):
            calls[0] += 1
            return -target.torch_log_prob(values["position"])

        kernel = NUTS(
            potential_fn=potential,
            adapt_step_size=True,
            adapt_mass_matrix=True,
            full_mass=target.dim <= 32,
            target_accept_prob=0.85,
            max_tree_depth=8,
        )
        run = MCMC(
            kernel,
            num_samples=draws,
            warmup_steps=warmup,
            num_chains=1,
            initial_params={"position": torch.as_tensor(start, dtype=torch.float64)},
            disable_progbar=True,
        )
        started = time.perf_counter()
        run.run()
        diagnostic = run.diagnostics()
        divergences = np.zeros(draws, dtype=bool)
        for indices in diagnostic["divergences"].values():
            divergences[np.asarray(indices, dtype=int)] = True
        results.append(
            dict(
                samples=run.get_samples()["position"].numpy(),
                divergent=divergences,
                seconds=time.perf_counter() - started,
                gradient_evaluations=calls[0],
                acceptance_rate=float(next(iter(diagnostic["acceptance rate"].values()))),
                final_step_size=kernel.step_size,
                energy=None,
                accepted=None,
                step_size=np.full(draws, kernel.step_size),
                final_preconditioner=None,
                warmup=None,
                nonfinite_proposals=0,
                swap_attempts=0,
                swap_accepts=0,
            )
        )
    return results
