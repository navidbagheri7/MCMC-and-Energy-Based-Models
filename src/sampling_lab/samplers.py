"""Owned NumPy kernels; adaptation occurs exclusively during warm-up.

Langevin uses x' = x + eta P grad(log pi) + sqrt(2 eta) chol(P) z.
Hamiltonian momentum covariance is inv(P), with kinetic p.T P p / 2.
Rejected/nonfinite proposals retain the previous state; repetitions are saved.
"""

import time

import numpy as np
from scipy.special import logsumexp

METHODS = (
    "rwm",
    "adaptive_rwm",
    "gibbs",
    "slice",
    "ula",
    "mala",
    "hmc",
    "adaptive_hmc",
    "parallel_tempering",
    "sgld",
    "sghmc",
    "nuts",
)


def leapfrog(target, x, p, step_size, steps, preconditioner):
    x, p = np.array(x, copy=True), np.array(p, copy=True)
    p += step_size / 2 * target.gradient(x)
    for i in range(steps):
        x += step_size * (preconditioner @ p)
        with np.errstate(over="ignore", invalid="ignore"):
            gradient = target.gradient(x)
        if not np.isfinite(x).all() or not np.isfinite(gradient).all():
            return x, p, False
        p += step_size * (0.5 if i == steps - 1 else 1) * gradient
    return x, -p, True


def mala_log_ratio(target, old, new, eta, preconditioner):
    inverse = np.linalg.inv(preconditioner)
    forward = new - old - eta * preconditioner @ target.gradient(old)
    reverse = old - new - eta * preconditioner @ target.gradient(new)
    return (
        target.log_prob(new)
        - target.log_prob(old)
        + (forward @ inverse @ forward - reverse @ inverse @ reverse) / (4 * eta)
    )


def gaussian_gibbs(target, x, rng):
    precision = target.precision
    for k in range(len(x)):
        conditional = (
            target.mean[k]
            - (precision[k] @ (x - target.mean) - precision[k, k] * (x[k] - target.mean[k]))
            / precision[k, k]
        )
        x[k] = conditional + rng.normal() / np.sqrt(precision[k, k])
    return x


def coordinate_slice(target, x, rng, width=1.0, max_steps=128):
    for k in rng.permutation(len(x)):
        level = target.log_prob(x) + np.log(rng.random())
        left = x[k] - width * rng.random()
        right = left + width
        j = rng.integers(max_steps)
        remaining = max_steps - 1 - j
        z = x.copy()
        while j > 0:
            z[k] = left
            if target.log_prob(z) <= level:
                break
            left -= width
            j -= 1
        while remaining > 0:
            z[k] = right
            if target.log_prob(z) <= level:
                break
            right += width
            remaining -= 1
        original = x[k]
        for _ in range(10000):
            proposal = rng.uniform(left, right)
            z[k] = proposal
            if target.log_prob(z) >= level:
                x = z.copy()
                break
            if proposal < original:
                left = proposal
            else:
                right = proposal
        else:
            raise RuntimeError("Slice shrinkage failed; no transition silently skipped")
    return x


def run_chain(
    target, method, start, draws, warmup, step_size, seed, leapfrog_steps=12, preconditioned=False
):
    if method not in METHODS or method == "nuts":
        raise ValueError("Use reference.run_nuts for the maintained NUTS implementation")
    if min(draws, step_size, leapfrog_steps) < 1e-12 or warmup < 0:
        raise ValueError("Invalid sampler settings")
    if method == "gibbs" and target.name not in ["gaussian", "regression", "ring"]:
        raise ValueError("Only targets with exact implemented conditionals support Gibbs")
    if method in ["sgld", "sghmc"] and target.name != "logistic":
        raise ValueError("Stochastic likelihood gradients are defined for logistic data only")
    rng = np.random.default_rng(seed)
    x = np.asarray(start, dtype=float).copy()
    P = np.array(target.cov if preconditioned else np.eye(target.dim), copy=True)
    cholesky = np.linalg.cholesky(P)
    momentum_chol = np.linalg.cholesky(np.linalg.inv(P))
    eta = float(step_size)
    log_eta = np.log(eta)
    mu = np.log(10 * eta)
    hbar = 0.0
    smooth = log_eta
    samples = []
    accept = []
    energies = []
    divergences = []
    steps_record = []
    work = 0
    history = []
    nonfinite = 0
    tempered = np.tile(x, (8, 1))
    betas = np.geomspace(1, 0.03, 8)
    swap_attempts = 0
    swap_accepts = 0
    velocity = np.zeros(target.dim)
    started = time.perf_counter()
    for iteration in range(draws + warmup):
        proposed = x.copy()
        log_ratio = 0.0
        divergent = False
        energy = float(-target.log_prob(x))
        iteration_work = 0
        if method in ["rwm", "adaptive_rwm"]:
            proposed = x + eta * cholesky @ rng.normal(size=target.dim)
            log_ratio = target.log_prob(proposed) - target.log_prob(x)
        elif method in ["ula", "mala", "sgld"]:
            current_eta = eta if method != "sgld" else eta / (1 + iteration / 1000) ** 0.55
            gradient = target.gradient(x, rng, 64 if method == "sgld" else None)
            iteration_work += 1
            proposed = (
                x
                + current_eta * (P @ gradient)
                + np.sqrt(2 * current_eta) * cholesky @ rng.normal(size=target.dim)
            )
            if method == "mala":
                log_ratio = mala_log_ratio(target, x, proposed, eta, P)
                iteration_work += 2
        elif method in ["hmc", "adaptive_hmc"]:
            p = momentum_chol @ rng.normal(size=target.dim)
            energy = float(-target.log_prob(x) + 0.5 * p @ P @ p)
            # Random trajectory length helps avoid resonant fixed integration times.
            nsteps = int(rng.integers(max(1, leapfrog_steps // 2), leapfrog_steps + 1))
            proposed, new_p, valid = leapfrog(target, x, p, eta, nsteps, P)
            iteration_work += nsteps + 1
            with np.errstate(over="ignore", invalid="ignore"):
                delta = -target.log_prob(proposed) + 0.5 * new_p @ P @ new_p - energy
            divergent = not valid or not np.isfinite(delta) or abs(delta) > 1000
            log_ratio = -delta if valid else -np.inf
        elif method == "gibbs":
            if target.name == "ring":
                logs = -0.5 * ((x - target.centers) ** 2).sum(1) / 0.35**2
                label = rng.choice(8, p=np.exp(logs - logsumexp(logs)))
                proposed = target.centers[label] + 0.35 * rng.normal(size=2)
            else:
                proposed = gaussian_gibbs(target, x.copy(), rng)
        elif method == "slice":
            proposed = coordinate_slice(target, x.copy(), rng, eta)
        elif method == "parallel_tempering":
            for replica, beta in enumerate(betas):
                candidate = tempered[replica] + eta / np.sqrt(beta) * cholesky @ rng.normal(
                    size=target.dim
                )
                ratio = beta * (target.log_prob(candidate) - target.log_prob(tempered[replica]))
                if np.log(rng.random()) < min(0, ratio):
                    tempered[replica] = candidate
            for k in range(iteration % 2, len(betas) - 1, 2):
                ratio = (betas[k] - betas[k + 1]) * (
                    target.log_prob(tempered[k + 1]) - target.log_prob(tempered[k])
                )
                swap_attempts += 1
                if np.log(rng.random()) < min(0, ratio):
                    tempered[[k, k + 1]] = tempered[[k + 1, k]]
                    swap_accepts += 1
            proposed = tempered[0].copy()
        elif method == "sghmc":
            # Momentum Euler integrator, friction .05, B=0 approximation. Gradient
            # noise is not estimated/cancelled: this is explicitly approximate SGHMC.
            current_eta = eta / (1 + iteration / 1000) ** 0.55
            velocity = (
                0.95 * velocity
                + current_eta * target.gradient(x, rng, 64)
                + np.sqrt(2 * 0.05 * current_eta) * rng.normal(size=target.dim)
            )
            proposed = x + velocity
            iteration_work += 1
        finite = (
            np.isfinite(proposed).all()
            and np.isfinite(log_ratio)
            and np.isfinite(target.log_prob(proposed))
        )
        probability = float(np.exp(min(0, log_ratio))) if finite else 0.0
        accepted = finite and np.log(rng.random()) < min(0, log_ratio)
        if accepted:
            x = proposed
        if not finite:
            nonfinite += 1
            if method == "sghmc":
                velocity *= 0
        if iteration < warmup:
            history.append(x.copy())
            if method == "adaptive_rwm":
                log_eta += (iteration + 10) ** (-0.6) * (probability - 0.234)
                eta = np.exp(np.clip(log_eta, -12, 2))
            if method == "adaptive_hmc":
                t = iteration + 1
                hbar = (1 - 1 / (t + 10)) * hbar + (0.8 - probability) / (t + 10)
                log_eta = mu - np.sqrt(t) * hbar / 0.05
                eta = np.exp(np.clip(log_eta, -12, 1))
                smooth = t ** (-0.75) * log_eta + (1 - t ** (-0.75)) * smooth
                # One warmup-only diagonal geometry window, followed by renewed
                # dual averaging. The reported sampling kernel is fixed thereafter.
                if t == warmup // 2 and len(history) >= 100:
                    variance = np.var(np.asarray(history)[-min(t, 500) :], axis=0)
                    P = np.diag(np.clip(variance, 1e-3, 100))
                    cholesky = np.linalg.cholesky(P)
                    momentum_chol = np.linalg.cholesky(np.linalg.inv(P))
                    mu = np.log(10 * eta)
                    hbar = 0.0
                    smooth = np.log(eta)
                if t == warmup:
                    eta = np.exp(np.clip(smooth, -12, 1))
        else:
            samples.append(x.copy())
            accept.append(accepted)
            energies.append(energy)
            divergences.append(divergent)
            steps_record.append(eta if method not in ["sgld", "sghmc"] else current_eta)
        work += iteration_work
    return dict(
        samples=np.asarray(samples),
        accepted=np.asarray(accept),
        energy=np.asarray(energies),
        divergent=np.asarray(divergences),
        step_size=np.asarray(steps_record),
        final_preconditioner=P,
        gradient_evaluations=work,
        seconds=time.perf_counter() - started,
        nonfinite_proposals=nonfinite,
        swap_accepts=swap_accepts,
        swap_attempts=swap_attempts,
        warmup=np.asarray(history),
    )
