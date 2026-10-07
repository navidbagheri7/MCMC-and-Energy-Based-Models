"""Independent gradient, reversibility, transition and invariant-law oracles."""

import arviz as az
import numpy as np
import pytest
import torch

from sampling_lab.diagnostics import split_rank_rhat_oracle
from sampling_lab.ebm import Energy, export, langevin, load, log_partition
from sampling_lab.samplers import leapfrog, mala_log_ratio, run_chain
from sampling_lab.targets import Target, load_target


@pytest.mark.parametrize(
    "name",
    [
        "gaussian",
        "perturbed",
        "ring",
        "banana",
        "funnel",
        "funnel_noncentered",
        "regression",
        "logistic",
    ],
)
def test_numpy_targets_against_autograd_and_finite_differences(name):
    target = load_target(name)
    x = target.mean.copy() if target.dim > 2 else np.array([0.4, -0.3])
    x += np.arange(target.dim) * 0.001
    tensor = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    value = target.torch_log_prob(tensor)
    value.backward()
    np.testing.assert_allclose(target.log_prob(x), value.detach().numpy(), rtol=1e-12, atol=1e-10)
    gradient = target.gradient(x)
    np.testing.assert_allclose(gradient, tensor.grad.numpy(), rtol=1e-10, atol=1e-9)
    h = 1e-5
    directions = np.eye(target.dim)
    differences = np.array(
        [(target.log_prob(x + h * d) - target.log_prob(x - h * d)) / (2 * h) for d in directions]
    )
    np.testing.assert_allclose(gradient, differences, rtol=2e-6, atol=1e-7)


def test_leapfrog_is_reversible_and_second_order_energy_error():
    target = load_target("gaussian")
    P = target.cov
    x = np.array([0.2, -0.8])
    p = np.array([0.1, -0.2])
    y, q, valid = leapfrog(target, x, p, 0.05, 20, P)
    z, r, valid2 = leapfrog(target, y, q, 0.05, 20, P)
    np.testing.assert_allclose(z, x, atol=1e-13)
    np.testing.assert_allclose(r, p, atol=1e-13)
    assert valid and valid2
    initial = -target.log_prob(x) + 0.5 * p @ P @ p
    errors = []
    for h, n in [(0.1, 10), (0.05, 20), (0.025, 40)]:
        y, q, _ = leapfrog(target, x, p, h, n, P)
        errors.append(abs(-target.log_prob(y) + 0.5 * q @ P @ q - initial))
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8


def test_mala_asymmetric_proposal_correction_matches_log_detailed_balance():
    target = load_target("banana")
    P = np.array([[2, 0.3], [0.3, 1.0]])
    x, y = np.array([0.3, -0.2]), np.array([0.9, 0.5])
    eta = 0.08
    inverse = np.linalg.inv(P)

    def logq(a, b):
        d = b - a - eta * P @ target.gradient(a)
        return -0.25 / eta * d @ inverse @ d

    ratio = mala_log_ratio(target, x, y, eta, P)
    np.testing.assert_allclose(
        ratio, target.log_prob(y) + logq(y, x) - target.log_prob(x) - logq(x, y)
    )
    assert abs(ratio - (target.log_prob(y) - target.log_prob(x))) > 0.01
    np.testing.assert_allclose(mala_log_ratio(target, y, x, eta, P), -ratio)


@pytest.mark.parametrize(
    "method", ["rwm", "adaptive_rwm", "gibbs", "slice", "mala", "hmc", "adaptive_hmc"]
)
def test_corrected_kernels_recover_analytic_gaussian_moments(method):
    target = Target("gaussian", 2, np.zeros(2), np.array([[1, 0.7], [0.7, 1]]))
    result = run_chain(
        target,
        method,
        np.array([4.0, -4.0]),
        12000,
        1000,
        0.3 if method not in ["mala", "hmc", "adaptive_hmc"] else 0.1,
        123,
        preconditioned=True,
    )
    values = result["samples"]
    np.testing.assert_allclose(values.mean(0), target.mean, atol=0.12)
    np.testing.assert_allclose(np.cov(values.T), target.cov, atol=0.15)
    assert np.unique(result["step_size"]).size == 1


def test_ula_has_the_predicted_discretization_variance():
    target = Target("gaussian", 2, np.zeros(2), np.eye(2))
    result = run_chain(target, "ula", np.zeros(2), 100000, 1000, 0.5, 73)
    variance = result["samples"].var(0)
    np.testing.assert_allclose(variance, 1 / (1 - 0.5 / 2), atol=0.025)
    assert min(variance) > 1.25


def test_rank_and_folded_rhat_matches_independent_definition():
    rng = np.random.default_rng(31)
    values = rng.normal(size=(4, 1000))
    values[0] *= 2
    expected = split_rank_rhat_oracle(values)
    actual = float(az.rhat(values, method="rank"))
    assert actual == pytest.approx(expected, abs=1e-12)
    assert actual > 1.01


def test_logistic_minibatch_gradient_is_unbiased_in_expectation():
    target = load_target("logistic")
    x = target.mean + 0.2
    rng = np.random.default_rng(44)
    gradient = np.mean([target.gradient(x, rng, 64) for _ in range(10000)], axis=0)
    np.testing.assert_allclose(gradient, target.gradient(x), atol=1.1)
    idx = set(target.data["train_indices"])
    assert idx.isdisjoint(target.data["validation_indices"]) and idx.isdisjoint(
        target.data["test_indices"]
    )


def test_conjugate_regression_posterior_satisfies_normal_equations():
    target = load_target("regression")
    X, y = target.data["X_train"], target.data["y_train"]
    precision = X.T @ X / 0.75**2 + np.eye(target.dim) / 4
    np.testing.assert_allclose(precision @ target.cov, np.eye(target.dim), atol=1e-11)
    np.testing.assert_allclose(precision @ target.mean, X.T @ y / 0.75**2, atol=1e-10)


def test_ebm_gaussian_base_partition_oracle_and_safe_export(tmp_path):
    model = Energy()
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    value = float(log_partition(model, 321, 18))
    assert value == pytest.approx(np.log(2 * np.pi * 9), abs=1e-6)
    path = tmp_path / "energy.npz"
    export(model, path)
    x = torch.randn(10, 2)
    torch.testing.assert_close(model(x), load(path)(x), atol=0, rtol=0)
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        load(path)


def test_negative_chains_are_detached_and_do_not_modify_parameter_gradients():
    model = Energy()
    x = torch.randn(20, 2)
    generated = langevin(model, x, 5, 0.01, torch.Generator().manual_seed(9))
    assert not generated.requires_grad and all(p.grad is None for p in model.parameters())
    loss = model(x).mean() - model(generated).mean()
    loss.backward()
    assert sum(p.grad.abs().sum() for p in model.parameters()) > 0


def test_portable_posterior_predicts_actual_heldout_features():
    from pathlib import Path

    from sklearn.datasets import load_breast_cancer, load_diabetes

    from sampling_lab.posterior import predict

    root = Path(__file__).resolve().parents[1]
    if not (root / "assets/source_data").exists():
        pytest.skip("Reference evidence is carried by the source archive")
    for family, loader in [("logistic", load_breast_cancer), ("regression", load_diabetes)]:
        path = root / "assets/source_data" / f"{family}_nuts_dispersed_0.1.npz"
        with np.load(path, allow_pickle=False) as data:
            raw = loader().data[data["test_indices"]]
            result = predict(path, raw, family)
            if family == "logistic":
                np.testing.assert_allclose(
                    result["probability"], data["probability_test"], atol=1e-12, rtol=1e-12
                )
            else:
                np.testing.assert_allclose(
                    result["mean"],
                    data["mean_test"] * float(data["y_std"]) + float(data["y_mean"]),
                    atol=1e-12,
                    rtol=1e-12,
                )
            with pytest.raises(ValueError, match="Finite"):
                predict(path, np.full_like(raw, np.nan), family)
            with pytest.raises(ValueError, match="features"):
                predict(path, raw[:, :-1], family)


def test_tempering_improves_mode_exploration_in_recorded_reference():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    path = root / "assets/source_data/ring_parallel_tempering_dispersed_0.6.npz"
    if not path.exists():
        pytest.skip("Reference evidence is carried by the source archive")
    target = load_target("ring")
    with np.load(path, allow_pickle=False) as data:
        samples = data["samples"].reshape(-1, 2)
        labels = np.argmin(((samples[:, None] - target.centers) ** 2).sum(-1), axis=1)
        counts = np.bincount(labels, minlength=8)
        assert (counts > 100).all()
