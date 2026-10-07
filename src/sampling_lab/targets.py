"""Analytic, geometrically difficult and train-only Bayesian posterior targets."""

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class Target:
    name: str
    dim: int = 2
    mean: np.ndarray = field(default_factory=lambda: np.array([2.0, -2.0]))
    cov: np.ndarray = field(default_factory=lambda: np.array([[1.0, 0.95], [0.95, 1.0]]))
    data: dict = field(default_factory=dict)

    def __post_init__(self):
        self.precision = np.linalg.inv(self.cov)
        if self.name == "ring":
            theta = np.arange(8) * 2 * np.pi / 8
            self.centers = 3 * np.stack([np.cos(theta), np.sin(theta)], axis=1)

    def log_prob(self, x):
        x = np.asarray(x)
        if self.name in ["gaussian", "perturbed", "regression"]:
            d = x - self.mean
            value = -0.5 * np.einsum("...i,ij,...j->...", d, self.precision, d)
            return (
                value + np.cos(x[..., 0]) + np.sin(x[..., 1]) if self.name == "perturbed" else value
            )
        if self.name == "ring":
            distance = x[..., None, :] - self.centers
            return (
                logsumexp(-0.5 * (distance**2).sum(-1) / 0.35**2, axis=-1)
                - np.log(8)
                - np.log(2 * np.pi * 0.35**2)
            )
        if self.name == "banana":
            u, v = x[..., 0], x[..., 1] - 0.25 * (x[..., 0] ** 2 - 4)
            return -0.5 * (u**2 / 4 + v**2)
        if self.name == "funnel":
            y, z = x[..., 0], x[..., 1]
            with np.errstate(over="ignore", invalid="ignore"):
                return -0.5 * y**2 / 9 - 0.5 * y - 0.5 * z**2 * np.exp(-y)
        if self.name == "funnel_noncentered":
            return -0.5 * x[..., 0] ** 2 / 9 - 0.5 * x[..., 1] ** 2
        if self.name == "logistic":
            logits = np.einsum("ni,...i->...n", self.data["X_train"], x)
            return (self.data["y_train"] * logits - np.logaddexp(0, logits)).sum(-1) - 0.5 * (
                (x / 2.5) ** 2
            ).sum(-1)
        raise ValueError(self.name)

    def gradient(self, x, rng=None, batch_size=None):
        x = np.asarray(x)
        if self.name in ["gaussian", "perturbed", "regression"]:
            result = -(x - self.mean) @ self.precision
            if self.name == "perturbed":
                result += np.array([-np.sin(x[0]), np.cos(x[1])])
            return result
        if self.name == "ring":
            delta = x - self.centers
            logw = -0.5 * (delta**2).sum(1) / 0.35**2
            weights = np.exp(logw - logsumexp(logw))
            return -(weights[:, None] * delta).sum(0) / 0.35**2
        if self.name == "banana":
            u, v = x[0], x[1] - 0.25 * (x[0] ** 2 - 4)
            return np.array([-u / 4 + 0.5 * u * v, -v])
        if self.name == "funnel":
            with np.errstate(over="ignore", invalid="ignore"):
                return np.array(
                    [-x[0] / 9 - 0.5 + 0.5 * x[1] ** 2 * np.exp(-x[0]), -x[1] * np.exp(-x[0])]
                )
        if self.name == "funnel_noncentered":
            return np.array([-x[0] / 9, -x[1]])
        if self.name == "logistic":
            X, y = self.data["X_train"], self.data["y_train"]
            scale = 1.0
            if batch_size:
                if rng is None:
                    raise ValueError("Minibatches need an explicit generator")
                idx = rng.choice(len(X), min(batch_size, len(X)), replace=False)
                scale = len(X) / len(idx)
                X, y = X[idx], y[idx]
            return scale * X.T @ (y - expit(X @ x)) - x / 2.5**2
        raise ValueError(self.name)

    def torch_log_prob(self, x):
        def as_tensor(a):
            return torch.as_tensor(a, dtype=x.dtype, device=x.device)

        if self.name in ["gaussian", "perturbed", "regression"]:
            d = x - as_tensor(self.mean)
            value = -0.5 * d @ as_tensor(self.precision) @ d
            return value + x[0].cos() + x[1].sin() if self.name == "perturbed" else value
        if self.name == "ring":
            return (
                torch.logsumexp(-0.5 * ((x - as_tensor(self.centers)) ** 2).sum(1) / 0.35**2, 0)
                - np.log(8)
                - np.log(2 * np.pi * 0.35**2)
            )
        if self.name == "banana":
            return -0.5 * (x[0] ** 2 / 4 + (x[1] - 0.25 * (x[0] ** 2 - 4)) ** 2)
        if self.name == "funnel":
            return -0.5 * x[0] ** 2 / 9 - 0.5 * x[0] - 0.5 * x[1] ** 2 * torch.exp(-x[0])
        if self.name == "funnel_noncentered":
            return -0.5 * x[0] ** 2 / 9 - 0.5 * x[1] ** 2
        logits = as_tensor(self.data["X_train"]) @ x
        return (
            as_tensor(self.data["y_train"]) * logits
            - torch.logaddexp(torch.zeros_like(logits), logits)
        ).sum() - 0.5 * (x / 2.5).square().sum()

    def observable(self, samples):
        samples = np.array(samples, copy=True)
        if self.name == "funnel_noncentered":
            samples[..., 1] *= np.exp(samples[..., 0] / 2)
        return samples

    def reference(self, size, rng):
        if self.name in ["gaussian", "regression"]:
            return rng.multivariate_normal(self.mean, self.cov, size)
        if self.name == "ring":
            return self.centers[rng.integers(8, size=size)] + 0.35 * rng.normal(size=(size, 2))
        if self.name == "banana":
            z = rng.normal(size=(size, 2))
            z[:, 0] *= 2
            z[:, 1] += 0.25 * (z[:, 0] ** 2 - 4)
            return z
        if self.name in ["funnel", "funnel_noncentered"]:
            z = rng.normal(size=(size, 2))
            z[:, 0] *= 3
            if self.name == "funnel":
                z[:, 1] *= np.exp(z[:, 0] / 2)
            return z
        if self.name == "perturbed":
            # Exact rejection draws under Gaussian envelope: cos+sin <= 2.
            out = []
            total = 0
            while total < size:
                proposal = rng.multivariate_normal(
                    self.mean, self.cov, max(128, 2 * (size - total))
                )
                selected = proposal[
                    np.log(rng.random(len(proposal)))
                    < np.cos(proposal[:, 0]) + np.sin(proposal[:, 1]) - 2
                ]
                out.append(selected)
                total += len(selected)
            return np.concatenate(out)[:size]
        raise ValueError("No exact posterior sampler; use independently diagnosed NUTS reference")

    def start(self, chain, seed, profile="dispersed"):
        rng = np.random.default_rng(seed + 900000 + chain)
        if profile == "warm":
            # Training-only MAP/known center, still independently jittered chains.
            return self.mean + rng.normal(size=self.dim) * 0.1
        if self.name in ["logistic", "regression"]:
            return self.mean + rng.normal(size=self.dim) * 2
        return (
            np.array([[-5.0, -5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, 5.0]])[chain % 4]
            + rng.normal(size=2) * 0.2
        )


def load_target(name):
    if name not in ["logistic", "regression"]:
        return Target(name)
    raw = load_breast_cancer() if name == "logistic" else load_diabetes()
    idx = np.arange(len(raw.target))
    train, other = train_test_split(
        idx, test_size=0.3, random_state=741, stratify=raw.target if name == "logistic" else None
    )
    val, test = train_test_split(
        other,
        test_size=0.5,
        random_state=742,
        stratify=raw.target[other] if name == "logistic" else None,
    )
    scaler = StandardScaler().fit(raw.data[train])
    arrays = {
        f"X_{split}": np.column_stack([np.ones(len(ids)), scaler.transform(raw.data[ids])])
        for split, ids in [("train", train), ("validation", val), ("test", test)]
    }
    if name == "regression":
        ymean, ystd = raw.target[train].mean(), raw.target[train].std()
        arrays.update(
            {
                f"y_{split}": (raw.target[ids] - ymean) / ystd
                for split, ids in [("train", train), ("validation", val), ("test", test)]
            }
        )
        arrays.update(y_mean=ymean, y_std=ystd, noise_sigma=0.75)
        X, y = arrays["X_train"], arrays["y_train"]
        precision = X.T @ X / 0.75**2 + np.eye(X.shape[1]) / 4
        cov = np.linalg.inv(precision)
        mean = cov @ X.T @ y / 0.75**2
    else:
        arrays.update(
            {
                f"y_{split}": raw.target[ids]
                for split, ids in [("train", train), ("validation", val), ("test", test)]
            }
        )
        X, y = arrays["X_train"], arrays["y_train"]

        def objective(w):
            return np.logaddexp(0, X @ w).sum() - y @ (X @ w) + 0.5 * (w / 2.5) @ (w / 2.5)

        def gradient(w):
            return X.T @ (expit(X @ w) - y) + w / 2.5**2

        result = minimize(
            objective, np.zeros(X.shape[1]), jac=gradient, method="BFGS", options={"gtol": 1e-7}
        )
        if np.linalg.norm(gradient(result.x), ord=np.inf) > 1e-5:
            raise RuntimeError("MAP solve did not reach declared gradient tolerance")
        mean = result.x
        p = expit(X @ mean)
        cov = np.linalg.inv(X.T @ (p[:, None] * (1 - p[:, None]) * X) + np.eye(X.shape[1]) / 2.5**2)
    arrays.update(
        train_indices=train,
        validation_indices=val,
        test_indices=test,
        feature_names=raw.feature_names,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
    )
    return Target(name, len(mean), mean, cov, arrays)
