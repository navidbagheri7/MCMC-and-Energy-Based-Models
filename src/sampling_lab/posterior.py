"""Portable posterior predictions from safe arrays; no private training weights."""

import numpy as np
from scipy.special import expit


def predict(path, raw, target):
    if target not in ["logistic", "regression"]:
        raise ValueError("Unknown posterior family")
    with np.load(path, allow_pickle=False) as arrays:
        posterior = np.array(arrays["samples"]).reshape(-1, arrays["samples"].shape[-1])
        mean, scale = arrays["scaler_mean"], arrays["scaler_scale"]
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(mean) or not np.isfinite(raw).all():
            raise ValueError("Finite raw features in the published feature order are required")
        if (
            posterior.shape[1] != len(mean) + 1
            or not np.isfinite(posterior).all()
            or (scale <= 0).any()
        ):
            raise ValueError("Invalid posterior/schema")
        X = np.column_stack([np.ones(len(raw)), (raw - mean) / scale])
        values = X @ posterior.T
        if target == "logistic":
            probabilities = expit(values)
            return dict(
                probability=probabilities.mean(1),
                probability_credible_lower=np.quantile(probabilities, 0.025, axis=1),
                probability_credible_upper=np.quantile(probabilities, 0.975, axis=1),
                scope="Posterior uncertainty for probability of class 1; this is not a prediction interval for a binary outcome.",
            )
        mu = values.mean(1)
        sigma = np.sqrt(values.var(1) + float(arrays["noise_sigma"]) ** 2)
        ymean, ystd = float(arrays["y_mean"]), float(arrays["y_std"])
        return dict(
            mean=mu * ystd + ymean,
            lower=(mu - 1.96 * sigma) * ystd + ymean,
            upper=(mu + 1.96 * sigma) * ystd + ymean,
            scope="95% Gaussian predictive moment approximation including fixed observation noise; raw Diabetes target units.",
        )
