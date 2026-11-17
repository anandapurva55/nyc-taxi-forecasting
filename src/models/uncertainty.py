"""Confidence intervals for error metrics on time-ordered data.

Errors within the same hour are correlated (busy hours are busy everywhere), so
resampling individual rows would make intervals too narrow. A block bootstrap
resamples whole blocks, for example whole hours, with replacement.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray], float]


def _block_indices(blocks: np.ndarray) -> list[np.ndarray]:
    _, inverse = np.unique(blocks, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    boundaries = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, boundaries)


def block_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    blocks: np.ndarray,
    metric: MetricFn,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Return the ``(1 - alpha)`` percentile interval of ``metric`` over block resamples."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    groups = _block_indices(np.asarray(blocks))
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in picked])
        stats[i] = metric(y_true[idx], y_pred[idx])
    low, high = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def paired_difference_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    blocks: np.ndarray,
    metric: MetricFn,
    *,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Return ``(observed, low, high)`` for ``metric(a) - metric(b)`` on the same resamples.

    An interval that contains zero means the two models cannot be told apart.
    """
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    groups = _block_indices(np.asarray(blocks))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        picked = rng.integers(0, len(groups), size=len(groups))
        idx = np.concatenate([groups[j] for j in picked])
        diffs[i] = metric(y_true[idx], pred_a[idx]) - metric(y_true[idx], pred_b[idx])
    observed = metric(y_true, pred_a) - metric(y_true, pred_b)
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(observed), float(low), float(high)
