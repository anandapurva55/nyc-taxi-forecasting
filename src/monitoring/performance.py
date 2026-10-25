"""Prediction performance monitoring vs. delayed actuals.

Once ground-truth values arrive (e.g. realised pickup counts or metered fares),
we compare them against the captured predictions to track live accuracy and
detect degradation relative to the model's offline (training-time) baseline.
"""

from __future__ import annotations

import numpy as np

from src.models.common import regression_metrics

# A metric is "degraded" when it worsens by more than this fraction of baseline.
DEFAULT_TOLERANCE = 0.20

# Metrics where a *higher* value is worse (error metrics). r2 is handled inversely.
_ERROR_METRICS = ("rmse", "mae", "mape")


def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the standard regression metric suite for live predictions."""
    return regression_metrics(y_true, y_pred)


def performance_degradation(
    current: dict[str, float],
    baseline: dict[str, float],
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, dict]:
    """Compare current metrics against a baseline and flag degradation.

    For error metrics (rmse/mae/mape) a positive relative change means *worse*;
    for r2 a drop means worse. Returns, per metric, the relative change and a
    ``degraded`` flag. Missing/zero baselines are skipped safely.
    """
    out: dict[str, dict] = {}
    for name, base in baseline.items():
        if name not in current:
            continue
        cur = current[name]
        if base is None or not np.isfinite(base) or base == 0:
            out[name] = {"current": cur, "baseline": base, "rel_change": None, "degraded": False}
            continue

        if name in _ERROR_METRICS:
            rel = (cur - base) / abs(base)  # positive => worse
            degraded = rel > tolerance
        else:  # r2 and similar: higher is better
            rel = (cur - base) / abs(base)  # negative => worse
            degraded = rel < -tolerance

        out[name] = {
            "current": float(cur),
            "baseline": float(base),
            "rel_change": float(rel),
            "degraded": bool(degraded),
        }
    return out


def any_degraded(degradation: dict[str, dict]) -> bool:
    """True if any monitored metric breached its tolerance."""
    return any(v.get("degraded") for v in degradation.values())
