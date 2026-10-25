"""SHAP-based explainability for the tree models (XGBoost / LightGBM).

Provides a thin wrapper around SHAP's ``TreeExplainer`` (exact and fast for
gradient-boosted trees) to produce:

- **global** feature importance (mean absolute SHAP value per feature), and
- **per-instance** explanations (signed contribution per feature for a single
  prediction), which satisfy the additivity property
  ``base_value + sum(contributions) == model_output``.

The optional matplotlib summary-plot helper guards its import so the core
explanation API stays lightweight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap


def build_explainer(model) -> shap.TreeExplainer:
    """Create a SHAP TreeExplainer for a fitted tree model."""
    return shap.TreeExplainer(model)


def _shap_values(explainer: shap.TreeExplainer, X: pd.DataFrame) -> np.ndarray:
    values = explainer.shap_values(X)
    # Some SHAP versions return a list (per output); regression -> single array.
    if isinstance(values, list):
        values = values[0]
    return np.asarray(values)


def global_importance(model, X: pd.DataFrame, explainer=None) -> pd.DataFrame:
    """Return a DataFrame of features ranked by mean absolute SHAP value."""
    explainer = explainer or build_explainer(model)
    values = _shap_values(explainer, X)
    importance = np.abs(values).mean(axis=0)
    return (
        pd.DataFrame({"feature": list(X.columns), "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@dataclass
class InstanceExplanation:
    """Signed SHAP contribution per feature for one prediction."""

    base_value: float
    prediction: float
    contributions: dict[str, float]

    def top(self, k: int = 5) -> list[tuple[str, float]]:
        """Return the ``k`` features with the largest absolute contribution."""
        ranked = sorted(self.contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)
        return ranked[:k]

    def to_dict(self) -> dict:
        return {
            "base_value": self.base_value,
            "prediction": self.prediction,
            "contributions": self.contributions,
        }


def explain_instance(model, x_row: pd.DataFrame, explainer=None) -> InstanceExplanation:
    """Explain a single-row prediction as base value + per-feature contributions."""
    if len(x_row) != 1:
        raise ValueError("explain_instance expects exactly one row.")
    explainer = explainer or build_explainer(model)
    values = _shap_values(explainer, x_row)[0]

    expected = explainer.expected_value
    if isinstance(expected, list | np.ndarray):
        expected = float(np.asarray(expected).ravel()[0])
    base_value = float(expected)

    contributions = {
        feature: float(v) for feature, v in zip(x_row.columns, values, strict=False)
    }
    prediction = base_value + float(np.sum(values))
    return InstanceExplanation(
        base_value=base_value, prediction=prediction, contributions=contributions
    )


def save_summary_plot(model, X: pd.DataFrame, path, explainer=None) -> None:  # pragma: no cover
    """Render and save a SHAP summary (beeswarm) plot; guards matplotlib import."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for summary plots.") from exc

    explainer = explainer or build_explainer(model)
    values = _shap_values(explainer, X)
    shap.summary_plot(values, X, show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
