"""SHAP explainability helpers for the trained tree models.

Computes SHAP values on a sample of rows and saves the standard summary (beeswarm)
and bar plots, plus returns the ranked mean-absolute SHAP importance so it can be
embedded in model cards and logged to MLflow.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend - no display required.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def compute_shap(model, X: pd.DataFrame, *, max_samples: int = 2000, random_state: int = 42):
    """Compute SHAP values for a tree model on up to ``max_samples`` rows.

    Returns ``(shap_values, X_sample)``.
    """
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=random_state)
    else:
        X_sample = X
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    return shap_values, X_sample


def shap_importance(shap_values: np.ndarray, X_sample: pd.DataFrame) -> pd.Series:
    """Rank features by mean absolute SHAP value (descending)."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.Series(mean_abs, index=X_sample.columns)
        .sort_values(ascending=False)
    )


def save_shap_plots(
    model,
    X: pd.DataFrame,
    out_dir: Path,
    *,
    prefix: str = "model",
    max_samples: int = 2000,
    random_state: int = 42,
) -> dict[str, Path]:
    """Compute SHAP values and save summary + bar plots as PNGs.

    Returns a dict with paths to the written ``summary`` and ``bar`` plots and the
    ranked importance Series under ``importance``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shap_values, X_sample = compute_shap(
        model, X, max_samples=max_samples, random_state=random_state
    )

    summary_path = out_dir / f"{prefix}_shap_summary.png"
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()

    bar_path = out_dir / f"{prefix}_shap_bar.png"
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(bar_path, dpi=120, bbox_inches="tight")
    plt.close()

    importance = shap_importance(shap_values, X_sample)
    logger.info("Saved SHAP plots to %s", out_dir)
    return {"summary": summary_path, "bar": bar_path, "importance": importance}
