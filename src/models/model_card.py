"""Generate a simple Markdown model card for a trained model.

Model cards summarise a model's intended use, training data window, the winning
algorithm and hyperparameters, held-out test metrics, the top SHAP feature
drivers and known limitations - a training artifact and the foundation for the
Responsible-AI evaluation.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_PROBLEM_DESCRIPTION = {
    "demand": (
        "Predicts the number of taxi pickups for a given zone and hour "
        "(demand forecasting)."
    ),
    "fare": (
        "Predicts the fare amount for a single taxi trip from its distance, "
        "zones, passenger count and calendar context (fare prediction)."
    ),
}

_LIMITATIONS = {
    "demand": [
        "Trained on a single month of NYC TLC Yellow Taxi data; seasonal effects "
        "beyond the training window are not captured.",
        "Lag/rolling features require recent history - cold-start zones/hours are "
        "less reliable.",
        "Only Yellow Taxi pickups are modelled (no green taxi / FHV).",
    ],
    "fare": [
        "Trained on a single month of NYC TLC Yellow Taxi data; tariff changes or "
        "surge conditions outside the window are not represented.",
        "Extreme fares are filtered during cleaning, so very long/expensive trips "
        "may be under-predicted.",
        "Tolls, tips and surcharges are not part of the modelled fare amount.",
    ],
}


def build_model_card(
    problem: str,
    estimator: str,
    params: dict[str, Any],
    test_metrics: dict[str, float],
    val_metrics: dict[str, float],
    top_features: pd.Series,
    *,
    data_year: int | None = None,
    data_month: int | None = None,
    n_train: int | None = None,
    registry_name: str | None = None,
) -> str:
    """Render the model card as a Markdown string."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines: list[str] = []
    lines.append(f"# Model Card - {registry_name or problem}")
    lines.append("")
    lines.append(f"_Generated {now}_")
    lines.append("")
    lines.append("## Overview")
    lines.append(_PROBLEM_DESCRIPTION.get(problem, problem))
    lines.append("")
    lines.append(f"- **Problem:** {problem}")
    lines.append(f"- **Best algorithm:** {estimator}")
    if registry_name:
        lines.append(f"- **Registry name:** `{registry_name}`")
    window = "single-month NYC TLC sample"
    if data_year and data_month:
        window = f"{data_year}-{data_month:02d}"
    lines.append(f"- **Training data window:** {window}")
    if n_train is not None:
        lines.append(f"- **Training rows:** {n_train:,}")
    lines.append("")

    lines.append("## Test-set performance")
    lines.append("")
    lines.append("| Metric | Test | Validation |")
    lines.append("|--------|------|------------|")
    for key in ("rmse", "mae", "mape", "r2"):
        t = test_metrics.get(key, float("nan"))
        v = val_metrics.get(key, float("nan"))
        lines.append(f"| {key.upper()} | {t:.4f} | {v:.4f} |")
    lines.append("")

    lines.append("## Top feature drivers (mean |SHAP|)")
    lines.append("")
    for rank, (feat, val) in enumerate(top_features.head(10).items(), start=1):
        lines.append(f"{rank}. `{feat}` - {val:.4f}")
    lines.append("")

    lines.append("## Hyperparameters")
    lines.append("")
    lines.append("```json")
    import json

    lines.append(json.dumps(params, indent=2, default=str))
    lines.append("```")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    for item in _LIMITATIONS.get(problem, []):
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def save_model_card(card: str, out_dir: Path, *, prefix: str = "model") -> Path:
    """Write the model card markdown to ``out_dir`` and return its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_model_card.md"
    path.write_text(card, encoding="utf-8")
    logger.info("Wrote model card: %s", path)
    return path
