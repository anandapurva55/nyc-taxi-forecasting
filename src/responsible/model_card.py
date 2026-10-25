"""Render a Markdown model card from model metadata, metrics and fairness.

Model cards (Mitchell et al., 2019) document a model's intended use, training
data, evaluation - including **subgroup** performance - and its ethical
considerations and limitations, so downstream users can judge whether the model
is appropriate for their context.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

from src.responsible.fairness import FairnessReport


def _metrics_table(metrics: dict[str, float]) -> str:
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in metrics.items())
    return "| Metric | Value |\n|--------|-------|\n" + rows


def _fairness_section(reports: Sequence[FairnessReport]) -> str:
    if not reports:
        return "_No subgroup analysis available._"
    parts: list[str] = []
    for rep in reports:
        flag = "⚠️ **flagged**" if rep.flagged else "ok"
        parts.append(
            f"### By {rep.dimension}\n\n"
            f"Disparity ratio (worst/best RMSE): **{rep.disparity:.2f}** "
            f"(threshold {rep.threshold}) - {flag}. "
            f"Worst: `{rep.worst_group}`, best: `{rep.best_group}`.\n\n"
            "| Group | n | RMSE | MAE | MAPE |\n|-------|---|------|-----|------|\n"
            + "\n".join(
                f"| {s.group} | {s.n} | {s.rmse:.3f} | {s.mae:.3f} | {s.mape:.2f} |"
                for s in rep.subgroups
            )
        )
    return "\n\n".join(parts)


def _importance_section(top_features: Sequence[tuple[str, float]]) -> str:
    if not top_features:
        return "_No feature-importance data available._"
    rows = "\n".join(f"| {name} | {value:.4f} |" for name, value in top_features)
    return "| Feature | Mean |SHAP| |\n|---------|-----------|\n" + rows


def render_model_card(
    *,
    name: str,
    problem: str,
    algorithm: str,
    target: str,
    metrics: dict[str, float],
    fairness: Sequence[FairnessReport] = (),
    top_features: Sequence[tuple[str, float]] = (),
    intended_use: str = "",
    limitations: Sequence[str] = (),
    ethical_considerations: Sequence[str] = (),
    version: str = "",
    generated_at: str | None = None,
) -> str:
    """Return a Markdown model card as a string."""
    generated_at = generated_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    limitations = list(limitations) or [
        "Trained on a single month of NYC Yellow Taxi data; may not generalise to "
        "other periods, boroughs served sparsely, or green/for-hire vehicles.",
        "Predictions degrade under distribution shift (see drift monitoring).",
    ]
    ethical_considerations = list(ethical_considerations) or [
        "No demographic or protected attributes are used or available; fairness is "
        "assessed as geographic/temporal service equity, not demographic parity.",
        "Large borough-level disparities could translate into unequal service "
        "quality across neighbourhoods and should be monitored.",
    ]

    lines = [
        f"# Model Card - {name}",
        "",
        f"_Generated: {generated_at}_",
        "",
        "## Model details",
        "",
        f"- **Problem:** {problem}",
        f"- **Algorithm:** {algorithm}",
        f"- **Target:** `{target}`",
        f"- **Registry version:** {version or 'n/a'}",
        "",
        "## Intended use",
        "",
        intended_use
        or (
            f"Operational forecasting of {target} to support fleet positioning and "
            "rider fare estimates. Not intended for individual pricing decisions or "
            "any use with legal/financial consequences for individuals."
        ),
        "",
        "## Evaluation - overall (test split)",
        "",
        _metrics_table(metrics),
        "",
        "## Evaluation - subgroup fairness",
        "",
        _fairness_section(fairness),
        "",
        "## Top features (SHAP)",
        "",
        _importance_section(top_features),
        "",
        "## Limitations",
        "",
        "\n".join(f"- {item}" for item in limitations),
        "",
        "## Ethical considerations",
        "",
        "\n".join(f"- {item}" for item in ethical_considerations),
        "",
    ]
    return "\n".join(lines)
