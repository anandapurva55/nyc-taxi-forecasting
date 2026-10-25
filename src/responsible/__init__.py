"""Responsible-AI tooling.

Because the NYC taxi trip data contains no demographic or otherwise protected
attributes, Responsible AI here focuses on **geographic & temporal service
equity** and **transparency** rather than classic demographic fairness:

- ``fairness``   - sliced performance metrics + disparity ratios across
  boroughs, time-of-day, weekend/holiday and airport zones (numpy/pandas only).
- ``explain``    - SHAP (TreeExplainer) global feature importance and
  per-prediction contributions for the tree models.
- ``model_card`` - render a markdown model card from metadata, metrics, fairness
  and top features.

``fairness`` and ``model_card`` are dependency-light and unit-testable; the SHAP
summary-plot helper guards its matplotlib import.
"""

from __future__ import annotations

from src.responsible.fairness import (
    FairnessReport,
    SubgroupMetric,
    add_borough,
    disparity_ratio,
    subgroup_metrics,
    time_of_day_bucket,
)
from src.responsible.model_card import render_model_card

__all__ = [
    "FairnessReport",
    "SubgroupMetric",
    "add_borough",
    "disparity_ratio",
    "subgroup_metrics",
    "time_of_day_bucket",
    "render_model_card",
]
