"""Subgroup performance parity (geographic & temporal service equity).

Given predictions on a held-out set, we slice the standard regression metrics by
subgroups that matter for equitable taxi service - pickup **borough**,
**time-of-day**, **weekend/holiday** and **airport** zones - and quantify the
gap with a **disparity ratio** (worst-subgroup RMSE / best-subgroup RMSE). A
ratio close to 1.0 means the model serves all subgroups comparably; a large
ratio flags that some neighbourhoods or times are served far less accurately.

This module is numpy/pandas-only so it stays fast and unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.models.common import mae, mape, rmse

DEFAULT_DISPARITY_THRESHOLD = 1.5
MIN_SUBGROUP_SIZE = 30

# Hour-of-day buckets used for the temporal fairness slice.
_TOD_BUCKETS = [
    (0, 6, "overnight"),
    (6, 12, "morning"),
    (12, 18, "afternoon"),
    (18, 24, "evening"),
]


def time_of_day_bucket(hour: int) -> str:
    """Map an hour (0-23) to a coarse part-of-day label."""
    h = int(hour) % 24
    for start, end, label in _TOD_BUCKETS:
        if start <= h < end:
            return label
    return "overnight"


def add_borough(
    df: pd.DataFrame, zone_lookup: pd.DataFrame, location_col: str = "PULocationID"
) -> pd.DataFrame:
    """Attach ``borough`` and ``service_zone`` columns via the taxi-zone lookup."""
    lookup = zone_lookup.rename(
        columns={"LocationID": location_col, "Borough": "borough"}
    )[[location_col, "borough", "service_zone"]]
    out = df.merge(lookup, on=location_col, how="left")
    out["borough"] = out["borough"].fillna("Unknown")
    out["service_zone"] = out["service_zone"].fillna("Unknown")
    return out


@dataclass
class SubgroupMetric:
    """Metrics for a single subgroup within a fairness dimension."""

    dimension: str
    group: str
    n: int
    rmse: float
    mae: float
    mape: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FairnessReport:
    """Sliced metrics for one dimension plus its disparity summary."""

    dimension: str
    subgroups: list[SubgroupMetric] = field(default_factory=list)
    threshold: float = DEFAULT_DISPARITY_THRESHOLD

    @property
    def disparity(self) -> float:
        return disparity_ratio([s.rmse for s in self.subgroups])

    @property
    def flagged(self) -> bool:
        return self.disparity > self.threshold

    @property
    def worst_group(self) -> str | None:
        if not self.subgroups:
            return None
        return max(self.subgroups, key=lambda s: s.rmse).group

    @property
    def best_group(self) -> str | None:
        if not self.subgroups:
            return None
        return min(self.subgroups, key=lambda s: s.rmse).group

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "disparity_ratio": self.disparity,
            "flagged": self.flagged,
            "worst_group": self.worst_group,
            "best_group": self.best_group,
            "threshold": self.threshold,
            "subgroups": [s.to_dict() for s in self.subgroups],
        }


def disparity_ratio(rmses: list[float]) -> float:
    """Worst/best RMSE ratio across subgroups (1.0 = perfect parity)."""
    vals = [r for r in rmses if r is not None and np.isfinite(r) and r > 0]
    if len(vals) < 2:
        return 1.0
    return float(max(vals) / min(vals))


def subgroup_metrics(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dimension: str,
    *,
    threshold: float = DEFAULT_DISPARITY_THRESHOLD,
    min_size: int = MIN_SUBGROUP_SIZE,
) -> FairnessReport:
    """Compute per-subgroup RMSE/MAE/MAPE for the given ``dimension`` column.

    ``df[dimension]`` provides the subgroup label per row. Subgroups smaller than
    ``min_size`` are skipped to avoid noisy, unstable metrics.
    """
    if dimension not in df.columns:
        raise ValueError(f"Dimension column not found: {dimension!r}")

    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    work = df.reset_index(drop=True)

    subgroups: list[SubgroupMetric] = []
    for group, idx in work.groupby(dimension).groups.items():
        pos = np.asarray(idx, dtype="int64")
        if pos.size < min_size:
            continue
        yt, yp = y_true[pos], y_pred[pos]
        subgroups.append(
            SubgroupMetric(
                dimension=dimension,
                group=str(group),
                n=int(pos.size),
                rmse=rmse(yt, yp),
                mae=mae(yt, yp),
                mape=mape(yt, yp),
            )
        )

    subgroups.sort(key=lambda s: s.rmse, reverse=True)
    return FairnessReport(dimension=dimension, subgroups=subgroups, threshold=threshold)
