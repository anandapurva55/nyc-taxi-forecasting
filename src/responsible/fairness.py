"""Subgroup performance parity (geographic & temporal service equity).

Given predictions on a held-out set, we slice the standard regression metrics by
subgroups that matter for equitable taxi service - pickup **borough**,
**time-of-day**, **weekend/holiday** and **airport** zones - and quantify the
gap with a **disparity ratio** of a scale-aware error: WAPE (total absolute
error / total actual). Raw RMSE grows with volume, so busy Manhattan or evening
groups would always look "worst" on RMSE alone; WAPE compares groups on
relative accuracy. The RMSE ratio is still reported for context. A ratio close
to 1.0 means the model serves all subgroups comparably.

WAPE divides by the group's total actual, so it is unstable for groups with
almost no volume (a borough averaging well under one pickup per zone-hour).
Groups whose mean actual is below ``MIN_MEAN_ACTUAL`` are still listed, but are
left out of the ratio.

A dimension with fewer than two comparable groups in the evaluation data (for
example, a test window with no weekend days) cannot be audited and is reported
as a coverage gap rather than as parity.

This module is numpy/pandas-only so it stays fast and unit-testable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.models.common import mae, mape, rmse


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted absolute percentage error: sum |error| / sum |actual|."""
    denom = float(np.sum(np.abs(y_true)))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(np.asarray(y_true) - np.asarray(y_pred))) / denom)

DEFAULT_DISPARITY_THRESHOLD = 1.5
MIN_SUBGROUP_SIZE = 30
# Groups averaging less than this per row (pickups per zone-hour, or dollars)
# are too small for a stable WAPE and are excluded from the disparity ratio.
MIN_MEAN_ACTUAL = 1.0

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
    wape: float = float("nan")
    mean_actual: float = float("nan")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FairnessReport:
    """Sliced metrics for one dimension plus its disparity summary."""

    dimension: str
    subgroups: list[SubgroupMetric] = field(default_factory=list)
    threshold: float = DEFAULT_DISPARITY_THRESHOLD
    min_mean_actual: float = MIN_MEAN_ACTUAL

    @property
    def compared(self) -> list[SubgroupMetric]:
        """Groups with enough volume for a stable WAPE."""
        return [s for s in self.subgroups if s.mean_actual >= self.min_mean_actual]

    @property
    def low_volume_groups(self) -> list[str]:
        return [s.group for s in self.subgroups if s.mean_actual < self.min_mean_actual]

    @property
    def coverage_gap(self) -> bool:
        """True when fewer than two comparable groups were present."""
        return len(self.compared) < 2

    @property
    def disparity(self) -> float:
        """Worst / best WAPE across comparable subgroups (scale-aware)."""
        return disparity_ratio([s.wape for s in self.compared])

    @property
    def rmse_ratio(self) -> float:
        return disparity_ratio([s.rmse for s in self.compared])

    @property
    def flagged(self) -> bool:
        return not self.coverage_gap and self.disparity > self.threshold

    @property
    def worst_group(self) -> str | None:
        if not self.compared:
            return None
        return max(self.compared, key=lambda s: s.wape).group

    @property
    def best_group(self) -> str | None:
        if not self.compared:
            return None
        return min(self.compared, key=lambda s: s.wape).group

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension,
            "disparity_metric": "wape",
            "disparity_ratio": self.disparity,
            "rmse_ratio": self.rmse_ratio,
            "flagged": self.flagged,
            "coverage_gap": self.coverage_gap,
            "worst_group": self.worst_group,
            "best_group": self.best_group,
            "threshold": self.threshold,
            "min_mean_actual": self.min_mean_actual,
            "low_volume_groups": self.low_volume_groups,
            "subgroups": [s.to_dict() for s in self.subgroups],
        }


def disparity_ratio(values: list[float]) -> float:
    """Worst/best ratio of an error metric across subgroups (1.0 = perfect parity)."""
    vals = [r for r in values if r is not None and np.isfinite(r) and r > 0]
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
                wape=wape(yt, yp),
                mean_actual=float(np.mean(yt)),
            )
        )

    subgroups.sort(key=lambda s: s.wape, reverse=True)
    return FairnessReport(dimension=dimension, subgroups=subgroups, threshold=threshold)
