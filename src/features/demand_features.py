"""Demand-forecasting feature engineering.

Transforms the dense hourly per-zone pickup series into a supervised learning
table with calendar features, lagged pickup counts and rolling-window
statistics. Lags/rolling windows are computed *within* each zone and shifted so
that only past information is used to predict the current hour's demand.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, get_config
from src.features.calendar import add_calendar_features

logger = logging.getLogger(__name__)

TARGET = "pickup_count"
TIME_COL = "pickup_hour"
GROUP_COL = "PULocationID"


def _add_lags(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(GROUP_COL)[TARGET]
    for lag in lags:
        out[f"lag_{lag}"] = grouped.shift(lag)
    return out


def _add_rolling(df: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    # Shift by 1 so the rolling window only sees past observations.
    shifted = out.groupby(GROUP_COL)[TARGET].shift(1)
    out["_shifted"] = shifted
    for window in windows:
        roll = (
            out.groupby(GROUP_COL)["_shifted"]
            .rolling(window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[f"rolling_mean_{window}"] = roll
        roll_std = (
            out.groupby(GROUP_COL)["_shifted"]
            .rolling(window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )
        out[f"rolling_std_{window}"] = roll_std.fillna(0.0)
    return out.drop(columns="_shifted")


def build_demand_features(
    demand: pd.DataFrame,
    config: Config | None = None,
    *,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build the supervised demand-forecasting feature table.

    Parameters
    ----------
    demand:
        Dense hourly demand series from :func:`aggregate_hourly_demand`.
    dropna:
        If ``True``, drop rows that lack the longest lag (the warm-up period).
    """
    config = config or get_config()
    feat_cfg = config.features
    lags = list(feat_cfg.get("demand_lags", [1, 24, 168]))
    windows = list(feat_cfg.get("demand_rolling_windows", [3, 24]))

    df = demand.sort_values([GROUP_COL, TIME_COL]).reset_index(drop=True)
    df = add_calendar_features(df, TIME_COL)
    df = _add_lags(df, lags)
    df = _add_rolling(df, windows)

    if dropna:
        max_lag = max(lags)
        lag_cols = [f"lag_{lag}" for lag in lags]
        before = len(df)
        df = df.dropna(subset=lag_cols).reset_index(drop=True)
        logger.info(
            "Demand features: dropped %d warm-up rows (max lag=%d)",
            before - len(df),
            max_lag,
        )

    logger.info("Demand feature table: %d rows x %d cols", *df.shape)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model input columns (everything except target/keys)."""
    exclude = {TARGET, TIME_COL, GROUP_COL}
    return [c for c in df.columns if c not in exclude]
