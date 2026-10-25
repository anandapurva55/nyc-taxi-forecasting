"""Shared calendar/temporal feature helpers.

Both the demand and fare models rely on cyclical calendar signals (hour of day,
day of week, month) plus a US public-holiday flag. Centralising these here keeps
the two feature pipelines consistent.
"""

from __future__ import annotations

import functools

import holidays
import numpy as np
import pandas as pd


@functools.lru_cache(maxsize=8)
def _us_holidays(year: int) -> frozenset:
    """Return the set of US public-holiday dates for a year (cached)."""
    return frozenset(holidays.UnitedStates(years=year).keys())


def is_holiday(timestamps: pd.Series) -> pd.Series:
    """Boolean Series flagging US public holidays for each timestamp."""
    ts = pd.to_datetime(timestamps)
    years = ts.dt.year.unique().tolist()
    holiday_dates: set = set()
    for y in years:
        holiday_dates |= set(_us_holidays(int(y)))
    return ts.dt.date.isin(holiday_dates)


def add_calendar_features(
    df: pd.DataFrame, time_col: str, prefix: str = ""
) -> pd.DataFrame:
    """Append calendar features derived from ``time_col``.

    Adds hour, day-of-week, month, weekend flag, holiday flag and the sine/cosine
    cyclical encodings of hour and day-of-week.
    """
    out = df.copy()
    ts = pd.to_datetime(out[time_col])

    hour = ts.dt.hour
    dow = ts.dt.dayofweek
    month = ts.dt.month

    out[f"{prefix}hour"] = hour
    out[f"{prefix}day_of_week"] = dow
    out[f"{prefix}month"] = month
    out[f"{prefix}is_weekend"] = (dow >= 5).astype("int64")
    out[f"{prefix}is_holiday"] = is_holiday(ts).astype("int64")

    out[f"{prefix}hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out[f"{prefix}hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out[f"{prefix}dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out[f"{prefix}dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return out
