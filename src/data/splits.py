"""Chronological train/validation/test splitting helpers.

Time-series problems require splits that respect temporal ordering - random
splits leak future information. These helpers sort by a time column and slice
contiguous train/val/test segments using the configured fractions.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Config, get_config

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    time_col: str,
    config: Config | None = None,
    *,
    train_frac: float | None = None,
    val_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ``df`` chronologically into (train, val, test).

    Rows are sorted by ``time_col`` and divided into contiguous segments. The
    test fraction is implied by ``1 - train_frac - val_frac``.
    """
    config = config or get_config()
    split_cfg = config.split
    train_frac = train_frac if train_frac is not None else split_cfg["train_frac"]
    val_frac = val_frac if val_frac is not None else split_cfg["val_frac"]

    if train_frac + val_frac > 1.0:
        raise ValueError("train_frac + val_frac must be <= 1.0")

    ordered = df.sort_values(time_col).reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train = ordered.iloc[:train_end].reset_index(drop=True)
    val = ordered.iloc[train_end:val_end].reset_index(drop=True)
    test = ordered.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Split on '%s': train=%d val=%d test=%d", time_col, len(train), len(val), len(test)
    )
    return train, val, test
