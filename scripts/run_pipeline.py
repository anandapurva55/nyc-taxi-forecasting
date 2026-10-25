"""End-to-end data pipeline.

Runs ingestion -> cleaning -> demand aggregation -> feature engineering ->
chronological splits, persisting intermediate artifacts to the configured data
directories.

Usage::

    python -m scripts.run_pipeline            # use config defaults
    python -m scripts.run_pipeline --year 2024 --month 1
    python -m scripts.run_pipeline --skip-download   # reuse cached raw data
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config import get_config
from src.data.aggregation import aggregate_hourly_demand
from src.data.ingestion import (
    ingest,
    load_trips,
    load_zone_lookup,
    trip_filename,
)
from src.data.splits import time_based_split
from src.data.validation import clean_trips
from src.features.demand_features import build_demand_features
from src.features.fare_features import build_fare_features

logger = logging.getLogger(__name__)


def _write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote %s (%d rows)", path, len(df))


def run(year: int | None, month: int | None, skip_download: bool) -> dict:
    config = get_config()
    config.ensure_dirs()

    year = year if year is not None else int(config.data.get("tlc_year"))
    month = month if month is not None else int(config.data.get("tlc_month"))

    # 1. Ingest --------------------------------------------------------------
    if skip_download:
        trips_path = config.raw_dir / trip_filename(year, month)
        zones_path = config.raw_dir / "taxi_zone_lookup.csv"
    else:
        paths = ingest(config, year=year, month=month)
        trips_path, zones_path = paths["trips"], paths["zones"]

    raw_trips = load_trips(trips_path)
    _ = load_zone_lookup(zones_path)

    # 2. Clean / validate ----------------------------------------------------
    clean, report = clean_trips(raw_trips, config, year=year, month=month)
    _write_parquet(clean, config.processed_dir / "trips_clean.parquet")

    # 3. Demand aggregation --------------------------------------------------
    demand = aggregate_hourly_demand(clean)
    _write_parquet(demand, config.processed_dir / "demand_hourly.parquet")

    # 4. Feature engineering -------------------------------------------------
    demand_feats = build_demand_features(demand, config)
    fare_feats = build_fare_features(clean, config)
    _write_parquet(demand_feats, config.features_dir / "demand_features.parquet")
    _write_parquet(fare_feats, config.features_dir / "fare_features.parquet")

    # 5. Chronological splits ------------------------------------------------
    for name, df, time_col in (
        ("demand", demand_feats, "pickup_hour"),
        ("fare", fare_feats, "tpep_pickup_datetime"),
    ):
        train, val, test = time_based_split(df, time_col, config)
        _write_parquet(train, config.features_dir / f"{name}_train.parquet")
        _write_parquet(val, config.features_dir / f"{name}_val.parquet")
        _write_parquet(test, config.features_dir / f"{name}_test.parquet")

    summary = {
        "year": year,
        "month": month,
        "cleaning": report.as_dict(),
        "demand_feature_rows": len(demand_feats),
        "fare_feature_rows": len(fare_feats),
    }
    summary_path = config.processed_dir / "pipeline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Pipeline summary: %s", json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data pipeline.")
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--month", type=int, default=None)
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Reuse already-cached raw data instead of downloading.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(args.year, args.month, args.skip_download)


if __name__ == "__main__":
    main()
