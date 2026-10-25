"""Ingest NYC TLC Yellow Taxi trip data and the taxi-zone lookup table.

The TLC publishes monthly Parquet files of trip records plus a small CSV that
maps ``LocationID`` values to boroughs/zones. This module downloads (and caches)
those files into the configured raw-data directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import Config, get_config

logger = logging.getLogger(__name__)

_CHUNK = 1 << 20  # 1 MiB download chunks.


def _download(url: str, dest: Path, *, overwrite: bool = False) -> Path:
    """Stream ``url`` to ``dest``, skipping the download if already cached."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        logger.info("Using cached file: %s", dest)
        return dest

    logger.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
        tmp.replace(dest)
    return dest


def trip_filename(year: int, month: int) -> str:
    """Return the canonical TLC Parquet filename for a year/month."""
    return f"yellow_tripdata_{year:04d}-{month:02d}.parquet"


def download_trips(
    config: Config | None = None,
    *,
    year: int | None = None,
    month: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a month of Yellow Taxi trip records, returning the local path."""
    config = config or get_config()
    year = year if year is not None else int(config.data.get("tlc_year", 2024))
    month = month if month is not None else int(config.data.get("tlc_month", 1))

    fname = trip_filename(year, month)
    url = f"{config.data['base_url'].rstrip('/')}/{fname}"
    dest = config.raw_dir / fname
    return _download(url, dest, overwrite=overwrite)


def download_zone_lookup(
    config: Config | None = None, *, overwrite: bool = False
) -> Path:
    """Download the taxi-zone lookup CSV, returning the local path."""
    config = config or get_config()
    url = config.data["zone_lookup_url"]
    dest = config.raw_dir / "taxi_zone_lookup.csv"
    return _download(url, dest, overwrite=overwrite)


def load_trips(path: str | Path) -> pd.DataFrame:
    """Load a TLC trip Parquet file into a DataFrame."""
    return pd.read_parquet(path)


def load_zone_lookup(path: str | Path) -> pd.DataFrame:
    """Load the taxi-zone lookup CSV into a DataFrame."""
    return pd.read_csv(path)


def ingest(
    config: Config | None = None,
    *,
    year: int | None = None,
    month: int | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download both the trip data and zone lookup; return their paths."""
    config = config or get_config()
    config.ensure_dirs()
    return {
        "trips": download_trips(
            config, year=year, month=month, overwrite=overwrite
        ),
        "zones": download_zone_lookup(config, overwrite=overwrite),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    paths = ingest()
    for key, value in paths.items():
        print(f"{key}: {value}")
