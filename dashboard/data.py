"""Static-artifact loaders (zone lookup, model cards, fairness reports)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import pandas as pd

from dashboard.config import (
    drift_dir,
    model_cards_dir,
    responsible_dir,
    zone_lookup_path,
)


@lru_cache(maxsize=1)
def zone_lookup() -> pd.DataFrame:
    """LocationID -> Borough / Zone / service_zone."""
    return pd.read_csv(zone_lookup_path())


def zone_labels(location_ids: list[int]) -> dict[int, str]:
    """Human-readable ``"<id> - <Zone> (<Borough>)"`` labels for zone IDs."""
    lookup = zone_lookup().set_index("LocationID")
    labels: dict[int, str] = {}
    for loc in location_ids:
        if loc in lookup.index:
            row = lookup.loc[loc]
            labels[loc] = f"{loc} - {row['Zone']} ({row['Borough']})"
        else:
            labels[loc] = str(loc)
    return labels


def enrich_with_zones(df: pd.DataFrame, id_col: str = "zone") -> pd.DataFrame:
    """Left-join borough/zone names onto a frame with a zone-id column."""
    lookup = zone_lookup().rename(columns={"LocationID": id_col})
    return df.merge(
        lookup[[id_col, "Borough", "Zone"]], on=id_col, how="left"
    )


def model_card(problem: str) -> str | None:
    path = model_cards_dir() / f"{problem}.md"
    return path.read_text(encoding="utf-8") if path.exists() else None


def fairness_report(problem: str) -> dict[str, Any] | None:
    path = responsible_dir() / f"{problem}_fairness.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def shap_summary_path(problem: str):
    path = responsible_dir() / f"{problem}_shap_summary.png"
    return path if path.exists() else None


def _looks_like_drift_report(obj: Any) -> bool:
    return isinstance(obj, dict) and "drifted" in obj and (
        "features" in obj or obj.get("status") == "no_data"
    )


def drift_reports() -> list[dict[str, Any]]:
    """Load drift-job outputs from ``drift_dir()``.

    Reads every ``*.json`` file that matches the drift-report schema written by
    ``scripts/monitoring/run_drift_job.py``. Each returned dict is the parsed
    report augmented with ``_source`` (filename) and ``_mtime`` (last modified).
    Reports are de-duplicated by ``problem``, keeping the most recently written,
    and sorted by problem name. boto3/S3/AWS are not required - this reads the
    local artifacts the job (or its local dry-run) leaves behind.
    """
    directory = drift_dir()
    if not directory.exists():
        return []

    by_problem: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.json")):
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not _looks_like_drift_report(obj):
            continue
        obj["_source"] = path.name
        obj["_mtime"] = path.stat().st_mtime
        problem = obj.get("problem", path.stem)
        existing = by_problem.get(problem)
        if existing is None or obj["_mtime"] >= existing["_mtime"]:
            by_problem[problem] = obj
    return [by_problem[p] for p in sorted(by_problem)]
