"""Runtime configuration for the dashboard.

Paths and the API base URL are resolved from environment variables so the same
code runs locally (against the repo layout) and inside the container (where the
artifacts are copied to fixed locations).
"""

from __future__ import annotations

import os
from pathlib import Path

# Repo root = two levels up from this file (dashboard/ -> repo/).
_REPO_ROOT = Path(__file__).resolve().parents[1]


def api_base_url() -> str:
    """Base URL of the prediction API (trailing slash stripped)."""
    return os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")


def _path(env_var: str, default: Path) -> Path:
    value = os.environ.get(env_var)
    return Path(value) if value else default


def zone_lookup_path() -> Path:
    return _path("ZONE_LOOKUP_PATH", _REPO_ROOT / "data" / "raw" / "taxi_zone_lookup.csv")


def model_cards_dir() -> Path:
    return _path("MODEL_CARDS_DIR", _REPO_ROOT / "docs" / "model_cards")


def responsible_dir() -> Path:
    return _path("RESPONSIBLE_DIR", _REPO_ROOT / "artifacts" / "responsible")


def drift_dir() -> Path:
    """Directory of drift-job outputs (``<problem>_drift_metrics.json``)."""
    return _path("DRIFT_DIR", _REPO_ROOT / "artifacts" / "drift")
