"""Configuration loading utilities.

Loads settings from ``config/config.yaml`` and allows selected values to be
overridden via environment variables (optionally sourced from a ``.env`` file).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is optional at import time.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dotenv always installed in practice.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def _resolve(path: str | Path) -> Path:
    """Resolve a possibly-relative path against the project root."""
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


@dataclass
class Config:
    """Typed view over the raw configuration dictionary."""

    raw: dict[str, Any] = field(default_factory=dict)

    # --- Convenience accessors -------------------------------------------------
    @property
    def data(self) -> dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def cleaning(self) -> dict[str, Any]:
        return self.raw.get("cleaning", {})

    @property
    def split(self) -> dict[str, Any]:
        return self.raw.get("split", {})

    @property
    def features(self) -> dict[str, Any]:
        return self.raw.get("features", {})

    @property
    def mlflow(self) -> dict[str, Any]:
        return self.raw.get("mlflow", {})

    @property
    def models(self) -> dict[str, Any]:
        return self.raw.get("models", {})

    @property
    def serving(self) -> dict[str, Any]:
        return self.raw.get("serving", {})

    @property
    def raw_dir(self) -> Path:
        return _resolve(self.data.get("raw_dir", "data/raw"))

    @property
    def processed_dir(self) -> Path:
        return _resolve(self.data.get("processed_dir", "data/processed"))

    @property
    def features_dir(self) -> Path:
        return _resolve(self.data.get("features_dir", "data/features"))

    def ensure_dirs(self) -> None:
        """Create the data directories if they do not yet exist."""
        for d in (self.raw_dir, self.processed_dir, self.features_dir):
            d.mkdir(parents=True, exist_ok=True)


def _apply_env_overrides(cfg: dict[str, Any]) -> dict[str, Any]:
    """Override a handful of config values from environment variables."""
    data = cfg.setdefault("data", {})
    if (year := os.getenv("TLC_TRIP_YEAR")) is not None:
        data["tlc_year"] = int(year)
    if (month := os.getenv("TLC_TRIP_MONTH")) is not None:
        data["tlc_month"] = int(month)
    for env_key, cfg_key in (
        ("DATA_RAW_DIR", "raw_dir"),
        ("DATA_PROCESSED_DIR", "processed_dir"),
        ("DATA_FEATURES_DIR", "features_dir"),
    ):
        if (val := os.getenv(env_key)) is not None:
            data[cfg_key] = val

    mlflow_cfg = cfg.setdefault("mlflow", {})
    if (uri := os.getenv("MLFLOW_TRACKING_URI")) is not None:
        mlflow_cfg["tracking_uri"] = uri
    if (exp := os.getenv("MLFLOW_EXPERIMENT_DEMAND")) is not None:
        mlflow_cfg["experiment_demand"] = exp
    if (exp := os.getenv("MLFLOW_EXPERIMENT_FARE")) is not None:
        mlflow_cfg["experiment_fare"] = exp
    if (root := os.getenv("MLFLOW_ARTIFACT_ROOT")) is not None:
        mlflow_cfg["artifact_root"] = root
    return cfg


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML with environment overrides applied."""
    if load_dotenv is not None:
        load_dotenv(PROJECT_ROOT / ".env")

    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    cfg = _apply_env_overrides(cfg)
    return Config(raw=cfg)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return a cached, process-wide configuration instance."""
    return load_config()
