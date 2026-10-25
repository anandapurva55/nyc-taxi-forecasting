"""Load models from the MLflow Model Registry.

The training pipeline registers the best model per problem under the names in
``config.models`` (``taxi-demand-forecaster`` / ``taxi-fare-predictor``). Here we
resolve a registry reference (a version number or ``"latest"``) and load the
sklearn-flavoured model, configuring the tracking URI first.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from src.config import Config, get_config
from src.models.common import registry_model_name
from src.utils.mlflow_utils import configure_mlflow

logger = logging.getLogger(__name__)
_CONTAINER_MLRUNS = Path("/app/mlruns")


def model_uri(problem: str, config: Config | None = None) -> str:
    """Build the ``models:/`` URI for a problem's registered model."""
    config = config or get_config()
    name = registry_model_name(problem, config)
    stage = str(config.serving.get("model_stage", "latest"))
    return f"models:/{name}/{stage}"


def _container_model_uri(problem: str, config: Config) -> str:
    """Map a host-backed registry source to the mounted container artifact tree."""
    name = registry_model_name(problem, config)
    stage = str(config.serving.get("model_stage", "latest"))
    client = mlflow.MlflowClient()
    if stage.isdigit():
        version = client.get_model_version(name, stage)
    elif stage.lower() == "latest":
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise MlflowException(f"No registered versions found for {name}")
        version = max(versions, key=lambda item: int(item.version))
    else:
        versions = client.get_latest_versions(name, stages=[stage])
        if not versions:
            raise MlflowException(f"No {stage} version found for {name}")
        version = versions[0]

    source = str(version.source).replace("\\", "/")
    marker = "/mlruns/"
    if marker not in source:
        raise MlflowException(f"Model source is not under mlruns: {source}")
    path = _CONTAINER_MLRUNS / source.split(marker, 1)[1]
    if not path.exists():
        raise MlflowException(f"Mounted model artifact not found: {path}")
    return path.as_uri()


def load_model(problem: str, config: Config | None = None):
    """Load the registered model for ``problem`` from the MLflow registry."""
    config = config or get_config()
    configure_mlflow(config)
    uri = model_uri(problem, config)
    logger.info("Loading model for %s from %s", problem, uri)
    try:
        return mlflow.sklearn.load_model(uri)
    except (MlflowException, OSError):
        if not _CONTAINER_MLRUNS.is_dir():
            raise
        container_uri = _container_model_uri(problem, config)
        logger.info("Retrying %s model from mounted artifact %s", problem, container_uri)
        return mlflow.sklearn.load_model(container_uri)
