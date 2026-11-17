"""Load models from the MLflow Model Registry.

The training pipeline registers the best model per problem under the names in
``config.models`` (``taxi-demand-forecaster`` / ``taxi-fare-predictor``) and points
the ``champion`` alias at it. Here we resolve a registry reference (an alias such
as ``"champion"``, a version number, or ``"latest"``) and load the
sklearn-flavoured model, configuring the tracking URI first.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException

from src.config import Config, get_config
from src.models.common import registry_model_name
from src.utils.mlflow_utils import configure_mlflow

logger = logging.getLogger(__name__)
_CONTAINER_MLRUNS = Path("/app/mlruns")


# Legacy MLflow stages; any other non-numeric reference is treated as an alias.
_LEGACY_STAGES = {"production", "staging", "archived", "none"}


def _reference(config: Config) -> str:
    return str(config.serving.get("model_stage", "latest"))


def _is_alias(reference: str) -> bool:
    return not reference.isdigit() and reference.lower() not in {"latest", *_LEGACY_STAGES}


def model_uri(problem: str, config: Config | None = None) -> str:
    """Build the ``models:/`` URI for a problem's registered model."""
    config = config or get_config()
    name = registry_model_name(problem, config)
    reference = _reference(config)
    if _is_alias(reference):
        return f"models:/{name}@{reference}"
    return f"models:/{name}/{reference}"


def _model_version(problem: str, config: Config):
    """Resolve the configured reference to a concrete registered ModelVersion."""
    name = registry_model_name(problem, config)
    reference = _reference(config)
    client = mlflow.MlflowClient()
    if reference.isdigit():
        return client.get_model_version(name, reference)
    if reference.lower() == "latest":
        versions = client.search_model_versions(f"name='{name}'")
        if not versions:
            raise MlflowException(f"No registered versions found for {name}")
        return max(versions, key=lambda item: int(item.version))
    if _is_alias(reference):
        return client.get_model_version_by_alias(name, reference)
    versions = client.get_latest_versions(name, stages=[reference])
    if not versions:
        raise MlflowException(f"No {reference} version found for {name}")
    return versions[0]


def resolve_version(problem: str, config: Config | None = None) -> dict | None:
    """Return ``{name, version, run_id, reference}`` for the model being served.

    Used to stamp predictions and ``/version`` with the exact model version
    instead of a mutable reference. Returns None if the registry is unreachable.
    """
    config = config or get_config()
    try:
        configure_mlflow(config)
        version = _model_version(problem, config)
    except Exception as exc:  # pragma: no cover - depends on registry availability
        logger.warning("Could not resolve %s model version: %s", problem, exc)
        return None
    return {
        "name": version.name,
        "version": int(version.version),
        "run_id": version.run_id,
        "reference": _reference(config),
    }


def _container_model_uri(problem: str, config: Config) -> str:
    """Map a host-backed registry source to the mounted container artifact tree."""
    version = _model_version(problem, config)
    source = str(version.source).replace("\\", "/")
    marker = "/mlruns/"
    if marker not in source:
        raise MlflowException(f"Model source is not under mlruns: {source}")
    path = _CONTAINER_MLRUNS / source.split(marker, 1)[1]
    if not path.exists():
        raise MlflowException(f"Mounted model artifact not found: {path}")
    return path.as_uri()


def _container_artifact_path(run_id: str, artifact: str) -> Path:
    """Map a run artifact recorded with a host path to the mounted container tree."""
    artifact_uri = str(mlflow.MlflowClient().get_run(run_id).info.artifact_uri).replace("\\", "/")
    marker = "/mlruns/"
    if marker not in artifact_uri:
        raise MlflowException(f"Run artifacts are not under mlruns: {artifact_uri}")
    path = _CONTAINER_MLRUNS / artifact_uri.split(marker, 1)[1] / artifact
    if not path.exists():
        raise MlflowException(f"Mounted run artifact not found: {path}")
    return path


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


def load_duration_lookup(config: Config | None = None):
    """Load the booking-time duration lookup logged with the served fare model.

    Returns None (callers fall back to a fixed minutes-per-mile estimate) when the
    fare model predates the lookup or the artifact cannot be read.
    """
    from src.features.duration import DurationLookup

    config = config or get_config()
    info = resolve_version("fare", config)
    if info is None:
        return None
    try:
        try:
            path = Path(
                mlflow.artifacts.download_artifacts(
                    run_id=info["run_id"], artifact_path="duration_lookup.json"
                )
            )
        except (MlflowException, OSError):
            if not _CONTAINER_MLRUNS.is_dir():
                raise
            path = _container_artifact_path(info["run_id"], "duration_lookup.json")
        return DurationLookup.from_dict(json.loads(path.read_text()))
    except Exception as exc:  # pragma: no cover - depends on registry contents
        logger.warning("No duration lookup for fare model %s: %s", info["version"], exc)
        return None
