"""Thin helpers around MLflow tracking configuration.

Centralises tracking-URI / experiment setup so training scripts can simply call
``configure_mlflow()`` and ``start_run(...)`` without repeating boilerplate.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import mlflow

from src.config import Config, get_config

logger = logging.getLogger(__name__)


def configure_mlflow(config: Config | None = None) -> None:
    """Point MLflow at the configured tracking store."""
    config = config or get_config()
    uri = config.mlflow.get("tracking_uri", "file:./mlruns")
    mlflow.set_tracking_uri(uri)
    logger.info("MLflow tracking URI: %s", uri)


def experiment_name(problem: str, config: Config | None = None) -> str:
    """Return the configured experiment name for 'demand' or 'fare'."""
    config = config or get_config()
    key = "experiment_demand" if problem == "demand" else "experiment_fare"
    default = f"taxi-{problem}"
    return config.mlflow.get(key, default)


def artifact_root(config: Config | None = None) -> str | None:
    """Return the configured MLflow artifact root (e.g. an S3 URI), or None.

    When set, new experiments are created with this ``artifact_location`` so run
    artifacts and models are stored in the cloud (e.g. ``s3://bucket/mlflow``)
    while metadata stays in the tracking backend.
    """
    config = config or get_config()
    root = config.mlflow.get("artifact_root") or None
    return root.strip() or None if isinstance(root, str) else root


def ensure_experiment(problem: str, config: Config | None = None) -> str:
    """Get-or-create the experiment for a problem, honouring the artifact root.

    Returns the experiment name. If an artifact root is configured and the
    experiment does not yet exist, it is created with that ``artifact_location``.
    """
    config = config or get_config()
    name = experiment_name(problem, config)
    root = artifact_root(config)
    existing = mlflow.get_experiment_by_name(name)
    if existing is None:
        if root:
            mlflow.create_experiment(name, artifact_location=root)
            logger.info("Created experiment %s with artifact root %s", name, root)
        else:
            mlflow.create_experiment(name)
    return name


@contextmanager
def start_run(
    problem: str,
    run_name: str | None = None,
    config: Config | None = None,
) -> Iterator[mlflow.ActiveRun]:
    """Context manager that configures MLflow and opens a run for a problem."""
    config = config or get_config()
    configure_mlflow(config)
    mlflow.set_experiment(ensure_experiment(problem, config))
    with mlflow.start_run(run_name=run_name) as run:
        yield run


# Library prefixes whose types may be stored in a saved model. MLflow 3 saves
# sklearn-style models with skops, which refuses to load types it was not told to
# trust; anything outside this list stops the save instead of being trusted.
TRUSTED_MODEL_TYPE_PREFIXES = ("xgboost.", "lightgbm.", "sklearn.", "numpy.", "collections.OrderedDict")


def log_sklearn_model(model, name: str = "model"):
    """Log ``model`` in the secure skops format, trusting only known library types.

    Returns MLflow's ``ModelInfo``; its ``model_uri`` is what gets registered.
    """
    import skops.io as sio

    needed = sio.get_untrusted_types(data=sio.dumps(model))
    unknown = [t for t in needed if not t.startswith(TRUSTED_MODEL_TYPE_PREFIXES)]
    if unknown:
        raise ValueError(f"Refusing to save a model containing untrusted types: {unknown}")
    return mlflow.sklearn.log_model(
        model, name=name, serialization_format="skops", skops_trusted_types=needed
    )

