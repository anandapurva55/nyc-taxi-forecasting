"""Unit tests for MLflow artifact-root configuration."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlflow

from src.config import Config
from src.utils import mlflow_utils
from src.utils.mlflow_utils import artifact_root, ensure_experiment


def _config(tmp_path, **mlflow_over) -> Config:
    mlflow_cfg = {
        "tracking_uri": f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        "experiment_demand": "taxi-demand-forecasting",
        "experiment_fare": "taxi-fare-prediction",
        "artifact_root": "",
    }
    mlflow_cfg.update(mlflow_over)
    return Config(raw={"mlflow": mlflow_cfg})


def test_artifact_root_empty_is_none():
    assert artifact_root(Config(raw={"mlflow": {"artifact_root": ""}})) is None
    assert artifact_root(Config(raw={"mlflow": {"artifact_root": "   "}})) is None


def test_artifact_root_returns_uri():
    cfg = Config(raw={"mlflow": {"artifact_root": "s3://bucket/mlflow"}})
    assert artifact_root(cfg) == "s3://bucket/mlflow"


def test_ensure_experiment_creates_with_artifact_location(tmp_path):
    root = (tmp_path / "artifacts").as_uri()
    cfg = _config(tmp_path, artifact_root=root)
    mlflow.set_tracking_uri(cfg.mlflow["tracking_uri"])

    name = ensure_experiment("demand", cfg)
    assert name == "taxi-demand-forecasting"

    exp = mlflow.get_experiment_by_name(name)
    assert exp is not None
    assert exp.artifact_location == root

    # Idempotent: a second call returns the same experiment without error.
    assert ensure_experiment("demand", cfg) == name


def test_configure_mlflow_sets_tracking_uri(monkeypatch):
    config = Config(raw={"mlflow": {"tracking_uri": "sqlite:///tracking.db"}})
    configured = []
    monkeypatch.setattr(mlflow_utils.mlflow, "set_tracking_uri", configured.append)

    mlflow_utils.configure_mlflow(config)

    assert configured == ["sqlite:///tracking.db"]


def test_start_run_configures_experiment_and_yields_run(monkeypatch):
    config = Config(raw={})
    calls = []
    active_run = SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    monkeypatch.setattr(
        mlflow_utils,
        "configure_mlflow",
        lambda received: calls.append(("configure", received)),
    )
    monkeypatch.setattr(
        mlflow_utils,
        "ensure_experiment",
        lambda problem, received: calls.append(("ensure", problem, received)) or "experiment",
    )
    monkeypatch.setattr(
        mlflow_utils.mlflow,
        "set_experiment",
        lambda name: calls.append(("set", name)),
    )

    @contextmanager
    def fake_mlflow_run(run_name=None):
        calls.append(("run", run_name))
        yield active_run

    monkeypatch.setattr(mlflow_utils.mlflow, "start_run", fake_mlflow_run)

    with mlflow_utils.start_run("fare", "candidate", config) as run:
        assert run is active_run

    assert calls == [
        ("configure", config),
        ("ensure", "fare", config),
        ("set", "experiment"),
        ("run", "candidate"),
    ]
