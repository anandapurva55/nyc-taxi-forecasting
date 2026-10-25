"""Unit tests for the cloud-training helpers (offline, no AWS SDK)."""

from __future__ import annotations

from src.aws.sagemaker import (
    DRIFT_NAMESPACE,
    DRIFT_OUTPUT_DIR,
    build_training_env,
    drift_job_arguments,
    training_job_arguments,
)
from src.config import Config


def _config(**mlflow_over) -> Config:
    mlflow = {
        "tracking_uri": "sqlite:///mlflow.db",
        "experiment_demand": "taxi-demand-forecasting",
        "experiment_fare": "taxi-fare-prediction",
        "artifact_root": "",
    }
    mlflow.update(mlflow_over)
    return Config(raw={"mlflow": mlflow})


# --- build_training_env ------------------------------------------------------
def test_build_training_env_omits_empty_artifact_root():
    env = build_training_env(_config())
    assert env["MLFLOW_TRACKING_URI"] == "sqlite:///mlflow.db"
    assert env["MLFLOW_EXPERIMENT_DEMAND"] == "taxi-demand-forecasting"
    assert env["MLFLOW_EXPERIMENT_FARE"] == "taxi-fare-prediction"
    assert "MLFLOW_ARTIFACT_ROOT" not in env


def test_build_training_env_includes_s3_artifact_root():
    env = build_training_env(_config(artifact_root="s3://bucket/mlflow"))
    assert env["MLFLOW_ARTIFACT_ROOT"] == "s3://bucket/mlflow"


# --- training_job_arguments --------------------------------------------------
def test_training_job_arguments_defaults():
    assert training_job_arguments() == ["--problem", "both"]


def test_training_job_arguments_full():
    args = training_job_arguments(problem="demand", n_trials=10, register=False)
    assert args == ["--problem", "demand", "--n-trials", "10", "--no-register"]


def test_training_job_arguments_register_true_omits_flag():
    assert "--no-register" not in training_job_arguments(register=True)


# --- drift_job_arguments -----------------------------------------------------
def test_drift_job_arguments_defaults():
    args = drift_job_arguments("fare")
    assert args == [
        "--problem", "fare",
        "--output-dir", DRIFT_OUTPUT_DIR,
        "--namespace", DRIFT_NAMESPACE,
    ]


def test_drift_job_arguments_with_current_s3():
    args = drift_job_arguments("demand", "s3://bucket/captured/")
    assert "--current-s3" in args
    assert args[args.index("--current-s3") + 1] == "s3://bucket/captured/"


def test_drift_job_arguments_full():
    args = drift_job_arguments(
        "fare",
        "s3://bucket/captured/",
        psi_threshold=0.3,
        ks_alpha=0.01,
        no_cloudwatch=True,
    )
    assert args[-1] == "--no-cloudwatch"
    assert args[args.index("--psi-threshold") + 1] == "0.3"
    assert args[args.index("--ks-alpha") + 1] == "0.01"
