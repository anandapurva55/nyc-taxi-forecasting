"""Offline tests for the retraining quality gate and deployment orchestration."""

from __future__ import annotations

import json
import sqlite3

import pytest

from retraining import build_and_deploy, pipeline, train_step
from scripts.monitoring import evaluate_models


def test_prepare_baseline_registry_repoints_registered_model_source(tmp_path):
    source = tmp_path / "source"
    (source / "mlruns").mkdir(parents=True)
    with sqlite3.connect(source / "mlflow.db") as connection:
        connection.execute("CREATE TABLE model_versions (source TEXT, storage_location TEXT)")
        windows_uri = "file:///C:/taxi-forecasting/mlruns/1/run/artifacts/model"
        connection.execute(
            "INSERT INTO model_versions VALUES (?, ?)", (windows_uri, windows_uri)
        )

    output = tmp_path / "output"
    evaluate_models.prepare_baseline_registry(source, output)

    with sqlite3.connect(output / "mlflow.db") as connection:
        source_uri, storage_uri = connection.execute(
            "SELECT source, storage_location FROM model_versions"
        ).fetchone()
    expected = f"file://{(output / 'mlruns').as_posix()}/1/run/artifacts/model"
    assert source_uri == expected
    assert storage_uri == expected


def test_evaluate_models_writes_quality_gate_report(monkeypatch, tmp_path):
    results = {
        "demand": {"rmse": 12.5, "mae": 8.0},
        "fare": {"rmse": 3.2, "mae": 2.1},
    }
    monkeypatch.setattr(evaluate_models, "PROBLEMS", ("demand", "fare"))
    monkeypatch.setattr(
        evaluate_models, "evaluate", lambda problem, _config=None: results[problem]
    )

    assert evaluate_models.main(["--output-dir", str(tmp_path)]) == 0
    assert json.loads((tmp_path / "evaluation.json").read_text()) == results


def test_evaluate_models_compares_candidate_with_seed_registry(monkeypatch, tmp_path):
    candidate = tmp_path / "candidate"
    baseline = tmp_path / "baseline"
    candidate.mkdir()
    baseline.mkdir()
    calls = []
    prepared = []

    def fake_evaluate(problem, config):
        tracking_uri = config.mlflow["tracking_uri"]
        calls.append((problem, tracking_uri))
        rmse = 12.1 if "candidate" in tracking_uri else 10.0
        return {"rmse": rmse, "mae": 5.0, "mape": 4.0, "r2": 0.9}

    monkeypatch.setattr(evaluate_models, "PROBLEMS", ("demand",))
    monkeypatch.setattr(evaluate_models, "evaluate", fake_evaluate)
    monkeypatch.setattr(
        evaluate_models,
        "prepare_baseline_registry",
        lambda source, output: prepared.append((source, output)),
    )

    result = evaluate_models.main(
        [
            "--output-dir",
            str(tmp_path / "evaluation"),
            "--registry-dir",
            str(candidate),
            "--baseline-registry-dir",
            str(baseline),
            "--baseline-working-dir",
            str(tmp_path / "prepared-baseline"),
        ]
    )

    report = json.loads((tmp_path / "evaluation" / "evaluation.json").read_text())
    assert result == 0
    assert report["demand"]["baseline_gate_passed"] is False
    assert report["demand"]["baseline"]["rmse"] == 10.0
    assert report["demand"]["baseline_comparison"]["rmse"]["rel_change"] == pytest.approx(
        0.21
    )
    assert [problem for problem, _uri in calls] == ["demand", "demand"]
    assert prepared == [(baseline, tmp_path / "prepared-baseline")]


def test_pipeline_cli_prints_definition_without_upsert(monkeypatch, capsys):
    class FakePipeline:
        def definition(self):
            return '{"name": "taxi-forecasting-retrain"}'

    monkeypatch.setattr(pipeline, "build_pipeline", lambda **_kwargs: FakePipeline())

    result = pipeline.main(
        [
            "--role-arn",
            "role",
            "--image-uri",
            "training:latest",
            "--features-s3-uri",
            "s3://bucket/features",
            "--registry-seed-s3-uri",
            "s3://bucket/registry",
            "--artifact-bucket",
            "bucket",
        ]
    )

    assert result == 0
    assert "taxi-forecasting-retrain" in capsys.readouterr().out


def test_pipeline_cli_upserts_when_requested(monkeypatch):
    calls = []

    class FakePipeline:
        def upsert(self, *, role_arn):
            calls.append(role_arn)

    monkeypatch.setattr(pipeline, "build_pipeline", lambda **_kwargs: FakePipeline())

    result = pipeline.main(
        [
            "--role-arn",
            "arn:aws:iam::123456789012:role/retrain",
            "--image-uri",
            "training:latest",
            "--features-s3-uri",
            "s3://bucket/features",
            "--registry-seed-s3-uri",
            "s3://bucket/registry",
            "--artifact-bucket",
            "bucket",
            "--upsert",
        ]
    )

    assert result == 0
    assert calls == ["arn:aws:iam::123456789012:role/retrain"]


def test_prepare_registry_repoints_seed(tmp_path):
    import sqlite3

    seed = tmp_path / "seed"
    (seed / "mlruns").mkdir(parents=True)
    database = seed / "mlflow.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE experiments (artifact_location TEXT)")
        connection.execute(
            "INSERT INTO experiments VALUES (?)", ("file:///old/path/mlruns/1",)
        )

    output = tmp_path / "output"
    output_db = train_step.prepare_registry(seed, output)

    with sqlite3.connect(output_db) as connection:
        value = connection.execute("SELECT artifact_location FROM experiments").fetchone()[0]
    assert value == f"file://{(output / 'mlruns').as_posix()}/1"


def test_upload_registry_publishes_database_and_runs(tmp_path):
    model_dir = tmp_path / "model"
    (model_dir / "mlruns" / "1").mkdir(parents=True)
    (model_dir / "mlflow.db").write_text("database")
    (model_dir / "mlruns" / "1" / "model.pkl").write_text("model")

    class FakeS3:
        def __init__(self):
            self.uploads = []

        def upload_file(self, source, bucket, key):
            self.uploads.append((source, bucket, key))

    client = FakeS3()
    assert build_and_deploy.upload_registry(model_dir, "artifacts", client) == 2
    assert [key for _source, _bucket, key in client.uploads] == [
        "mlflow.db",
        "mlruns/1/model.pkl",
    ]


def test_start_deployment_build_uses_immutable_release_id():
    class FakeCodeBuild:
        def start_build(self, **kwargs):
            assert kwargs["projectName"] == "taxi-ci"
            assert kwargs["environmentVariablesOverride"][0]["value"] == "release-7"
            return {"build": {"id": "taxi-ci:build-1"}}

    assert (
        build_and_deploy.start_deployment_build("taxi-ci", "release-7", FakeCodeBuild())
        == "taxi-ci:build-1"
    )


def test_wait_for_build_returns_terminal_status():
    class FakeCodeBuild:
        def batch_get_builds(self, *, ids):
            assert ids == ["taxi-ci:build-1"]
            return {"builds": [{"buildStatus": "SUCCEEDED"}]}

    assert build_and_deploy.wait_for_build("taxi-ci:build-1", FakeCodeBuild(), 0) == "SUCCEEDED"
