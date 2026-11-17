"""The baked registry must point at /app/mlruns whatever OS trained the models."""

from __future__ import annotations

import importlib.util
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    spec = importlib.util.spec_from_file_location("portable", ROOT / "scripts" / "make_mlflow_portable.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repoints_plain_paths_and_file_uris_but_not_remote_stores(tmp_path):
    src = tmp_path / "mlflow.db"
    with sqlite3.connect(src) as conn:
        conn.execute("CREATE TABLE model_versions (source TEXT)")
        conn.executemany(
            "INSERT INTO model_versions VALUES (?)",
            [
                ("/Users/me/project/mlruns/1/abc/artifacts/model",),
                ("file:///C:/project/mlruns/2/def/artifacts/model",),
                ("s3://bucket/mlruns/3/ghi/artifacts/model",),
            ],
        )
    dest = tmp_path / "mlflow.container.db"
    changed = _module().make_portable(src, dest, "file:///app/mlruns")
    with sqlite3.connect(dest) as conn:
        values = [row[0] for row in conn.execute("SELECT source FROM model_versions ORDER BY rowid")]
    assert changed == 2
    assert values == [
        "file:///app/mlruns/1/abc/artifacts/model",
        "file:///app/mlruns/2/def/artifacts/model",
        "s3://bucket/mlruns/3/ghi/artifacts/model",
    ]


def test_model_logging_trusts_only_known_library_types(tmp_path, monkeypatch):
    import mlflow.sklearn
    import pytest
    from sklearn.linear_model import LinearRegression

    from src.utils import mlflow_utils

    calls = []
    monkeypatch.setattr(mlflow.sklearn, "log_model", lambda model, **kw: calls.append(kw) or "info")
    model = LinearRegression().fit([[0.0], [1.0]], [0.0, 1.0])
    assert mlflow_utils.log_sklearn_model(model) == "info"
    assert calls[0]["serialization_format"] == "skops" and calls[0]["name"] == "model"

    class Custom:  # an arbitrary class must never be trusted silently
        pass

    with pytest.raises(ValueError, match="untrusted types"):
        mlflow_utils.log_sklearn_model({"model": Custom()})
