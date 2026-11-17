"""The static demo answers only from the recording, and answers exactly what was recorded."""

from __future__ import annotations

import json

import pytest


@pytest.fixture()
def recorded(tmp_path, monkeypatch):
    recording = {
        "build": {"git_commit": "abc1234", "recorded_at": "2026-11-17T13:31:39Z", "models": {}},
        "health": {"status": "ok", "zones": 2},
        "forecast": {
            "hours": ["2024-03-01T00:00:00", "2024-03-01T01:00:00", "2024-03-01T02:00:00"],
            "methods": ["model", "model", "same_hour_last_week"],
            "model_horizon_hours": 2,
            "model_reference": "taxi-demand-forecaster/2",
            "zones": {"132": [178.2, 55.5, 15.0], "230": [40.0, 21.0, 9.0]},
        },
        "fare_samples": [
            {"label": "JFK to Times Square", "request": {"trip_distance": 17.0, "pu_location_id": 132},
             "prediction": {"predicted_fare": 69.94}, "explanation": {"prediction": 69.94}},
        ],
        "demand_samples": [
            {"zone": 132, "target_hour": "2024-02-29T20:00:00",
             "prediction": {"predicted_pickups": 120.0}, "explanation": {"prediction": 120.0}},
        ],
    }
    path = tmp_path / "recording.json"
    path.write_text(json.dumps(recording))
    monkeypatch.setenv("DEMO_RECORDING", str(path))
    from dashboard import demo_backend

    demo_backend.recording.cache_clear()
    yield demo_backend
    demo_backend.recording.cache_clear()


def test_forecast_slices_the_recorded_hours_and_zones(recorded):
    out = recorded.forecast_demand(2, (132,))
    assert [p["predicted_pickups"] for p in out["points"]] == [178.2, 55.5]
    assert [p["method"] for p in out["points"]] == ["model", "model"]
    everything = recorded.forecast_demand(72)
    assert everything["horizon_hours"] == 3 and everything["zones"] == 2
    assert recorded.known_zones() == [132, 230]


def test_demand_explanations_come_only_from_recorded_requests(recorded):
    assert recorded.explain_demand(132, "2024-02-29T20:00:00")["prediction"] == 120.0
    with pytest.raises(recorded.NotRecorded):
        recorded.explain_demand(230, "2024-02-29T20:00:00")


def test_browser_fare_models_reproduce_the_recorded_api_fares(monkeypatch):
    """The demo prices any trip with the exported models; they must match the API exactly."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    if not (root / "demo" / "models" / "fare.txt").exists():
        pytest.skip("demo not built")
    monkeypatch.setenv("DEMO_MODELS_DIR", str(root / "demo" / "models"))
    from dashboard import demo_models

    demo_models.model.cache_clear()
    rec = json.loads((root / "demo" / "recording.json").read_text())
    for sample in rec["fare_samples"]:
        local = demo_models.predict_fare(sample["request"], rec["build"]["models"])
        for key in ("predicted_fare", "model", "low_confidence", "notes"):
            assert local[key] == sample["prediction"][key], (sample["label"], key)
        explained = demo_models.explain_fare(sample["request"])
        assert explained["prediction"] == pytest.approx(sample["explanation"]["prediction"], abs=1e-3)
        assert explained["base_value"] + sum(c["value"] for c in explained["contributions"]) == pytest.approx(
            explained["prediction"], abs=1e-3
        )
    demo_models.model.cache_clear()


def test_committed_recording_matches_the_demo_inputs():
    """The recording shipped with the site has every page's data."""
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "demo" / "recording.json"
    if not path.exists():
        pytest.skip("demo not built")
    rec = json.loads(path.read_text())
    assert len(rec["forecast"]["hours"]) == 72
    assert all(len(v) == 72 for v in rec["forecast"]["zones"].values())
    assert len(rec["fare_samples"]) >= 5 and len(rec["demand_samples"]) >= 5
    assert rec["health"]["status"] == "ok"


def test_demo_page_mounts_every_web_app_file_and_model_card():
    """A web app file added without rebuilding the demo would break the site silently."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    page = root / "demo" / "index.html"
    if not page.exists():
        pytest.skip("demo not built")
    html = page.read_text()
    needed = [p.relative_to(root).as_posix() for p in (root / "dashboard").glob("*.py")]
    needed += [p.relative_to(root).as_posix() for p in (root / "docs" / "model_cards").glob("*.md")]
    needed += ["src/serving/predict.py", "src/features/fare_features.py", "config/config.yaml",
               "demo/models/fare.txt", "demo/models/fare_booking.txt"]
    missing = [path for path in needed if f'"{path}"' not in html]
    assert not missing, f"Rebuild the demo (python -m scripts.build_static_demo); not mounted: {missing}"
