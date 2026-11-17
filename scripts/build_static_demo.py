"""Record the static demo: everything the web app needs when there is no server.

    python -m scripts.build_static_demo        (needs the trained models and the pipeline outputs)

GitHub Pages cannot run the API, so this script starts the real API in-process
with the registered models, calls it once, and writes down what it answered:

    demo/recording.json        health, the build, a 72-hour forecast for every zone,
                               sample fare quotes with their SHAP explanations, and
                               demand explanations for sample zones and hours
    demo/models/               the two fare models as LightGBM model text, which the
                               browser runs so visitors can price any trip; the build
                               fails unless they reproduce the API's recorded fares
    demo/assets/               the fairness reports, SHAP summaries, drift reports
                               and zone lookup the web app reads from disk
    demo/index.html            loads the web app with stlite (Streamlit in the browser)
                               and mounts the files above

Nothing is written by hand: running this again after a change records the new
answers. Only the fare models run in the browser, and the demo costs nothing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
ASSETS = DEMO / "assets"
STLITE_VERSION = "1.12.2"
FORECAST_HOURS = 72

# (label, request). Trips without a duration are quoted by the booking-time model.
FARE_TRIPS = [
    ("JFK Airport to Times Square, 17 mi, Tue 18:30, quoted before the trip",
     {"trip_distance": 17.0, "pu_location_id": 132, "do_location_id": 230, "passenger_count": 1,
      "pickup_datetime": "2024-02-27T18:30:00"}),
    ("JFK Airport to Times Square, 17 mi, Tue 18:30, 45-minute trip",
     {"trip_distance": 17.0, "pu_location_id": 132, "do_location_id": 230, "passenger_count": 1,
      "pickup_datetime": "2024-02-27T18:30:00", "trip_duration_min": 45.0}),
    ("Midtown Center to Times Square, 0.9 mi, Wed 08:15",
     {"trip_distance": 0.9, "pu_location_id": 161, "do_location_id": 230, "passenger_count": 1,
      "pickup_datetime": "2024-02-28T08:15:00"}),
    ("LaGuardia Airport to Upper East Side South, 9.5 mi, Wed 17:45",
     {"trip_distance": 9.5, "pu_location_id": 138, "do_location_id": 237, "passenger_count": 2,
      "pickup_datetime": "2024-02-28T17:45:00"}),
    ("Upper West Side North to Financial District North, 7.2 mi, Thu 07:40, 32-minute trip",
     {"trip_distance": 7.2, "pu_location_id": 238, "do_location_id": 87, "passenger_count": 1,
      "pickup_datetime": "2024-02-29T07:40:00", "trip_duration_min": 32.0}),
    ("Crown Heights North to Williamsburg, 3.8 mi, Thu 23:10",
     {"trip_distance": 3.8, "pu_location_id": 61, "do_location_id": 255, "passenger_count": 1,
      "pickup_datetime": "2024-02-29T23:10:00"}),
    ("Penn Station to East Village, 2.1 mi, Tue 13:00",
     {"trip_distance": 2.1, "pu_location_id": 186, "do_location_id": 79, "passenger_count": 3,
      "pickup_datetime": "2024-02-27T13:00:00"}),
    ("Unknown pickup zone (264) to Midtown Center, 2.5 mi, Wed 12:00",
     {"trip_distance": 2.5, "pu_location_id": 264, "do_location_id": 161, "passenger_count": 1,
      "pickup_datetime": "2024-02-28T12:00:00"}),
]

# Explanations need every model input observed, so these hours sit at the end of
# the history (29 February) and one hour past it.
DEMAND_ZONES = [132, 161, 230, 138, 237, 79, 48, 1]
DEMAND_HOURS = ["2024-02-29T20:00:00", "2024-03-01T00:00:00"]

# Files the web app reads from disk: (source in the repository, path in the demo).
STATIC_FILES = [
    *[(f"artifacts/responsible/{p}_fairness.json", f"assets/responsible/{p}_fairness.json")
      for p in ("demand", "fare", "fare_booking")],
    *[(f"artifacts/responsible/{p}_shap_summary.png", f"assets/responsible/{p}_shap_summary.png")
      for p in ("demand", "fare", "fare_booking")],
    *[(f"artifacts/drift/{p}_drift_metrics.json", f"assets/drift/{p}_drift_metrics.json")
      for p in ("demand", "fare", "fare_booking")],
    ("data/raw/taxi_zone_lookup.csv", "assets/taxi_zone_lookup.csv"),
]

# The API code the in-browser fare models reuse to build their input rows.
SOURCE_FILES = [
    "src/__init__.py", "src/config.py", "config/config.yaml",
    "src/features/__init__.py", "src/features/calendar.py", "src/features/demand_features.py",
    "src/features/duration.py", "src/features/fare_features.py", "src/features/zones.py",
    "src/serving/__init__.py", "src/serving/features.py", "src/serving/predict.py",
]

# Where the web app expects each file inside the browser's file system.
MOUNTS = {
    "assets/responsible/": "artifacts/responsible/",
    "assets/drift/": "artifacts/drift/",
    "assets/taxi_zone_lookup.csv": "data/raw/taxi_zone_lookup.csv",
    "models/": "demo/models/",
}


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def record() -> dict:
    os.environ.pop("API_KEY", None)  # call the API the way the web app does, locally
    from fastapi.testclient import TestClient

    from src.serving.api import app

    with TestClient(app) as client:
        def call(method: str, path: str, body: dict | None = None) -> dict:
            resp = client.request(method, path, json=body)
            resp.raise_for_status()
            return resp.json()

        health = call("GET", "/health")
        version = call("GET", "/version")
        forecast = call("POST", "/forecast/demand", {"horizon_hours": FORECAST_HOURS})
        fares = [
            {"label": label, "request": req,
             "prediction": call("POST", "/predict/fare", req),
             "explanation": call("POST", "/explain/fare", req)}
            for label, req in FARE_TRIPS
        ]
        demand = [
            {"zone": zone, "target_hour": hour,
             "prediction": call("POST", "/predict/demand", {"zone": zone, "target_hour": hour}),
             "explanation": call("POST", "/explain/demand", {"zone": zone, "target_hour": hour})}
            for zone in DEMAND_ZONES
            for hour in DEMAND_HOURS
        ]

    points = forecast["points"]
    hours = sorted({p["pickup_hour"] for p in points})
    methods = {p["pickup_hour"]: p["method"] for p in points}
    zones: dict[str, list[float]] = {}
    for p in sorted(points, key=lambda p: (p["zone"], p["pickup_hour"])):
        zones.setdefault(str(p["zone"]), []).append(p["predicted_pickups"])
    return {
        "build": {
            "git_commit": _commit(),
            "recorded_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "models": {k: f"{v['name']}/{v['version']}" for k, v in version["models"].items()},
        },
        "health": health,
        "forecast": {
            "hours": hours,
            "methods": [methods[h] for h in hours],
            "model_horizon_hours": forecast["model_horizon_hours"],
            "model_reference": forecast["model_reference"],
            "zones": zones,
        },
        "fare_samples": fares,
        "demand_samples": demand,
    }


def export_fare_models() -> list[str]:
    """Write both fare models as LightGBM model text for the browser."""
    from src.config import get_config
    from src.serving import registry

    config = get_config()
    written = []
    for problem in ("fare", "fare_booking"):
        booster = registry.load_model(problem, config).booster_
        path = DEMO / "models" / f"{problem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(booster.model_to_string(), encoding="utf-8")
        written.append(f"models/{problem}.txt")
    return written


def check_fare_parity(recording: dict) -> None:
    """The browser's models must give exactly the fares the API recorded."""
    from dashboard import demo_models

    demo_models.model.cache_clear()
    os.environ["DEMO_MODELS_DIR"] = str(DEMO / "models")
    for sample in recording["fare_samples"]:
        local = demo_models.predict_fare(sample["request"], recording["build"]["models"])
        api = sample["prediction"]
        for key in ("predicted_fare", "model", "low_confidence", "notes"):
            if local[key] != api[key]:
                raise SystemExit(f"Fare parity failed for {sample['label']!r}: {key} {local[key]} != {api[key]}")
        explained = demo_models.explain_fare(sample["request"])
        if abs(explained["prediction"] - sample["explanation"]["prediction"]) > 1e-3:
            raise SystemExit(f"SHAP parity failed for {sample['label']!r}")


def copy_assets() -> list[str]:
    if ASSETS.exists():
        shutil.rmtree(ASSETS)
    copied = []
    for src, dest in STATIC_FILES:
        target = DEMO / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / src, target)
        copied.append(dest)
    return copied


def _mount_path(site_path: str) -> str:
    for prefix, mounted in MOUNTS.items():
        if site_path.startswith(prefix):
            return mounted + site_path[len(prefix):]
    return site_path


def write_index(assets: list[str]) -> None:
    """The page: stlite runs the web app's own Python files, with demo mode switched on."""
    app_files = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "dashboard").glob("*.py"))
    cards = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "docs" / "model_cards").glob("*.md"))
    files = {path: path for path in app_files + cards + SOURCE_FILES}
    files.update({_mount_path(a): a for a in assets})
    files["demo/recording.json"] = "recording.json"
    mounts = ",\n".join(f'          "{dest}": {{ url: "./{src}" }}' for dest, src in sorted(files.items()))
    html = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>NYC Taxi Forecasting: interactive demo</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.css" />
  </head>
  <body>
    <div id="root"></div>
    <!-- Generated by scripts/build_static_demo.py. The web app's own Streamlit code runs in the
         browser with stlite, and answers come from recording.json, recorded from the real API. -->
    <script type="module">
      import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build/stlite.js";
      mount(
        {{
          requirements: ["pandas", "plotly", "requests", "lightgbm", "holidays", "pyyaml"],
          entrypoint: "streamlit_app.py",
          files: {{
          "streamlit_app.py": `import os, runpy
os.environ["TAXI_DEMO"] = "1"
runpy.run_path("dashboard/app.py", run_name="__main__")
`,
{mounts}
          }},
          streamlitConfig: {{ "client.toolbarMode": "viewer" }},
        }},
        document.getElementById("root"),
      );
    </script>
  </body>
</html>
"""
    (DEMO / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    DEMO.mkdir(exist_ok=True)
    recording = record()
    (DEMO / "recording.json").write_text(json.dumps(recording, separators=(",", ":")), encoding="utf-8")
    models = export_fare_models()
    check_fare_parity(recording)
    write_index(copy_assets() + models)
    size = (DEMO / "recording.json").stat().st_size
    print(
        f"Recorded {len(recording['forecast']['zones'])} zones x {len(recording['forecast']['hours'])} hours, "
        f"{len(recording['fare_samples'])} fare quotes and {len(recording['demand_samples'])} demand "
        f"explanations at commit {recording['build']['git_commit']} ({size / 1e6:.1f} MB); "
        f"fare models exported and matching the API on all {len(recording['fare_samples'])} recorded quotes."
    )


if __name__ == "__main__":
    main()
