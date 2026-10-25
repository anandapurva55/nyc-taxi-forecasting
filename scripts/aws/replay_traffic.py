"""Replay held-out February trips through the live API to feed drift monitoring.

Each request is captured by the API (CloudWatch Logs -> Firehose -> S3), so the
SageMaker drift job then compares realistic production-like inputs against the
training distribution.

    # realistic traffic: a random sample of real test-split trips
    python -m scripts.aws.replay_traffic --api-url "$API_URL" --rows 1000

    # shifted traffic: the same trips with distances multiplied by 3
    python -m scripts.aws.replay_traffic --api-url "$API_URL" --rows 1000 --distance-factor 3
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
import requests

from src.config import get_config

_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
    return _local.session


def fare_payloads(trips: pd.DataFrame, distance_factor: float) -> list[dict[str, Any]]:
    """Build /predict/fare bodies from fare split rows."""
    return [
        {
            "trip_distance": round(min(float(row.trip_distance) * distance_factor, 100.0), 2),
            "pu_location_id": int(row.PULocationID),
            "do_location_id": int(row.DOLocationID),
            "passenger_count": int(row.passenger_count),
            "pickup_datetime": pd.Timestamp(row.tpep_pickup_datetime).isoformat(),
            "trip_duration_min": round(float(row.trip_duration_min), 2),
        }
        for row in trips.itertuples(index=False)
        if row.trip_distance > 0
    ]


def demand_payloads(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Build /predict/demand bodies from demand split rows."""
    return [
        {"zone": int(row.PULocationID), "target_hour": pd.Timestamp(row.pickup_hour).isoformat()}
        for row in rows.itertuples(index=False)
    ]


def _post(url: str, body: dict[str, Any], timeout: float) -> bool:
    try:
        return _session().post(url, json=body, timeout=timeout).ok
    except requests.RequestException:
        return False


def replay(api_url: str, rows: int, distance_factor: float, seed: int, concurrency: int) -> dict:
    features_dir = get_config().features_dir
    fare = pd.read_parquet(features_dir / "fare_test.parquet").sample(rows, random_state=seed)
    demand = pd.read_parquet(features_dir / "demand_test.parquet").sample(rows, random_state=seed)

    jobs = [(f"{api_url}/predict/fare", body) for body in fare_payloads(fare, distance_factor)]
    jobs += [(f"{api_url}/predict/demand", body) for body in demand_payloads(demand)]

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda job: _post(job[0], job[1], 60.0), jobs))

    return {
        "fare_requests": sum(1 for url, _ in jobs if url.endswith("/fare")),
        "demand_requests": sum(1 for url, _ in jobs if url.endswith("/demand")),
        "succeeded": sum(results),
        "failed": len(results) - sum(results),
        "distance_factor": distance_factor,
        "seed": seed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--rows", type=int, default=1000, help="Trips and zone-hours to send.")
    parser.add_argument("--distance-factor", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--concurrency", type=int, default=10)
    args = parser.parse_args()

    summary = replay(args.api_url.rstrip("/"), args.rows, args.distance_factor, args.seed, args.concurrency)
    print(json.dumps(summary, indent=2))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
