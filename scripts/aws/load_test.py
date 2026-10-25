"""Run a bounded concurrent load probe against the deployed API and dashboard."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class Probe:
    name: str
    method: str
    url: str
    payload: dict[str, Any] | None = None
    p95_target_ms: float | None = None


@dataclass(frozen=True)
class Sample:
    name: str
    status_code: int
    elapsed_ms: float
    passed: bool


def _percentile(values: list[float], percentile: float) -> float:
    """Return the nearest-rank percentile for a non-empty sample."""
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _request(probe: Probe, timeout: float) -> Sample:
    started = time.perf_counter()
    try:
        response = requests.request(
            probe.method, probe.url, json=probe.payload, timeout=timeout
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return Sample(probe.name, response.status_code, round(elapsed_ms, 2), response.ok)
    except requests.RequestException:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return Sample(probe.name, 0, round(elapsed_ms, 2), False)


def _summarize(
    samples: list[Sample], p95_target_ms: float | None = None
) -> dict[str, Any]:
    latencies = [sample.elapsed_ms for sample in samples]
    passed = sum(sample.passed for sample in samples)
    p95 = round(_percentile(latencies, 0.95), 2)
    return {
        "requests": len(samples),
        "passed": passed,
        "failed": len(samples) - passed,
        "error_rate": round((len(samples) - passed) / len(samples), 4),
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 2),
            "p95": p95,
            "maximum": round(max(latencies), 2),
        },
        "p95_target_ms": p95_target_ms,
        "latency_target_met": (
            None if p95_target_ms is None else p95 <= p95_target_ms
        ),
    }


def run(
    probes: list[Probe], *, requests_per_probe: int, concurrency: int, timeout: float
) -> dict[str, Any]:
    """Warm each endpoint once, then execute a bounded concurrent probe."""
    for probe in probes:
        _request(probe, timeout)

    scheduled = [probe for probe in probes for _ in range(requests_per_probe)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_request, probe, timeout) for probe in scheduled]
        samples = [future.result() for future in as_completed(futures)]
    duration_seconds = time.perf_counter() - started

    endpoint_results = {}
    for probe in probes:
        endpoint_samples = [sample for sample in samples if sample.name == probe.name]
        endpoint_results[probe.name] = _summarize(
            endpoint_samples, probe.p95_target_ms
        )
    return {
        "requests": len(samples),
        "concurrency": concurrency,
        "duration_seconds": round(duration_seconds, 2),
        "requests_per_second": round(len(samples) / duration_seconds, 2),
        "passed": all(sample.passed for sample in samples)
        and all(
            result["latency_target_met"] is not False
            for result in endpoint_results.values()
        ),
        "endpoints": endpoint_results,
        "samples": [asdict(sample) for sample in samples],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True)
    parser.add_argument("--dashboard-url")
    parser.add_argument("--requests-per-endpoint", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--prediction-p95-target-ms", type=float, default=500.0)
    parser.add_argument("--dashboard-p95-target-ms", type=float, default=3000.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_url = args.api_url.rstrip("/")
    probes = [
        Probe("health", "GET", f"{api_url}/health"),
        Probe(
            "fare prediction",
            "POST",
            f"{api_url}/predict/fare",
            {
                "trip_distance": 3.4,
                "pu_location_id": 132,
                "do_location_id": 230,
                "passenger_count": 1,
                "pickup_datetime": "2024-01-15T18:30:00",
            },
            args.prediction_p95_target_ms,
        ),
        Probe(
            "demand prediction",
            "POST",
            f"{api_url}/predict/demand",
            {"zone": 132, "target_hour": "2024-01-31T20:00:00"},
            args.prediction_p95_target_ms,
        ),
    ]
    if args.dashboard_url:
        probes.append(
            Probe(
                "dashboard",
                "GET",
                args.dashboard_url.rstrip("/") + "/",
                p95_target_ms=args.dashboard_p95_target_ms,
            )
        )

    payload = run(
        probes,
        requests_per_probe=args.requests_per_endpoint,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
