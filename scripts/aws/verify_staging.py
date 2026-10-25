"""Run repeatable HTTP smoke checks against a deployed API and dashboard."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True)
class CheckResult:
    name: str
    method: str
    path: str
    status_code: int
    elapsed_ms: float
    passed: bool
    detail: str = ""


def _request(
    session: requests.Session,
    *,
    name: str,
    method: str,
    url: str,
    path: str,
    expected_status: int,
    payload: dict[str, Any] | None = None,
    timeout: float,
) -> CheckResult:
    started = time.perf_counter()
    try:
        response = session.request(method, f"{url.rstrip('/')}{path}", json=payload, timeout=timeout)
        elapsed_ms = (time.perf_counter() - started) * 1000
        detail = ""
        if response.status_code != expected_status:
            detail = response.text[:300]
        return CheckResult(
            name=name,
            method=method,
            path=path,
            status_code=response.status_code,
            elapsed_ms=round(elapsed_ms, 2),
            passed=response.status_code == expected_status,
            detail=detail,
        )
    except requests.RequestException as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        return CheckResult(
            name=name,
            method=method,
            path=path,
            status_code=0,
            elapsed_ms=round(elapsed_ms, 2),
            passed=False,
            detail=str(exc),
        )


def run_checks(
    api_url: str,
    dashboard_url: str | None = None,
    *,
    timeout: float = 30.0,
    session: requests.Session | None = None,
) -> list[CheckResult]:
    """Exercise the public API contract and optional dashboard landing page."""
    http = session or requests.Session()
    fare = {
        "trip_distance": 3.4,
        "pu_location_id": 132,
        "do_location_id": 230,
        "passenger_count": 1,
        "pickup_datetime": "2024-01-15T18:30:00",
    }
    demand = {"zone": 132, "target_hour": "2024-01-31T20:00:00"}
    checks = [
        ("health", "GET", "/health", 200, None),
        ("fare prediction", "POST", "/predict/fare", 200, fare),
        ("demand prediction", "POST", "/predict/demand", 200, demand),
        ("demand forecast", "POST", "/forecast/demand", 200, {"horizon_hours": 2, "zones": [132]}),
        ("fare explanation", "POST", "/explain/fare", 200, fare),
        ("demand explanation", "POST", "/explain/demand", 200, demand),
        ("metrics", "GET", "/metrics", 200, None),
        ("invalid fare rejected", "POST", "/predict/fare", 422, {**fare, "trip_distance": -1}),
        ("unknown zone rejected", "POST", "/predict/demand", 400, {**demand, "zone": 99999}),
    ]
    results = [
        _request(
            http,
            name=name,
            method=method,
            url=api_url,
            path=path,
            expected_status=expected,
            payload=payload,
            timeout=timeout,
        )
        for name, method, path, expected, payload in checks
    ]
    if dashboard_url:
        results.append(
            _request(
                http,
                name="dashboard landing page",
                method="GET",
                url=dashboard_url,
                path="/",
                expected_status=200,
                timeout=timeout,
            )
        )
    return results


def report(results: list[CheckResult], api_url: str, dashboard_url: str | None) -> dict[str, Any]:
    latencies = [result.elapsed_ms for result in results]
    return {
        "api_url": api_url,
        "dashboard_url": dashboard_url,
        "passed": all(result.passed for result in results),
        "checks_passed": sum(result.passed for result in results),
        "checks_total": len(results),
        "latency_ms": {
            "median": round(statistics.median(latencies), 2),
            "maximum": round(max(latencies), 2),
        },
        "checks": [asdict(result) for result in results],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", required=True, help="Deployed API base URL.")
    parser.add_argument("--dashboard-url", help="Optional deployed Streamlit base URL.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds.")
    parser.add_argument("--output", type=Path, help="Optional path for the JSON report.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = run_checks(args.api_url, args.dashboard_url, timeout=args.timeout)
    payload = report(results, args.api_url, args.dashboard_url)
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(f"{rendered}\n", encoding="utf-8")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
