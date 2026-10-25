"""Tests for the credential-free staging smoke checker."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from scripts.aws.verify_staging import report, run_checks


@dataclass
class FakeResponse:
    status_code: int
    text: str = ""


class FakeSession:
    def __init__(self, statuses: dict[tuple[str, str], int] | None = None):
        self.statuses = statuses or {}
        self.calls: list[tuple[str, str, dict | None]] = []

    def request(self, method, url, json=None, timeout=None):
        self.calls.append((method, url, json))
        path = requests.utils.urlparse(url).path
        default = 422 if path == "/predict/fare" and json and json.get("trip_distance") == -1 else 200
        if path == "/predict/demand" and json and json.get("zone") == 99999:
            default = 400
        status = self.statuses.get((method, path), default)
        return FakeResponse(status_code=status, text="failure")


def test_run_checks_exercises_full_contract():
    session = FakeSession()

    results = run_checks("https://api.example", "https://dashboard.example", session=session)

    assert len(results) == 10
    assert all(result.passed for result in results)
    assert any(call[1] == "https://api.example/forecast/demand" for call in session.calls)
    assert session.calls[-1][1] == "https://dashboard.example/"


def test_report_fails_when_a_check_has_unexpected_status():
    session = FakeSession({("GET", "/health"): 503})

    results = run_checks("https://api.example", session=session)
    payload = report(results, "https://api.example", None)

    assert payload["passed"] is False
    assert payload["checks_passed"] == payload["checks_total"] - 1
    health = next(check for check in payload["checks"] if check["name"] == "health")
    assert health["status_code"] == 503
    assert health["detail"] == "failure"
