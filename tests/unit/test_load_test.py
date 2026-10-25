"""Unit tests for the bounded staging load probe."""

from __future__ import annotations

from scripts.aws import load_test


def test_summarize_evaluates_optional_p95_target():
    samples = [
        load_test.Sample("fare", 200, 100.0, True),
        load_test.Sample("fare", 200, 450.0, True),
    ]

    measured = load_test._summarize(samples, p95_target_ms=500.0)
    availability_only = load_test._summarize(samples)

    assert measured["latency_ms"]["p95"] == 450.0
    assert measured["latency_target_met"] is True
    assert availability_only["latency_target_met"] is None


def test_run_fails_when_endpoint_misses_latency_target(monkeypatch):
    def fake_request(probe, timeout):
        del timeout
        elapsed = 600.0 if probe.name == "fare" else 700.0
        return load_test.Sample(probe.name, 200, elapsed, True)

    monkeypatch.setattr(load_test, "_request", fake_request)
    probes = [
        load_test.Probe("health", "GET", "http://example/health"),
        load_test.Probe(
            "fare", "POST", "http://example/predict/fare", {}, p95_target_ms=500.0
        ),
    ]

    result = load_test.run(probes, requests_per_probe=2, concurrency=1, timeout=1.0)

    assert result["endpoints"]["health"]["latency_target_met"] is None
    assert result["endpoints"]["fare"]["latency_target_met"] is False
    assert result["passed"] is False
