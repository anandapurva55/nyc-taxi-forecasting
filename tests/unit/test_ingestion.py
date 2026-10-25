"""Offline tests for TLC data ingestion and cache behavior."""

from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data import ingestion


class FakeResponse:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.status_checked = False

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        self.status_checked = True

    def iter_content(self, *, chunk_size: int):
        assert chunk_size == ingestion._CHUNK
        return iter(self.chunks)


def _config(tmp_path) -> Config:
    return Config(
        {
            "data": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "features_dir": str(tmp_path / "features"),
                "base_url": "https://example.test/trips/",
                "zone_lookup_url": "https://example.test/zones.csv",
                "tlc_year": 2024,
                "tlc_month": 2,
            }
        }
    )


def test_download_streams_atomically_and_reuses_cache(tmp_path, monkeypatch):
    response = FakeResponse([b"first", b"", b"-second"])
    calls = []

    def fake_get(url, *, stream, timeout):
        calls.append((url, stream, timeout))
        return response

    monkeypatch.setattr(ingestion.requests, "get", fake_get)
    destination = tmp_path / "nested" / "sample.bin"

    assert ingestion._download("https://example.test/sample", destination) == destination
    assert destination.read_bytes() == b"first-second"
    assert response.status_checked
    assert not destination.with_suffix(".bin.part").exists()
    assert ingestion._download("https://example.test/sample", destination) == destination
    assert calls == [("https://example.test/sample", True, 120)]


def test_download_helpers_build_expected_urls_and_paths(tmp_path, monkeypatch):
    config = _config(tmp_path)
    calls = []

    def fake_download(url, destination, *, overwrite=False):
        calls.append((url, destination, overwrite))
        return destination

    monkeypatch.setattr(ingestion, "_download", fake_download)

    trips = ingestion.download_trips(config, overwrite=True)
    zones = ingestion.download_zone_lookup(config)

    assert ingestion.trip_filename(2024, 2) == "yellow_tripdata_2024-02.parquet"
    assert trips == config.raw_dir / "yellow_tripdata_2024-02.parquet"
    assert zones == config.raw_dir / "taxi_zone_lookup.csv"
    assert calls == [
        ("https://example.test/trips/yellow_tripdata_2024-02.parquet", trips, True),
        ("https://example.test/zones.csv", zones, False),
    ]


def test_local_loaders_round_trip_dataframes(tmp_path):
    trips_path = tmp_path / "trips.parquet"
    zones_path = tmp_path / "zones.csv"
    trips = pd.DataFrame({"trip_distance": [1.2, 3.4]})
    zones = pd.DataFrame({"LocationID": [1], "Zone": ["Test"]})
    trips.to_parquet(trips_path, index=False)
    zones.to_csv(zones_path, index=False)

    pd.testing.assert_frame_equal(ingestion.load_trips(trips_path), trips)
    pd.testing.assert_frame_equal(ingestion.load_zone_lookup(zones_path), zones)


def test_ingest_ensures_directories_and_returns_both_paths(tmp_path, monkeypatch):
    config = _config(tmp_path)
    trips_path = config.raw_dir / "trips.parquet"
    zones_path = config.raw_dir / "zones.csv"
    monkeypatch.setattr(ingestion, "download_trips", lambda *_args, **_kwargs: trips_path)
    monkeypatch.setattr(ingestion, "download_zone_lookup", lambda *_args, **_kwargs: zones_path)

    result = ingestion.ingest(config, year=2025, month=3, overwrite=True)

    assert result == {"trips": trips_path, "zones": zones_path}
    assert config.raw_dir.is_dir()
    assert config.processed_dir.is_dir()
    assert config.features_dir.is_dir()
