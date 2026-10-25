"""Parse captured inference records into analysis-ready DataFrames.

The serving API emits one JSON line per prediction (``event="prediction"``):

    {"event": "prediction", "timestamp": "...", "request_id": "...",
     "problem": "fare", "model_version": "2",
     "features": {"trip_distance": 3.1, ...}, "prediction": 14.2}

Kinesis Firehose ships these lines to S3. This module turns a stream of such
lines (from a local file, an iterable, or S3) into a flat DataFrame whose
columns are the model input features plus ``prediction`` - exactly the shape
``compute_drift`` expects. boto3/S3 access is imported lazily so local parsing
needs no AWS dependencies.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

PREDICTION_EVENT = "prediction"


def _capture_lines_from_bytes(payload: bytes) -> list[str]:
    """Extract log messages from raw, GZIP, or CloudWatch Logs payloads."""
    while payload.startswith(b"\x1f\x8b"):
        payload = gzip.decompress(payload)

    text = payload.decode("utf-8", errors="replace")
    decoder = json.JSONDecoder()
    envelopes: list[object] = []
    position = 0
    try:
        while position < len(text):
            while position < len(text) and text[position].isspace():
                position += 1
            if position < len(text):
                envelope, position = decoder.raw_decode(text, position)
                envelopes.append(envelope)
    except json.JSONDecodeError:
        return text.splitlines()

    lines: list[str] = []
    for envelope in envelopes:
        if not isinstance(envelope, dict) or envelope.get("messageType") not in {
            "DATA_MESSAGE",
            "CONTROL_MESSAGE",
        }:
            lines.append(json.dumps(envelope))
            continue
        lines.extend(
            event["message"]
            for event in envelope.get("logEvents", [])
            if isinstance(event, dict) and isinstance(event.get("message"), str)
        )
    return lines


def _records_from_lines(lines: Iterable[str], problem: str | None) -> list[dict]:
    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("event") != PREDICTION_EVENT:
            continue
        if problem is not None and obj.get("problem") != problem:
            continue
        flat: dict = {}
        features = obj.get("features") or {}
        if isinstance(features, dict):
            flat.update(features)
        for key in ("prediction", "model_version", "problem", "timestamp", "request_id"):
            if key in obj:
                flat[key] = obj[key]
        records.append(flat)
    return records


def parse_capture_lines(lines: Iterable[str], problem: str | None = None) -> pd.DataFrame:
    """Parse an iterable of JSON log lines into a DataFrame of captured predictions."""
    return pd.DataFrame(_records_from_lines(lines, problem))


def load_capture_file(path: str | Path, problem: str | None = None) -> pd.DataFrame:
    """Load captured predictions from a local newline-delimited JSON file."""
    with open(path, encoding="utf-8") as fh:
        return parse_capture_lines(fh, problem=problem)


def load_capture_s3(bucket: str, prefix: str, problem: str | None = None):  # pragma: no cover
    """Load and concatenate all captured records under an S3 prefix.

    boto3 is an AWS-job-only dependency (see ``requirements-monitoring.txt``) and
    is imported lazily so local parsing stays dependency-free.
    """
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is not installed; add requirements-monitoring.txt in the job image."
        ) from exc

    s3 = boto3.client("s3")
    frames: list[pd.DataFrame] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read()
            lines = _capture_lines_from_bytes(body)
            frames.append(parse_capture_lines(lines, problem=problem))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
