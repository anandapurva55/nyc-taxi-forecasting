"""S3 helpers for the data lake and cloud artifacts.

Pure-Python URI helpers (``build_s3_uri`` / ``parse_s3_uri``) are dependency-free
and unit-testable offline. The transfer helpers (``upload_file`` / ``upload_dir``
/ ``download_dir``) import ``boto3`` lazily so that importing this module never
requires the AWS SDK.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

S3_SCHEME = "s3://"


def build_s3_uri(bucket: str, *key_parts: str) -> str:
    """Join a bucket and key parts into a normalised ``s3://bucket/key`` URI."""
    if not bucket:
        raise ValueError("bucket must be non-empty")
    cleaned: list[str] = []
    for part in key_parts:
        if part is None:
            continue
        cleaned.extend(seg for seg in str(part).strip("/").split("/") if seg)
    key = "/".join(cleaned)
    return f"{S3_SCHEME}{bucket}/{key}" if key else f"{S3_SCHEME}{bucket}"


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into ``(bucket, key)``.

    ``key`` is the (possibly empty) path after the bucket, without a leading slash.
    """
    if not uri.startswith(S3_SCHEME):
        raise ValueError(f"not an s3 uri: {uri!r}")
    remainder = uri[len(S3_SCHEME) :]
    bucket, _, key = remainder.partition("/")
    if not bucket:
        raise ValueError(f"s3 uri missing bucket: {uri!r}")
    return bucket, key.strip("/")


def _client(region: str | None = None):  # pragma: no cover - thin boto3 wrapper
    try:
        import boto3
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "boto3 is not installed; add requirements-monitoring.txt in the job image."
        ) from exc
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")


def upload_file(
    path: str | Path,
    bucket: str,
    key: str,
    *,
    region: str | None = None,
    client=None,
) -> str:
    """Upload a single local file to ``s3://bucket/key`` and return the URI."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"not a file: {path}")
    s3 = client or _client(region)
    s3.upload_file(str(path), bucket, key)
    uri = build_s3_uri(bucket, key)
    logger.info("Uploaded %s -> %s", path, uri)
    return uri


def _iter_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file():
            yield p


def upload_dir(
    local_dir: str | Path,
    bucket: str,
    prefix: str,
    *,
    region: str | None = None,
    client=None,
) -> list[str]:
    """Recursively upload ``local_dir`` under ``s3://bucket/prefix``.

    Returns the list of uploaded object URIs. The directory tree is preserved
    relative to ``local_dir``.
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"not a directory: {local_dir}")
    s3 = client or _client(region)
    base = prefix.strip("/")
    uris: list[str] = []
    for file_path in _iter_files(local_dir):
        rel = file_path.relative_to(local_dir).as_posix()
        key = f"{base}/{rel}" if base else rel
        s3.upload_file(str(file_path), bucket, key)
        uris.append(build_s3_uri(bucket, key))
    logger.info("Uploaded %d files from %s to s3://%s/%s", len(uris), local_dir, bucket, base)
    return uris


def download_dir(
    bucket: str,
    prefix: str,
    local_dir: str | Path,
    *,
    region: str | None = None,
    client=None,
) -> list[Path]:
    """Download every object under ``s3://bucket/prefix`` into ``local_dir``.

    The key structure below ``prefix`` is recreated under ``local_dir``. Returns
    the list of written local paths.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    s3 = client or _client(region)
    base = prefix.strip("/")
    written: list[Path] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=base):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):
                continue
            rel = key[len(base) :].lstrip("/") if base else key
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(dest))
            written.append(dest)
    logger.info("Downloaded %d files from s3://%s/%s to %s", len(written), bucket, base, local_dir)
    return written
