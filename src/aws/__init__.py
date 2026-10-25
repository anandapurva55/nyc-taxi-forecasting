"""AWS helper utilities (lazily-imported boto3).

These helpers keep the AWS SDK an optional, job-only dependency: ``boto3`` is
imported inside the functions that need it (mirroring :mod:`src.monitoring.capture`).
Local code and the lean serving image therefore never require boto3.
"""

from __future__ import annotations

from .s3 import (
    build_s3_uri,
    download_dir,
    parse_s3_uri,
    upload_dir,
    upload_file,
)

__all__ = [
    "build_s3_uri",
    "download_dir",
    "parse_s3_uri",
    "upload_dir",
    "upload_file",
]
