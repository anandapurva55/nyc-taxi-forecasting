"""Unit tests for the S3 data-lake helpers (offline, no boto3)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.aws.s3 import (
    build_s3_uri,
    download_dir,
    parse_s3_uri,
    upload_dir,
    upload_file,
)


class FakeS3Client:
    """In-memory stand-in for boto3's S3 client used by the transfer helpers."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename: str, bucket: str, key: str) -> None:
        self.objects[f"{bucket}/{key}"] = Path(filename).read_bytes()

    def download_file(self, bucket: str, key: str, filename: str) -> None:
        Path(filename).write_bytes(self.objects[f"{bucket}/{key}"])

    def get_paginator(self, _name: str):
        client = self

        class _Paginator:
            def paginate(self, Bucket: str, Prefix: str = ""):  # noqa: N803
                contents = [
                    {"Key": key.split("/", 1)[1]}
                    for key in client.objects
                    if key.startswith(f"{Bucket}/")
                    and key.split("/", 1)[1].startswith(Prefix)
                ]
                yield {"Contents": contents}

        return _Paginator()


# --- URI helpers -------------------------------------------------------------
def test_build_s3_uri_joins_and_normalises():
    assert build_s3_uri("bkt", "a", "b/c") == "s3://bkt/a/b/c"
    assert build_s3_uri("bkt", "/a/", "/b/") == "s3://bkt/a/b"
    assert build_s3_uri("bkt") == "s3://bkt"


def test_build_s3_uri_requires_bucket():
    with pytest.raises(ValueError):
        build_s3_uri("")


def test_parse_s3_uri_roundtrip():
    assert parse_s3_uri("s3://bkt/a/b/c") == ("bkt", "a/b/c")
    assert parse_s3_uri("s3://bkt/") == ("bkt", "")
    assert parse_s3_uri("s3://bkt") == ("bkt", "")


def test_parse_s3_uri_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_s3_uri("https://example.com/x")
    with pytest.raises(ValueError):
        parse_s3_uri("s3:///no-bucket")


# --- Transfer helpers (FakeS3Client) -----------------------------------------
def test_upload_file(tmp_path: Path):
    src = tmp_path / "f.txt"
    src.write_text("hello")
    client = FakeS3Client()
    uri = upload_file(src, "bkt", "prefix/f.txt", client=client)
    assert uri == "s3://bkt/prefix/f.txt"
    assert client.objects["bkt/prefix/f.txt"] == b"hello"


def test_upload_file_missing(tmp_path: Path):
    client = FakeS3Client()
    with pytest.raises(FileNotFoundError):
        upload_file(tmp_path / "nope.txt", "bkt", "k", client=client)


def test_upload_dir_preserves_tree(tmp_path: Path):
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a")
    (root / "sub" / "b.txt").write_text("b")

    client = FakeS3Client()
    uris = upload_dir(root, "bkt", "lake/raw", client=client)

    assert set(uris) == {"s3://bkt/lake/raw/a.txt", "s3://bkt/lake/raw/sub/b.txt"}
    assert client.objects["bkt/lake/raw/a.txt"] == b"a"
    assert client.objects["bkt/lake/raw/sub/b.txt"] == b"b"


def test_upload_dir_empty_prefix(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "a.txt").write_text("a")
    client = FakeS3Client()
    uris = upload_dir(root, "bkt", "", client=client)
    assert uris == ["s3://bkt/a.txt"]


def test_upload_dir_requires_directory(tmp_path: Path):
    client = FakeS3Client()
    with pytest.raises(NotADirectoryError):
        upload_dir(tmp_path / "missing", "bkt", "p", client=client)


def test_download_dir_roundtrip(tmp_path: Path):
    root = tmp_path / "data"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("a")
    (root / "sub" / "b.txt").write_text("b")

    client = FakeS3Client()
    upload_dir(root, "bkt", "lake", client=client)

    dest = tmp_path / "restored"
    written = download_dir("bkt", "lake", dest, client=client)

    assert (dest / "a.txt").read_text() == "a"
    assert (dest / "sub" / "b.txt").read_text() == "b"
    assert len(written) == 2

    shutil.rmtree(dest)
