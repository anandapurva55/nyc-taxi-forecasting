"""Publish a gated registry and trigger the privileged CodeBuild deployment."""

from __future__ import annotations

import argparse
import time
from pathlib import Path


def upload_registry(model_dir: Path, bucket: str, s3_client) -> int:
    """Upload the gated MLflow database and artifact tree for CodeBuild."""
    database = model_dir / "mlflow.db"
    runs = model_dir / "mlruns"
    if not database.exists() or not runs.is_dir():
        raise FileNotFoundError(f"Model output must contain mlflow.db and mlruns/: {model_dir}")

    s3_client.upload_file(str(database), bucket, "mlflow.db")
    uploaded = 1
    for path in runs.rglob("*"):
        if path.is_file():
            key = f"mlruns/{path.relative_to(runs).as_posix()}"
            s3_client.upload_file(str(path), bucket, key)
            uploaded += 1
    return uploaded


def start_deployment_build(project: str, release_id: str, codebuild_client) -> str:
    response = codebuild_client.start_build(
        projectName=project,
        environmentVariablesOverride=[
            {"name": "IMAGE_TAG", "value": release_id, "type": "PLAINTEXT"}
        ],
    )
    return response["build"]["id"]


def wait_for_build(build_id: str, codebuild_client, poll_seconds: int = 15) -> str:
    terminal = {"SUCCEEDED", "FAILED", "FAULT", "STOPPED", "TIMED_OUT"}
    while True:
        builds = codebuild_client.batch_get_builds(ids=[build_id])["builds"]
        if not builds:
            raise RuntimeError(f"CodeBuild did not return build {build_id}")
        status = builds[0]["buildStatus"]
        if status in terminal:
            return status
        time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    import boto3

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--artifacts-bucket", required=True)
    parser.add_argument("--codebuild-project", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--region", default="us-east-1")
    args = parser.parse_args(argv)

    session = boto3.Session(region_name=args.region)
    uploaded = upload_registry(args.model_dir, args.artifacts_bucket, session.client("s3"))
    build_id = start_deployment_build(
        args.codebuild_project, args.release_id, session.client("codebuild")
    )
    status = wait_for_build(build_id, session.client("codebuild"))
    if status != "SUCCEEDED":
        raise RuntimeError(f"CodeBuild deployment {build_id} ended with {status}")
    print(f"Deployed release {args.release_id}; uploaded {uploaded} registry files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
