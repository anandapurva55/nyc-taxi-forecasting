"""Upload the data pipeline outputs to an S3 data lake.

After ``scripts.run_pipeline`` produces local ``raw/``,
``processed/`` and ``features/`` artifacts, this pushes them to an S3 bucket laid
out as a small data lake::

    s3://<bucket>/<prefix>/raw/...
    s3://<bucket>/<prefix>/processed/...
    s3://<bucket>/<prefix>/features/...

The bucket is created by ``infra/terraform/datalake.tf`` (Terraform output
``datalake_bucket``). Requires AWS credentials and ``boto3`` (installed via
``requirements-monitoring.txt`` in the job image, or locally for the demo).

Usage::

    python -m scripts.aws.upload_data_lake --bucket my-taxi-datalake
    python -m scripts.aws.upload_data_lake --bucket my-bucket --prefix taxi --region us-east-1
    python -m scripts.aws.upload_data_lake --bucket my-bucket --stages raw processed
"""

from __future__ import annotations

import argparse
import logging

from src.aws import build_s3_uri, upload_dir
from src.config import get_config

logger = logging.getLogger(__name__)

STAGE_DIRS = ("raw", "processed", "features")


def run(
    bucket: str,
    prefix: str,
    region: str | None,
    stages: tuple[str, ...],
) -> dict[str, int]:
    """Upload the requested pipeline stages to S3; return a per-stage file count."""
    config = get_config()
    stage_paths = {
        "raw": config.raw_dir,
        "processed": config.processed_dir,
        "features": config.features_dir,
    }

    counts: dict[str, int] = {}
    for stage in stages:
        local_dir = stage_paths[stage]
        if not local_dir.is_dir():
            logger.warning("Skipping '%s': %s does not exist", stage, local_dir)
            counts[stage] = 0
            continue
        key_prefix = f"{prefix.strip('/')}/{stage}" if prefix.strip("/") else stage
        uris = upload_dir(local_dir, bucket, key_prefix, region=region)
        counts[stage] = len(uris)

    total = sum(counts.values())
    logger.info(
        "Uploaded %d files to %s (per stage: %s)",
        total,
        build_s3_uri(bucket, prefix),
        counts,
    )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Upload data to an S3 data lake.")
    parser.add_argument("--bucket", required=True, help="Target S3 bucket (datalake_bucket output).")
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional key prefix under the bucket (e.g. 'taxi'). Default: bucket root.",
    )
    parser.add_argument("--region", default=None, help="AWS region (default: environment).")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGE_DIRS,
        default=list(STAGE_DIRS),
        help="Which pipeline stages to upload. Default: all.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(args.bucket, args.prefix, args.region, tuple(args.stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
