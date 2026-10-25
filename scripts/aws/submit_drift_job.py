"""CLI to submit the feature-drift detection as a SageMaker Processing job.

Runs ``scripts/monitoring/run_drift_job.py`` on managed cloud compute - the same
entrypoint the weekly EventBridge schedule invokes. Use it to produce an
on-demand Processing job (e.g. before a console walkthrough). Requires AWS
credentials, ``requirements-monitoring.txt`` (sagemaker + boto3), the SageMaker
execution role (Terraform output ``sagemaker_role_arn``), and a monitoring image
in ECR that bakes in this repo's code + deps.

Usage::

    python -m scripts.aws.submit_drift_job \
        --role-arn <sagemaker_role_arn> \
        --image-uri <ecr>/taxi-monitoring:latest \
        --problem fare \
        --current-s3 s3://<monitoring_bucket>/captured/ \
        --output-s3-uri s3://<monitoring_bucket>/drift-reports/
"""

from __future__ import annotations

import argparse
import logging

from src.aws.sagemaker import DEFAULT_DRIFT_INSTANCE_TYPE, submit_drift_job

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit drift detection to SageMaker.")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--image-uri", required=True, help="Monitoring image URI (ECR).")
    parser.add_argument("--problem", choices=["demand", "fare"], default="fare")
    parser.add_argument(
        "--current-s3", default=None, help="s3://bucket/prefix of captured records."
    )
    parser.add_argument(
        "--output-s3-uri", default=None, help="s3://bucket/prefix for the drift report."
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default=DEFAULT_DRIFT_INSTANCE_TYPE)
    parser.add_argument(
        "--no-wait", action="store_true", help="Return immediately instead of blocking."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    job_name = submit_drift_job(
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        problem=args.problem,
        current_s3=args.current_s3,
        output_s3_uri=args.output_s3_uri,
        region=args.region,
        instance_type=args.instance_type,
        wait=not args.no_wait,
    )
    print(job_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
