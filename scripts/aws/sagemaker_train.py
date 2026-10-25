"""CLI to submit model training as a SageMaker job.

Runs the same ``scripts/train_models.py`` on managed cloud compute. Requires AWS
credentials, ``requirements-monitoring.txt`` (sagemaker + boto3), a SageMaker
execution role (Terraform output ``sagemaker_role_arn``), and a training image in
ECR that bakes in this repo's code + deps.

Usage::

    python -m scripts.aws.sagemaker_train \
        --role-arn <sagemaker_role_arn> \
        --image-uri <ecr>/taxi-training:latest \
        --problem demand --n-trials 10

Point runs at a shared store first (so the cloud job logs where you can see it)::

    $env:MLFLOW_TRACKING_URI="http://<mlflow-server>:5000"
    $env:MLFLOW_ARTIFACT_ROOT="s3://<bucket>/mlflow"
"""

from __future__ import annotations

import argparse
import logging

from src.aws.sagemaker import DEFAULT_INSTANCE_TYPE, submit_training_job

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Submit training to SageMaker.")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--image-uri", required=True, help="Training image URI (ECR).")
    parser.add_argument("--problem", choices=["demand", "fare", "both"], default="both")
    parser.add_argument("--n-trials", type=int, default=None, help="Optuna trials per estimator.")
    parser.add_argument(
        "--no-register", action="store_true", help="Skip the MLflow Model Registry."
    )
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument(
        "--no-wait", action="store_true", help="Return immediately instead of blocking."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    job_name = submit_training_job(
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        problem=args.problem,
        n_trials=args.n_trials,
        register=not args.no_register,
        region=args.region,
        instance_type=args.instance_type,
        wait=not args.no_wait,
    )
    print(job_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
