"""Submit the training job to SageMaker.

Run the *same* ``scripts/train_models.py`` on managed cloud
compute instead of a laptop. A SageMaker ``ScriptProcessor`` runs the training
entrypoint inside the project's training image; runs and models are logged to the
configured MLflow store (point ``MLFLOW_TRACKING_URI`` at a remote server and/or
set ``MLFLOW_ARTIFACT_ROOT`` to an S3 URI so artifacts are cloud-backed).

Pure helpers (``build_training_env`` / ``training_job_arguments``) are
dependency-free and unit-testable; ``submit_training_job`` imports ``sagemaker``
lazily so importing this module never requires the SDK.
"""

from __future__ import annotations

import logging

from src.config import Config, get_config

logger = logging.getLogger(__name__)

DEFAULT_INSTANCE_TYPE = "ml.m5.large"
DEFAULT_BASE_JOB_NAME = "taxi-forecasting-train"
TRAIN_ENTRYPOINT = "scripts/train_models.py"

DEFAULT_DRIFT_INSTANCE_TYPE = "ml.t3.medium"
DEFAULT_DRIFT_BASE_JOB_NAME = "taxi-forecasting-drift"
DRIFT_ENTRYPOINT = "scripts/monitoring/run_drift_job.py"
DRIFT_OUTPUT_DIR = "/opt/ml/processing/output"
DRIFT_NAMESPACE = "TaxiForecasting/Drift"


def build_training_env(config: Config | None = None) -> dict[str, str]:
    """Environment variables passed to the job so it logs to the shared MLflow store.

    Only non-empty values are included. This lets the cloud job write runs/models
    to the same tracking backend and (optional) S3 artifact root as local runs.
    """
    config = config or get_config()
    mlflow_cfg = config.mlflow
    candidates = {
        "MLFLOW_TRACKING_URI": mlflow_cfg.get("tracking_uri"),
        "MLFLOW_ARTIFACT_ROOT": mlflow_cfg.get("artifact_root"),
        "MLFLOW_EXPERIMENT_DEMAND": mlflow_cfg.get("experiment_demand"),
        "MLFLOW_EXPERIMENT_FARE": mlflow_cfg.get("experiment_fare"),
    }
    return {k: str(v) for k, v in candidates.items() if v}


def training_job_arguments(
    problem: str = "both",
    n_trials: int | None = None,
    register: bool = True,
) -> list[str]:
    """Build the CLI arguments passed to ``scripts/train_models.py`` in the job."""
    args = ["--problem", problem]
    if n_trials is not None:
        args += ["--n-trials", str(n_trials)]
    if not register:
        args.append("--no-register")
    return args


def submit_training_job(
    role_arn: str,
    image_uri: str,
    *,
    problem: str = "both",
    n_trials: int | None = None,
    register: bool = True,
    region: str = "us-east-1",
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    base_job_name: str = DEFAULT_BASE_JOB_NAME,
    wait: bool = True,
    config: Config | None = None,
) -> str:  # pragma: no cover - requires the sagemaker SDK + AWS
    """Run ``scripts/train_models.py`` as a SageMaker Processing job.

    ``image_uri`` is a training image containing this repo's code + deps
    (``requirements.txt`` + ``requirements-monitoring.txt``). Returns the job name.
    """
    try:
        import boto3
        from sagemaker.processing import ScriptProcessor
        from sagemaker.session import Session
    except ImportError as exc:
        raise RuntimeError(
            "sagemaker/boto3 not installed; add requirements-monitoring.txt."
        ) from exc

    session = Session(boto_session=boto3.Session(region_name=region))
    processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        role=role_arn,
        base_job_name=base_job_name,
        env=build_training_env(config),
        sagemaker_session=session,
    )
    processor.run(
        code=TRAIN_ENTRYPOINT,
        arguments=training_job_arguments(problem, n_trials, register),
        wait=wait,
    )
    job_name = processor.latest_job.job_name
    logger.info("Submitted SageMaker training job: %s", job_name)
    return job_name


def drift_job_arguments(
    problem: str,
    current_s3: str | None = None,
    *,
    output_dir: str = DRIFT_OUTPUT_DIR,
    namespace: str = DRIFT_NAMESPACE,
    psi_threshold: float | None = None,
    ks_alpha: float | None = None,
    no_cloudwatch: bool = False,
) -> list[str]:
    """Build the CLI arguments passed to ``scripts/monitoring/run_drift_job.py``.

    ``current_s3`` is the ``s3://bucket/prefix`` of Firehose-captured inference
    records; omit it only for a reference-only dry run.
    """
    args = ["--problem", problem, "--output-dir", output_dir, "--namespace", namespace]
    if current_s3:
        args += ["--current-s3", current_s3]
    if psi_threshold is not None:
        args += ["--psi-threshold", str(psi_threshold)]
    if ks_alpha is not None:
        args += ["--ks-alpha", str(ks_alpha)]
    if no_cloudwatch:
        args.append("--no-cloudwatch")
    return args


def submit_drift_job(
    role_arn: str,
    image_uri: str,
    *,
    problem: str,
    current_s3: str | None = None,
    output_s3_uri: str | None = None,
    region: str = "us-east-1",
    instance_type: str = DEFAULT_DRIFT_INSTANCE_TYPE,
    base_job_name: str = DEFAULT_DRIFT_BASE_JOB_NAME,
    wait: bool = True,
) -> str:  # pragma: no cover - requires the sagemaker SDK + AWS
    """Run ``scripts/monitoring/run_drift_job.py`` as a SageMaker Processing job.

    ``image_uri`` is a monitoring image containing this repo's code + deps
    (``requirements.txt`` + ``requirements-monitoring.txt``). The job reads
    captured records from ``current_s3`` and uploads ``drift_metrics.json`` to
    ``output_s3_uri`` (when given). Returns the job name.
    """
    try:
        import boto3
        from sagemaker.processing import ProcessingOutput, ScriptProcessor
        from sagemaker.session import Session
    except ImportError as exc:
        raise RuntimeError(
            "sagemaker/boto3 not installed; add requirements-monitoring.txt."
        ) from exc

    session = Session(boto_session=boto3.Session(region_name=region))
    processor = ScriptProcessor(
        image_uri=image_uri,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        role=role_arn,
        base_job_name=base_job_name,
        sagemaker_session=session,
    )
    outputs = (
        [ProcessingOutput(source=DRIFT_OUTPUT_DIR, destination=output_s3_uri)]
        if output_s3_uri
        else None
    )
    processor.run(
        code=DRIFT_ENTRYPOINT,
        arguments=drift_job_arguments(problem, current_s3),
        outputs=outputs,
        wait=wait,
    )
    job_name = processor.latest_job.job_name
    logger.info("Submitted SageMaker drift job: %s", job_name)
    return job_name
