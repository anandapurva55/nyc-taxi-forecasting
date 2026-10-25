"""SageMaker Pipeline definition for gated retraining.

This builds a ``sagemaker.workflow`` Pipeline that retrains both models, compares
them with the seed production registry and absolute limits, and - only if every
gate passes and deployment is authorized - triggers a rebuild of the
self-contained serving image (ECR) and an ECS force-new-deployment.

Design notes
------------
* The heavy ``sagemaker`` SDK is imported lazily inside :func:`build_pipeline`;
  importing this module does not require the SDK (so it stays lint/compile-safe
  and the SDK only ships in ``requirements-monitoring.txt``).
* Deployment uses the project's **immutable-image** model: rather than pushing a
  model artifact to an always-on endpoint, we rebuild the baked image and roll
  the Fargate service. The build+deploy work lives in
  :mod:`retraining.build_and_deploy`, invoked from the pipeline's final step
  (e.g. via a CodeBuild project or a processor running that script).

Usage (requires AWS creds + ``requirements-monitoring.txt``)::

    python -m retraining.pipeline --upsert --role-arn arn:aws:iam::...:role/... \
        --image-uri <ecr>/taxi-training:latest
"""

from __future__ import annotations

import argparse

DEFAULT_PIPELINE_NAME = "taxi-forecasting-retrain"
DEFAULT_INSTANCE_TYPE = "ml.m5.large"


def build_pipeline(
    role_arn: str,
    image_uri: str,
    features_s3_uri: str,
    registry_seed_s3_uri: str,
    artifact_bucket: str,
    *,
    codebuild_project: str = "",
    deployment_artifact_bucket: str = "",
    release_id: str = "retrained",
    region: str = "us-east-1",
    pipeline_name: str = DEFAULT_PIPELINE_NAME,
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    rmse_max_demand: float = 50.0,
    rmse_max_fare: float = 5.0,
):
    """Construct (but do not submit) the retraining pipeline.

    ``image_uri`` is a training image containing this repo's code + deps. The
    pipeline runs three ``ProcessingStep``s (train -> evaluate -> conditional
    build+deploy) wired through a ``ConditionStep`` quality gate.
    """
    import boto3
    from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
    from sagemaker.workflow.condition_step import ConditionStep
    from sagemaker.workflow.conditions import ConditionEquals, ConditionLessThanOrEqualTo
    from sagemaker.workflow.functions import Join, JsonGet
    from sagemaker.workflow.parameters import ParameterInteger, ParameterString
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.pipeline_context import PipelineSession
    from sagemaker.workflow.properties import PropertyFile
    from sagemaker.workflow.steps import ProcessingStep

    session = PipelineSession(
        boto_session=boto3.Session(region_name=region),
        default_bucket=artifact_bucket,
        default_bucket_prefix="sagemaker-pipelines",
    )

    p_image = ParameterString(name="TrainingImageUri", default_value=image_uri)
    p_codebuild = ParameterString(name="CodeBuildProject", default_value=codebuild_project)
    p_artifacts = ParameterString(
        name="DeploymentArtifactBucket", default_value=deployment_artifact_bucket
    )
    p_release = ParameterString(name="ReleaseId", default_value=release_id)
    p_features = ParameterString(name="FeaturesS3Uri", default_value=features_s3_uri)
    p_registry = ParameterString(name="RegistrySeedS3Uri", default_value=registry_seed_s3_uri)
    p_trials = ParameterInteger(name="TrainingTrials", default_value=25)
    p_deploy = ParameterString(name="DeployAfterQualityGate", default_value="false")

    processor = ScriptProcessor(
        image_uri=p_image,
        command=["python3"],
        instance_type=instance_type,
        instance_count=1,
        role=role_arn,
        sagemaker_session=session,
        base_job_name=f"{pipeline_name}-train",
        env={"PYTHONPATH": "/app"},
    )

    features_input = ProcessingInput(
        source=p_features,
        destination="/opt/ml/processing/input/features",
        input_name="features",
    )

    # 1. Train from precomputed splits and export the refreshed MLflow registry.
    train_step = ProcessingStep(
        name="TrainModels",
        processor=processor,
        code="retraining/train_step.py",
        inputs=[
            features_input,
            ProcessingInput(
                source=p_registry,
                destination="/opt/ml/processing/input/registry",
                input_name="registry-seed",
            ),
        ],
        job_arguments=[
            "--features-dir",
            "/opt/ml/processing/input/features",
            "--registry-seed-dir",
            "/opt/ml/processing/input/registry",
            "--output-dir",
            "/opt/ml/processing/model",
            "--n-trials",
            Join(on="", values=[p_trials]),
        ],
        outputs=[
            ProcessingOutput(output_name="model", source="/opt/ml/processing/model"),
        ],
    )

    # 2. Evaluate - writes evaluation.json with per-model RMSE for the gate.
    eval_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    eval_step = ProcessingStep(
        name="EvaluateModels",
        processor=processor,
        code="scripts/monitoring/evaluate_models.py",
        inputs=[
            features_input,
            ProcessingInput(
                source=train_step.properties.ProcessingOutputConfig.Outputs[
                    "model"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/model",
                input_name="trained-registry",
            ),
            ProcessingInput(
                source=p_registry,
                destination="/opt/ml/processing/baseline",
                input_name="baseline-registry",
            ),
        ],
        job_arguments=[
            "--features-dir",
            "/opt/ml/processing/input/features",
            "--registry-dir",
            "/opt/ml/processing/model",
            "--baseline-registry-dir",
            "/opt/ml/processing/baseline",
            "--output-dir",
            "/opt/ml/processing/evaluation",
        ],
        cache_config=None,
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        property_files=[eval_report],
    )

    # 3. Publish the gated registry and let privileged CodeBuild deploy it.
    deploy_step = ProcessingStep(
        name="BuildAndDeploy",
        processor=processor,
        code="retraining/build_and_deploy.py",
        inputs=[
            ProcessingInput(
                source=train_step.properties.ProcessingOutputConfig.Outputs[
                    "model"
                ].S3Output.S3Uri,
                destination="/opt/ml/processing/model",
                input_name="gated-registry",
            )
        ],
        job_arguments=[
            "--model-dir",
            "/opt/ml/processing/model",
            "--artifacts-bucket",
            p_artifacts,
            "--codebuild-project",
            p_codebuild,
            "--release-id",
            p_release,
            "--region",
            region,
        ],
    )

    gate = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="demand.rmse",
                ),
                right=rmse_max_demand,
            ),
            ConditionLessThanOrEqualTo(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="fare.rmse",
                ),
                right=rmse_max_fare,
            ),
            ConditionEquals(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="demand.baseline_gate_passed",
                ),
                right=True,
            ),
            ConditionEquals(
                left=JsonGet(
                    step_name=eval_step.name,
                    property_file=eval_report,
                    json_path="fare.baseline_gate_passed",
                ),
                right=True,
            ),
            ConditionEquals(left=p_deploy, right="true"),
        ],
        if_steps=[deploy_step],
        else_steps=[],
    )

    return Pipeline(
        name=pipeline_name,
        parameters=[
            p_image,
            p_codebuild,
            p_artifacts,
            p_release,
            p_features,
            p_registry,
            p_trials,
            p_deploy,
        ],
        steps=[train_step, eval_step, gate],
        sagemaker_session=session,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create/upsert the retraining pipeline.")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN.")
    parser.add_argument("--image-uri", required=True, help="Training image URI (ECR).")
    parser.add_argument("--features-s3-uri", required=True)
    parser.add_argument("--registry-seed-s3-uri", required=True)
    parser.add_argument("--artifact-bucket", required=True)
    parser.add_argument("--codebuild-project", default="")
    parser.add_argument("--deployment-artifact-bucket", default="")
    parser.add_argument("--release-id", default="retrained")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--pipeline-name", default=DEFAULT_PIPELINE_NAME)
    parser.add_argument("--upsert", action="store_true", help="Create/update the pipeline in AWS.")
    args = parser.parse_args(argv)

    pipeline = build_pipeline(
        role_arn=args.role_arn,
        image_uri=args.image_uri,
        features_s3_uri=args.features_s3_uri,
        registry_seed_s3_uri=args.registry_seed_s3_uri,
        artifact_bucket=args.artifact_bucket,
        codebuild_project=args.codebuild_project,
        deployment_artifact_bucket=args.deployment_artifact_bucket,
        release_id=args.release_id,
        region=args.region,
        pipeline_name=args.pipeline_name,
    )
    if args.upsert:
        pipeline.upsert(role_arn=args.role_arn)
        print(f"Upserted pipeline: {args.pipeline_name}")
    else:
        print(pipeline.definition())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
