"""Generate the architecture diagrams used in the README.

Uses the `diagrams` library (https://diagrams.mingrammer.com) with the official
AWS icon set. Graphviz must be installed (`brew install graphviz`).

    python -m venv .venv-diagrams
    .venv-diagrams/bin/pip install -r docs/diagrams/requirements.txt
    .venv-diagrams/bin/python docs/diagrams/generate.py

PNG files are written to docs/images/.
"""

from __future__ import annotations

from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import Athena, GlueDataCatalog, KinesisDataFirehose
from diagrams.aws.compute import ECR, ElasticContainerServiceService, Fargate
from diagrams.aws.devtools import Codebuild
from diagrams.aws.integration import EventbridgeScheduler, SimpleNotificationServiceSns
from diagrams.aws.management import Cloudwatch, CloudwatchAlarm, CloudwatchLogs
from diagrams.aws.ml import SagemakerModel
from diagrams.aws.network import ElbApplicationLoadBalancer
from diagrams.aws.security import IdentityAndAccessManagementIamRole
from diagrams.aws.storage import SimpleStorageServiceS3Bucket
from diagrams.generic.storage import Storage
from diagrams.onprem.client import Users
from diagrams.onprem.mlops import Mlflow
from diagrams.onprem.vcs import Github
from diagrams.programming.flowchart import Database, Document
from diagrams.programming.framework import Fastapi
from diagrams.programming.language import Python

OUT = Path(__file__).resolve().parents[1] / "images"
OUT.mkdir(parents=True, exist_ok=True)

GRAPH = {"fontsize": "20", "pad": "0.4", "splines": "spline", "nodesep": "0.7", "ranksep": "0.9"}
NODE = {"fontsize": "13"}


def out(name: str) -> str:
    return str(OUT / name)


def ml_lifecycle() -> None:
    with Diagram(
        "End-to-End ML Lifecycle",
        filename=out("ml_lifecycle"),
        show=False,
        direction="LR",
        graph_attr=GRAPH,
        node_attr=NODE,
    ):
        source = Storage("NYC TLC\nYellow Taxi\nFeb 2024")

        with Cluster("Data pipeline"):
            pipeline = Python("ingest, validate,\naggregate, features")
            splits = Database("chronological splits\ntrain, val, test\nParquet")

        with Cluster("Training and registry"):
            tuning = Python("Optuna tuning\nXGBoost vs LightGBM")
            registry = Mlflow("MLflow tracking\n+ model registry")

        with Cluster("Serving"):
            api = Fastapi("FastAPI\nreal-time API")
            batch = Python("batch forecast\n(predictions.db)")

        with Cluster("Monitoring, responsible AI, dashboard"):
            drift = Python("drift detection\nPSI + KS")
            rai = Python("SHAP + subgroup\nerror analysis")
            dashboard = Python("Streamlit\ndashboard")

        source >> pipeline >> splits >> tuning >> Edge(label="validation RMSE winner") >> registry
        registry >> api
        registry >> batch
        splits >> Edge(style="dashed", label="reference") >> drift
        registry >> rai
        api >> dashboard
        rai >> Edge(style="dashed") >> dashboard
        drift >> Edge(style="dashed") >> dashboard


def data_pipeline() -> None:
    with Diagram(
        "Data Pipeline and Lineage",
        filename=out("data_pipeline"),
        show=False,
        direction="LR",
        graph_attr=GRAPH,
        node_attr=NODE,
    ):
        raw = Storage("yellow_tripdata_2024-02\n3,007,526 trips")
        zones = Document("taxi_zone_lookup.csv\n265 zones")
        clean = Database("trips_clean.parquet\n2,722,931 trips")
        summary = Document("pipeline_summary.json\nrow counts + drop reasons")

        with Cluster("Demand branch (one row per zone and hour)"):
            hourly = Database("demand_hourly.parquet")
            demand_feats = Database("demand_features.parquet\nlags, rolling means,\ncalendar")
            demand_splits = Database("demand train, val, test")

        with Cluster("Fare branch (one row per trip)"):
            fare_feats = Database("fare_features.parquet\ndistance, zones,\ntime of day")
            fare_splits = Database("fare train, val, test")

        raw >> Edge(label="validate + clean") >> clean
        zones >> Edge(style="dashed") >> clean
        clean >> Edge(style="dashed") >> summary
        clean >> Edge(label="aggregate") >> hourly >> demand_feats >> Edge(label="80%, 10%, 10% by time") >> demand_splits
        clean >> fare_feats >> Edge(label="80%, 10%, 10% by time") >> fare_splits


def aws_architecture() -> None:
    with Diagram(
        "AWS Architecture",
        filename=out("aws_architecture"),
        show=False,
        direction="LR",
        graph_attr=GRAPH,
        node_attr=NODE,
    ):
        users = Users("API clients\nand analysts")

        with Cluster("Release path (CI/CD)"):
            github = Github("GitHub\nmain branch")
            codebuild = Codebuild("CodeBuild\nruff + pytest\ndocker build")
            artifacts = SimpleStorageServiceS3Bucket("S3 build inputs\nmlflow.db, mlruns,\ndemand history")
            ecr = ECR("ECR\nAPI, dashboard and\nmonitoring images")

        with Cluster("Serving path (default VPC, ECS cluster)"):
            alb = ElbApplicationLoadBalancer("API ALB\nHTTP :80")
            api = Fargate("API task\nFastAPI + models\n1 vCPU, 2 GB")
            dash_alb = ElbApplicationLoadBalancer("Dashboard ALB\nHTTP :80")
            dash = ElasticContainerServiceService("Dashboard task\nStreamlit")

        with Cluster("Monitoring path"):
            logs = CloudwatchLogs("CloudWatch Logs\nprediction events")
            firehose = KinesisDataFirehose("Data Firehose")
            capture = SimpleStorageServiceS3Bucket("S3 monitoring\ncaptured/ and\ndrift-reports/")
            scheduler = EventbridgeScheduler("EventBridge\nScheduler (weekly)")
            sagemaker = SagemakerModel("SageMaker\nProcessing job\n(drift)")
            metrics = Cloudwatch("CloudWatch metrics\n+ dashboard")
            alarm = CloudwatchAlarm("Alarms\ndrift, API 5xx")
            sns = SimpleNotificationServiceSns("SNS email\nalerts")

        with Cluster("Data lake"):
            lake = SimpleStorageServiceS3Bucket("S3 data lake\n(versioned)")
            glue = GlueDataCatalog("Glue catalog")
            athena = Athena("Athena")

        iam = IdentityAndAccessManagementIamRole("IAM roles\n(one per service)")

        github >> Edge(label="webhook") >> codebuild
        artifacts >> codebuild >> Edge(label="push image") >> ecr
        codebuild >> Edge(label="update service", style="dashed") >> api
        ecr >> Edge(style="dashed", label="pull") >> api

        users >> alb >> api
        users >> dash_alb >> dash >> Edge(label="REST calls") >> alb

        api >> Edge(label="stdout JSON") >> logs >> Edge(label="subscription filter") >> firehose >> capture
        scheduler >> sagemaker
        capture >> sagemaker >> Edge(label="drift_metrics.json") >> capture
        sagemaker >> metrics >> alarm >> sns

        lake >> glue >> athena
        iam >> Edge(style="dotted") >> api


def release_flow() -> None:
    with Diagram(
        "CI/CD Release Flow",
        filename=out("cicd_release"),
        show=False,
        direction="LR",
        graph_attr=GRAPH,
        node_attr=NODE,
    ):
        dev = Github("git push\nto main")
        with Cluster("AWS CodeBuild (buildspec.yml)"):
            install = Python("install\nrequirements")
            gate = Python("quality gate\nruff + pytest\n(coverage)")
            build = Codebuild("docker build\nself-contained\nAPI image")
        ecr = ECR("ECR\n:<commit> and :latest")
        ecs = ElasticContainerServiceService("ECS rolling\nupdate")
        health = Fastapi("/health = ok\nsmoke test")

        dev >> install >> gate >> build >> ecr >> ecs >> health


if __name__ == "__main__":
    ml_lifecycle()
    data_pipeline()
    aws_architecture()
    release_flow()
    print(f"Diagrams written to {OUT}")
