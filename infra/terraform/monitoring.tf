# =============================================================================
# Operations - Monitoring, Drift Detection & Retraining (hybrid, pay-per-use)
#
# Everything here is gated on var.enable_monitoring and is pay-per-use: an S3
# bucket for captured inference data + drift reports, an SNS alert topic, a
# Kinesis Firehose that ships the API's prediction-capture logs to S3, CloudWatch
# alarms + dashboard, a SageMaker execution role for the drift/retraining jobs,
# and an EventBridge schedule that triggers the SageMaker retraining pipeline
# weekly when retrain_pipeline_arn is set.
#
# No always-on SageMaker endpoint is created - serving stays on ECS Fargate.
# =============================================================================

data "aws_caller_identity" "current" {}

locals {
  monitoring_enabled = var.enable_monitoring ? 1 : 0
  # Retrain schedule is only created once the pipeline ARN is known.
  retrain_enabled = var.enable_monitoring && var.retrain_pipeline_arn != "" ? 1 : 0
  # Drift Processing-job schedule is only created once a monitoring image exists.
  drift_enabled = var.enable_monitoring && var.monitoring_image_uri != "" ? 1 : 0
}

# --- S3: captured inference data + drift reports ------------------------------
resource "aws_s3_bucket" "monitoring" {
  count         = local.monitoring_enabled
  bucket        = "${local.name}-monitoring-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "monitoring" {
  count                   = local.monitoring_enabled
  bucket                  = aws_s3_bucket.monitoring[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "monitoring" {
  count  = local.monitoring_enabled
  bucket = aws_s3_bucket.monitoring[0].id

  rule {
    id     = "expire-captured-data"
    status = "Enabled"

    filter {
      prefix = ""
    }

    expiration {
      days = var.capture_retention_days
    }
  }
}

# --- SNS: alerts (drift, API 5xx, retraining) ---------------------------------
resource "aws_sns_topic" "alerts" {
  count = local.monitoring_enabled
  name  = "${local.name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  count     = local.monitoring_enabled
  topic_arn = aws_sns_topic.alerts[0].arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# --- Kinesis Firehose: prediction-capture logs -> S3 --------------------------
data "aws_iam_policy_document" "firehose_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["firehose.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "firehose" {
  count              = local.monitoring_enabled
  name               = "${local.name}-firehose"
  assume_role_policy = data.aws_iam_policy_document.firehose_assume.json
}

resource "aws_iam_role_policy" "firehose_s3" {
  count = local.monitoring_enabled
  name  = "${local.name}-firehose-s3"
  role  = aws_iam_role.firehose[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:AbortMultipartUpload",
          "s3:GetBucketLocation",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:ListBucketMultipartUploads",
          "s3:PutObject"
        ]
        Resource = [
          aws_s3_bucket.monitoring[0].arn,
          "${aws_s3_bucket.monitoring[0].arn}/*"
        ]
      }
    ]
  })
}

resource "aws_kinesis_firehose_delivery_stream" "capture" {
  count       = local.monitoring_enabled
  name        = "${local.name}-capture"
  destination = "extended_s3"

  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose[0].arn
    bucket_arn          = aws_s3_bucket.monitoring[0].arn
    prefix              = "captured/"
    error_output_prefix = "captured-errors/"
    buffering_size      = 5
    buffering_interval  = 300
    compression_format  = "GZIP"
  }
}

# --- CloudWatch Logs -> Firehose subscription (prediction-capture records) -----
data "aws_iam_policy_document" "logs_to_firehose_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["logs.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "logs_to_firehose" {
  count              = local.monitoring_enabled
  name               = "${local.name}-logs-to-firehose"
  assume_role_policy = data.aws_iam_policy_document.logs_to_firehose_assume.json
}

resource "aws_iam_role_policy" "logs_to_firehose" {
  count = local.monitoring_enabled
  name  = "${local.name}-logs-to-firehose"
  role  = aws_iam_role.logs_to_firehose[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["firehose:PutRecord", "firehose:PutRecordBatch"]
        Resource = [aws_kinesis_firehose_delivery_stream.capture[0].arn]
      }
    ]
  })
}

resource "aws_cloudwatch_log_subscription_filter" "capture" {
  count           = local.monitoring_enabled
  name            = "${local.name}-capture"
  log_group_name  = aws_cloudwatch_log_group.api.name
  filter_pattern  = "{ $.event = \"prediction\" }"
  destination_arn = aws_kinesis_firehose_delivery_stream.capture[0].arn
  role_arn        = aws_iam_role.logs_to_firehose[0].arn
}

# --- SageMaker: execution role for drift + retraining jobs --------------------
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  count              = local.monitoring_enabled
  name               = "${local.name}-sagemaker"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  count      = local.monitoring_enabled
  role       = aws_iam_role.sagemaker[0].name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# Extra permissions the jobs need: read/write monitoring S3, publish CloudWatch
# metrics, pull/push ECR images, and roll the ECS service after retraining.
resource "aws_iam_role_policy" "sagemaker_extra" {
  count = local.monitoring_enabled
  name  = "${local.name}-sagemaker-extra"
  role  = aws_iam_role.sagemaker[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = concat([
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:PutObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.monitoring[0].arn,
          "${aws_s3_bucket.monitoring[0].arn}/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["cloudwatch:PutMetricData"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchCheckLayerAvailability",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["ecs:UpdateService", "ecs:DescribeServices"]
        Resource = "*"
      }
      ],
      var.enable_cicd ? tolist([
        {
          Effect = "Allow"
          Action = tolist(["s3:GetObject", "s3:PutObject", "s3:ListBucket"])
          Resource = tolist([
            aws_s3_bucket.cicd_artifacts[0].arn,
            "${aws_s3_bucket.cicd_artifacts[0].arn}/*"
          ])
        },
        {
          Effect   = "Allow"
          Action   = tolist(["codebuild:StartBuild", "codebuild:BatchGetBuilds"])
          Resource = tolist([aws_codebuild_project.ci[0].arn])
        }
      ]) : tolist([])
    )
  })
}

# --- CloudWatch: drift + API health alarms ------------------------------------
resource "aws_cloudwatch_metric_alarm" "drift" {
  count               = local.monitoring_enabled
  alarm_name          = "${local.name}-feature-drift"
  alarm_description   = "One or more input features drifted vs. the training distribution."
  namespace           = "TaxiForecasting/Drift"
  metric_name         = "FeaturesDrifted"
  statistic           = "Maximum"
  period              = 86400
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts[0].arn]
}

resource "aws_cloudwatch_metric_alarm" "api_5xx" {
  count               = local.monitoring_enabled
  alarm_name          = "${local.name}-api-5xx"
  alarm_description   = "Elevated 5xx responses from the API target group."
  namespace           = "AWS/ApplicationELB"
  metric_name         = "HTTPCode_Target_5XX_Count"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts[0].arn]

  dimensions = {
    LoadBalancer = aws_lb.api.arn_suffix
    TargetGroup  = aws_lb_target_group.api.arn_suffix
  }
}

# --- CloudWatch: dashboard ----------------------------------------------------
resource "aws_cloudwatch_dashboard" "monitoring" {
  count          = local.monitoring_enabled
  dashboard_name = "${local.name}-monitoring"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Feature drift"
          region = var.aws_region
          metrics = [
            ["TaxiForecasting/Drift", "FeaturesDrifted"],
            ["TaxiForecasting/Drift", "MaxPSI"]
          ]
          view   = "timeSeries"
          period = 86400
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "API requests & errors"
          region = var.aws_region
          metrics = [
            ["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.api.arn_suffix],
            ["AWS/ApplicationELB", "HTTPCode_Target_5XX_Count", "LoadBalancer", aws_lb.api.arn_suffix]
          ]
          view   = "timeSeries"
          period = 300
        }
      }
    ]
  })
}

# --- EventBridge Scheduler: retrain (weekly) -----------------------------------
data "aws_iam_policy_document" "scheduler_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["scheduler.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "scheduler" {
  count              = local.monitoring_enabled
  name               = "${local.name}-scheduler"
  assume_role_policy = data.aws_iam_policy_document.scheduler_assume.json
}

resource "aws_iam_role_policy" "scheduler" {
  count = local.monitoring_enabled
  name  = "${local.name}-scheduler"
  role  = aws_iam_role.scheduler[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sagemaker:CreateProcessingJob", "sagemaker:StartPipelineExecution"]
        Resource = "*"
      },
      {
        Effect   = "Allow"
        Action   = ["iam:PassRole"]
        Resource = [aws_iam_role.sagemaker[0].arn]
      }
    ]
  })
}

resource "aws_scheduler_schedule" "retrain" {
  count = local.retrain_enabled
  name  = "${local.name}-retrain"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = var.retrain_schedule

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution"
    role_arn = aws_iam_role.scheduler[0].arn
    input = jsonencode({
      PipelineName       = var.retrain_pipeline_arn
      ClientRequestToken = "<aws.scheduler.execution-id>"
      PipelineParameters = [
        {
          Name  = "DeployAfterQualityGate"
          Value = "false"
        }
      ]
    })
  }
}

# --- EventBridge Scheduler: drift-detection Processing job (weekly) ------------
# Runs scripts/monitoring/run_drift_job.py as a one-off SageMaker Processing job:
# reads Firehose-captured records from S3, writes drift_metrics.json back to S3,
# and publishes drift metrics to CloudWatch (feeding the existing drift alarm).
resource "aws_scheduler_schedule" "drift" {
  count = local.drift_enabled
  name  = "${local.name}-drift"

  flexible_time_window {
    mode = "OFF"
  }

  schedule_expression = var.drift_schedule

  target {
    arn      = "arn:aws:scheduler:::aws-sdk:sagemaker:createProcessingJob"
    role_arn = aws_iam_role.scheduler[0].arn
    input = jsonencode({
      ProcessingJobName = "${local.name}-drift-<aws.scheduler.execution-id>"
      RoleArn           = aws_iam_role.sagemaker[0].arn
      AppSpecification = {
        ImageUri            = var.monitoring_image_uri
        ContainerEntrypoint = ["python3", "scripts/monitoring/run_drift_job.py"]
        ContainerArguments = [
          "--problem", var.drift_problem,
          "--current-s3", "s3://${aws_s3_bucket.monitoring[0].bucket}/captured/",
          "--output-dir", "/opt/ml/processing/output",
          "--namespace", "TaxiForecasting/Drift",
        ]
      }
      ProcessingResources = {
        ClusterConfig = {
          InstanceCount  = 1
          InstanceType   = var.drift_instance_type
          VolumeSizeInGB = 20
        }
      }
      ProcessingOutputConfig = {
        Outputs = [
          {
            OutputName = "drift"
            S3Output = {
              S3Uri        = "s3://${aws_s3_bucket.monitoring[0].bucket}/drift-reports/"
              LocalPath    = "/opt/ml/processing/output"
              S3UploadMode = "EndOfJob"
            }
          }
        ]
      }
    })
  }
}
