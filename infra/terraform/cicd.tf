# =============================================================================
# AWS-native CI/CD (CodeBuild) - the active hosted pipeline.
#
# Runs lint + tests on every push/PR and, on a push to the default branch,
# builds the self-contained image, pushes it to ECR, and rolls the ECS service.
# GitHub supplies source and webhook events; CodeBuild executes the pipeline.
#
# Gated on var.enable_cicd (default false) and requires a GitHub token so
# CodeBuild can read the repo and register a webhook. terraform validate passes
# without any of these set.
# =============================================================================

locals {
  cicd_enabled = var.enable_cicd ? 1 : 0
}

# --- S3: baked build artifacts (mlruns/, mlflow.db, demand parquet) -----------
# The image bakes in gitignored artifacts absent from a fresh checkout; upload
# them here once with scripts/aws/upload_artifacts.ps1. CodeBuild pulls them.
resource "aws_s3_bucket" "cicd_artifacts" {
  count         = local.cicd_enabled
  bucket        = "${local.name}-cicd-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "cicd_artifacts" {
  count                   = local.cicd_enabled
  bucket                  = aws_s3_bucket.cicd_artifacts[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- GitHub source credential (token) so CodeBuild can clone + add a webhook ---
resource "aws_codebuild_source_credential" "github" {
  count       = var.enable_cicd && var.github_token != "" ? 1 : 0
  auth_type   = "PERSONAL_ACCESS_TOKEN"
  server_type = "GITHUB"
  token       = var.github_token

  lifecycle {
    ignore_changes = [token]
  }
}

# --- IAM: CodeBuild service role ----------------------------------------------
data "aws_iam_policy_document" "codebuild_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["codebuild.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "codebuild" {
  count              = local.cicd_enabled
  name               = "${local.name}-codebuild"
  assume_role_policy = data.aws_iam_policy_document.codebuild_assume.json
}

resource "aws_iam_role_policy" "codebuild" {
  count = local.cicd_enabled
  name  = "${local.name}-codebuild"
  role  = aws_iam_role.codebuild[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = ["s3:GetObject", "s3:ListBucket"]
        Resource = [
          aws_s3_bucket.cicd_artifacts[0].arn,
          "${aws_s3_bucket.cicd_artifacts[0].arn}/*"
        ]
      },
      {
        Effect   = "Allow"
        Action   = ["ecr:GetAuthorizationToken"]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:GetDownloadUrlForLayer",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload"
        ]
        Resource = concat(
          [aws_ecr_repository.api.arn],
          var.enable_dashboard ? [aws_ecr_repository.dashboard[0].arn] : []
        )
      },
      {
        Effect   = "Allow"
        Action   = ["ecs:UpdateService", "ecs:DescribeServices"]
        Resource = "*"
      }
    ]
  })
}

# --- CloudWatch log group for build output ------------------------------------
resource "aws_cloudwatch_log_group" "codebuild" {
  count             = local.cicd_enabled
  name              = "/codebuild/${local.name}"
  retention_in_days = 14
}

# --- CodeBuild project --------------------------------------------------------
resource "aws_codebuild_project" "ci" {
  count        = local.cicd_enabled
  name         = "${local.name}-ci"
  description  = "Lint/test on push+PR; build+push+deploy on ${var.deploy_branch}."
  service_role = aws_iam_role.codebuild[0].arn


  lifecycle {
    precondition {
      condition     = var.github_token != ""
      error_message = "enable_cicd=true requires TF_VAR_github_token for CodeBuild source access."
    }
  }
  artifacts {
    type = "NO_ARTIFACTS"
  }

  environment {
    compute_type    = "BUILD_GENERAL1_SMALL"
    image           = "aws/codebuild/standard:7.0"
    type            = "LINUX_CONTAINER"
    privileged_mode = true # required to build Docker images

    environment_variable {
      name  = "AWS_REGION"
      value = var.aws_region
    }
    environment_variable {
      name  = "ECR_REPOSITORY_URL"
      value = aws_ecr_repository.api.repository_url
    }
    environment_variable {
      name  = "ECS_CLUSTER"
      value = aws_ecs_cluster.main.name
    }
    environment_variable {
      name  = "ECS_SERVICE"
      value = aws_ecs_service.api.name
    }
    environment_variable {
      name  = "ARTIFACTS_BUCKET"
      value = aws_s3_bucket.cicd_artifacts[0].bucket
    }
    environment_variable {
      name  = "IMAGE_TAG"
      value = var.image_tag
    }
    environment_variable {
      name  = "DEPLOY_BRANCH"
      value = var.deploy_branch
    }
  }

  source {
    type                = "GITHUB"
    location            = var.github_repo_url
    git_clone_depth     = 1
    buildspec           = "buildspec.yml"
    report_build_status = true
  }

  logs_config {
    cloudwatch_logs {
      group_name = aws_cloudwatch_log_group.codebuild[0].name
    }
  }
}

# --- Webhook: trigger on push to deploy branch + PRs --------------------------
resource "aws_codebuild_webhook" "ci" {
  count        = var.enable_cicd && var.github_token != "" ? 1 : 0
  project_name = aws_codebuild_project.ci[0].name
  build_type   = "BUILD"

  filter_group {
    filter {
      type    = "EVENT"
      pattern = "PUSH"
    }
    filter {
      type    = "HEAD_REF"
      pattern = "refs/heads/${var.deploy_branch}"
    }
  }

  filter_group {
    filter {
      type    = "EVENT"
      pattern = "PULL_REQUEST_CREATED,PULL_REQUEST_UPDATED,PULL_REQUEST_REOPENED"
    }
  }

  depends_on = [aws_codebuild_source_credential.github]
}
