variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Deployment environment name (dev/staging/prod)."
  type        = string
  default     = "staging"
}

variable "project_name" {
  description = "Prefix used when naming resources (also the ECR repo name)."
  type        = string
  default     = "taxi-forecasting"
}

variable "image_tag" {
  description = "Tag of the image in ECR that the ECS task runs."
  type        = string
  default     = "latest"
}

variable "container_port" {
  description = "Port the API listens on inside the container."
  type        = number
  default     = 8000
}

variable "desired_count" {
  description = "Number of Fargate tasks to run."
  type        = number
  default     = 1
}

variable "task_cpu" {
  description = "Fargate task CPU units (256 = 0.25 vCPU)."
  type        = number
  default     = 256
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 512
}

# --- monitoring & retraining ----------------------------------------
variable "enable_monitoring" {
  description = "Master switch for the monitoring stack (S3, SNS, alarms, Firehose, schedules)."
  type        = bool
  default     = false
}

variable "alert_email" {
  description = "Email address subscribed to the SNS alerts topic (drift / API 5xx / retrain)."
  type        = string
  default     = ""
}

variable "retrain_pipeline_arn" {
  description = "ARN of the SageMaker retraining pipeline. Leave empty until the pipeline is created; the weekly retrain schedule is only created when this is set."
  type        = string
  default     = ""
}

variable "retrain_schedule" {
  description = "EventBridge Scheduler expression for the retraining pipeline."
  type        = string
  default     = "rate(7 days)"
}

variable "monitoring_image_uri" {
  description = "ECR image URI (repo code + requirements-monitoring.txt) that the scheduled drift Processing job runs. Leave empty to skip creating the drift schedule."
  type        = string
  default     = ""
}

variable "drift_schedule" {
  description = "EventBridge Scheduler expression for the drift-detection Processing job."
  type        = string
  default     = "rate(7 days)"
}

variable "drift_problem" {
  description = "Which model's feature distribution the scheduled drift job inspects (demand or fare)."
  type        = string
  default     = "fare"
}

variable "drift_instance_type" {
  description = "SageMaker instance type for the scheduled drift Processing job."
  type        = string
  default     = "ml.t3.medium"
}

variable "capture_retention_days" {
  description = "Days to retain captured inference data / drift reports in S3 before expiry."
  type        = number
  default     = 90
}

# --- AWS-native CI/CD (CodeBuild) --------------------------------------------
variable "enable_cicd" {
  description = "Create the CodeBuild CI/CD stack (lint/test on push+PR, build+push+deploy on the default branch). Requires github_token."
  type        = bool
  default     = false
}

variable "github_repo_url" {
  description = "HTTPS URL of the GitHub repo CodeBuild builds from."
  type        = string
  default     = ""
}

variable "github_token" {
  description = "GitHub personal access token (repo + admin:repo_hook) so CodeBuild can clone and register a webhook. Provide via TF_VAR_github_token; never commit it."
  type        = string
  default     = ""
  sensitive   = true
}

variable "deploy_branch" {
  description = "Branch whose pushes trigger a build + ECR push + ECS deploy (PRs run tests only)."
  type        = string
  default     = "main"
}

# --- S3 data lake -------------------------------------------------
variable "enable_datalake" {
  description = "Create the S3 data-lake bucket (raw/processed/features). Optionally Athena/Glue via enable_athena."
  type        = bool
  default     = false
}

variable "enable_athena" {
  description = "Also create a Glue database + Athena workgroup + results bucket so the parquet data lake is queryable with SQL. Requires enable_datalake."
  type        = bool
  default     = false
}

# --- Streamlit showcase dashboard -----------------------------------
variable "enable_dashboard" {
  description = "Create the Streamlit dashboard stack (ECR + ALB + ECS service). Requires the dashboard image to be built and pushed."
  type        = bool
  default     = false
}

variable "dashboard_image_tag" {
  description = "Tag of the dashboard image in ECR that the ECS task runs."
  type        = string
  default     = "latest"
}

variable "dashboard_container_port" {
  description = "Port Streamlit listens on inside the container."
  type        = number
  default     = 8501
}

variable "dashboard_desired_count" {
  description = "Number of Fargate tasks to run for the dashboard."
  type        = number
  default     = 1
}

variable "dashboard_task_cpu" {
  description = "Fargate task CPU units for the dashboard (512 = 0.5 vCPU)."
  type        = number
  default     = 512
}

variable "dashboard_task_memory" {
  description = "Fargate task memory in MiB for the dashboard."
  type        = number
  default     = 1024
}
