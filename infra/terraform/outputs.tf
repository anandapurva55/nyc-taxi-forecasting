output "api_url" {
  description = "Public HTTP endpoint of the API (via the ALB)."
  value       = "http://${aws_lb.api.dns_name}"
}

output "resource_group_name" {
  description = "AWS Resource Group that lists all tagged resources for this deployment."
  value       = aws_resourcegroups_group.all.name
}

output "alb_dns_name" {
  description = "DNS name of the Application Load Balancer."
  value       = aws_lb.api.dns_name
}

output "ecr_repository_url" {
  description = "ECR repository the ECS task pulls the image from."
  value       = aws_ecr_repository.api.repository_url
}

output "ecs_cluster_name" {
  description = "Name of the ECS cluster."
  value       = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  description = "Name of the ECS service (use to force new deployments)."
  value       = aws_ecs_service.api.name
}

# --- monitoring outputs ---------------------------------------------
output "monitoring_bucket" {
  description = "S3 bucket holding captured inference data and drift reports."
  value       = var.enable_monitoring ? aws_s3_bucket.monitoring[0].bucket : ""
}

output "alerts_topic_arn" {
  description = "SNS topic ARN for drift / API / retraining alerts."
  value       = var.enable_monitoring ? aws_sns_topic.alerts[0].arn : ""
}

output "sagemaker_role_arn" {
  description = "IAM role ARN assumed by the drift / retraining SageMaker jobs."
  value       = var.enable_monitoring ? aws_iam_role.sagemaker[0].arn : ""
}

output "capture_firehose_name" {
  description = "Kinesis Firehose delivery stream shipping capture logs to S3."
  value       = var.enable_monitoring ? aws_kinesis_firehose_delivery_stream.capture[0].name : ""
}

output "drift_schedule_name" {
  description = "EventBridge Scheduler that starts the weekly drift-detection SageMaker Processing job (empty unless monitoring_image_uri is set)."
  value       = local.drift_enabled == 1 ? aws_scheduler_schedule.drift[0].name : ""
}

# --- CI/CD outputs -----------------------------------------------------------
output "cicd_project_name" {
  description = "CodeBuild project running the AWS-native CI/CD pipeline."
  value       = var.enable_cicd ? aws_codebuild_project.ci[0].name : ""
}

output "cicd_artifacts_bucket" {
  description = "S3 bucket where baked build artifacts (mlruns/, mlflow.db, parquet) are uploaded."
  value       = var.enable_cicd ? aws_s3_bucket.cicd_artifacts[0].bucket : ""
}

# --- Data-lake outputs --------------------------------------------
output "datalake_bucket" {
  description = "S3 bucket holding the data lake (raw/processed/features)."
  value       = var.enable_datalake ? aws_s3_bucket.datalake[0].bucket : ""
}

output "glue_database" {
  description = "Glue catalog database backing Athena queries over the data lake."
  value       = var.enable_datalake && var.enable_athena ? aws_glue_catalog_database.datalake[0].name : ""
}

output "athena_workgroup" {
  description = "Athena workgroup for querying the parquet data lake."
  value       = var.enable_datalake && var.enable_athena ? aws_athena_workgroup.datalake[0].name : ""
}

# --- dashboard outputs ----------------------------------------------
output "dashboard_url" {
  description = "Public HTTP endpoint of the Streamlit dashboard (via its ALB)."
  value       = var.enable_dashboard ? "http://${aws_lb.dashboard[0].dns_name}" : ""
}

output "dashboard_ecr_repository_url" {
  description = "ECR repository the dashboard ECS task pulls the image from."
  value       = var.enable_dashboard ? aws_ecr_repository.dashboard[0].repository_url : ""
}

output "dashboard_service_name" {
  description = "Name of the dashboard ECS service (use to force new deployments)."
  value       = var.enable_dashboard ? aws_ecs_service.dashboard[0].name : ""
}
