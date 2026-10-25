# =============================================================================
# Storage - Data lake (S3) + optional Athena/Glue
#
# Gated on var.enable_datalake. Provides durable, team-shareable storage for the
# pipeline outputs (raw / processed / features), plus an optional Glue
# database and Athena workgroup so the parquet data lake is queryable with SQL.
#
# Cost: S3 storage for a month of TLC data is a few cents; Athena is priced per
# query (per TB scanned). Review current prices and expected usage before enabling.
# =============================================================================

locals {
  datalake_enabled = var.enable_datalake ? 1 : 0
  athena_enabled   = var.enable_datalake && var.enable_athena ? 1 : 0
}

# --- S3: the data lake bucket -------------------------------------------------
resource "aws_s3_bucket" "datalake" {
  count         = local.datalake_enabled
  bucket        = "${local.name}-datalake-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "datalake" {
  count                   = local.datalake_enabled
  bucket                  = aws_s3_bucket.datalake[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "datalake" {
  count  = local.datalake_enabled
  bucket = aws_s3_bucket.datalake[0].id
  versioning_configuration {
    status = "Enabled"
  }
}

# --- Athena query results bucket ---------------------------------------------
resource "aws_s3_bucket" "athena_results" {
  count         = local.athena_enabled
  bucket        = "${local.name}-athena-results-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "athena_results" {
  count                   = local.athena_enabled
  bucket                  = aws_s3_bucket.athena_results[0].id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# --- Glue catalog database (for Athena SQL over the parquet lake) -------------
resource "aws_glue_catalog_database" "datalake" {
  count = local.athena_enabled
  name  = replace("${local.name}_datalake", "-", "_")
}

# --- Athena workgroup ---------------------------------------------------------
resource "aws_athena_workgroup" "datalake" {
  count         = local.athena_enabled
  name          = "${local.name}-datalake"
  force_destroy = true

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = false

    result_configuration {
      output_location = "s3://${aws_s3_bucket.athena_results[0].bucket}/results/"
    }
  }
}
