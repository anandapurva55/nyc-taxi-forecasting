<#
.SYNOPSIS
  Seed the CI/CD artifacts bucket with the gitignored static files the
  dashboard image bakes in (Responsible-AI reports, drift reports, and taxi zone
  lookup).

.DESCRIPTION
  The dashboard image (docker/Dockerfile.dashboard) COPYs artifacts/responsible/,
  artifacts/drift/, and data/raw/taxi_zone_lookup.csv, which are gitignored and
  therefore absent from the CodeBuild GitHub checkout. buildspec-dashboard.yml
  pulls them from s3://<cicd_artifacts_bucket>/dashboard/ at build time. Run this
  once (and again whenever those artifacts change).
#>
param(
  [string]$Region = "us-east-1"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tfDir = Join-Path $repoRoot "infra\terraform"

Write-Host "==> Reading CI/CD artifacts bucket from Terraform output"
$bucket = terraform -chdir="$tfDir" output -raw cicd_artifacts_bucket
if (-not $bucket) { throw "cicd_artifacts_bucket output is empty; is enable_cicd applied?" }

Write-Host "==> Uploading dashboard artifacts to s3://$bucket/dashboard"
aws s3 sync (Join-Path $repoRoot "artifacts\responsible") "s3://$bucket/dashboard/responsible" --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "Failed to upload Responsible-AI dashboard artifacts." }
aws s3 sync (Join-Path $repoRoot "artifacts\drift") "s3://$bucket/dashboard/drift" --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "Failed to upload drift dashboard artifacts." }
aws s3 cp (Join-Path $repoRoot "data\raw\taxi_zone_lookup.csv") "s3://$bucket/dashboard/taxi_zone_lookup.csv" --region $Region --only-show-errors
if ($LASTEXITCODE -ne 0) { throw "Failed to upload the dashboard taxi zone lookup." }

Write-Host "OK: dashboard artifacts uploaded to s3://$bucket/dashboard"
