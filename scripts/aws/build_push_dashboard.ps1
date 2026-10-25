<#
.SYNOPSIS
  Build the Streamlit dashboard image and push it to Amazon ECR.

.DESCRIPTION
  1. Logs Docker in to ECR.
  2. Builds docker/Dockerfile.dashboard and pushes <repo>:<Tag>.

  The dashboard image is stateless (no models / MLflow); it talks to the API
  over HTTP via API_BASE_URL, which the Terraform task definition injects.

  Requires: AWS CLI, Docker, Terraform. Run after `enable_dashboard=true` has
  been applied so the dashboard ECR repo exists.

.EXAMPLE
  .\scripts\aws\build_push_dashboard.ps1 -Region us-east-1 -Tag latest
#>

[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [string]$Tag = "latest",
    # Dashboard ECR repository URL. If omitted, read from the Terraform output.
    [string]$RepoUrl
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TfDir = Join-Path $RepoRoot "infra\terraform"

function Assert-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required tool '$name' not found on PATH."
    }
}

Assert-Tool aws
Assert-Tool docker
Assert-Tool terraform

Set-Location $RepoRoot

# 1. Resolve the dashboard ECR repository URL.
if (-not $RepoUrl) {
    Write-Host "==> Reading dashboard ECR repository URL from Terraform output" -ForegroundColor Cyan
    $RepoUrl = (terraform -chdir="$TfDir" output -raw dashboard_ecr_repository_url).Trim()
}
if (-not $RepoUrl) { throw "Could not determine dashboard ECR repository URL (is enable_dashboard applied?)." }
$Registry = $RepoUrl.Split("/")[0]
$Image = "${RepoUrl}:${Tag}"

# 2. Log Docker in to ECR.
Write-Host "==> Logging in to ECR: $Registry" -ForegroundColor Cyan
(aws ecr get-login-password --region $Region) | docker login --username AWS --password-stdin $Registry
if ($LASTEXITCODE -ne 0) { throw "ECR login failed." }

# 3. Build and push.
Write-Host "==> Building dashboard image $Image" -ForegroundColor Cyan
docker build -f docker\Dockerfile.dashboard -t $Image .
if ($LASTEXITCODE -ne 0) { throw "docker build failed." }

Write-Host "==> Pushing dashboard image $Image" -ForegroundColor Cyan
docker push $Image
if ($LASTEXITCODE -ne 0) { throw "docker push failed." }

Write-Host "OK: pushed $Image" -ForegroundColor Green
