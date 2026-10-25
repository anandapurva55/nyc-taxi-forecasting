<#
.SYNOPSIS
  Build the self-contained API image and push it to Amazon ECR.

.DESCRIPTION
  1. Regenerates the container-ready MLflow DB (mlflow.container.db) so baked
     artifact paths resolve at /app/mlruns inside the Linux image.
  2. Logs Docker in to ECR.
  3. Builds docker/Dockerfile and pushes <repo>:<Tag>.

  Requires: AWS CLI, Docker, and the project venv (for the portability script).
  Called by deploy.ps1, or run standalone once the ECR repo exists.

.EXAMPLE
  .\scripts\aws\build_push.ps1 -Region us-east-1 -Tag latest
#>

[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [string]$Tag = "latest",
    # ECR repository URL (e.g. <acct>.dkr.ecr.<region>.amazonaws.com/<your-ecr-repository>).
    # If omitted, it is read from the Terraform output.
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

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

Set-Location $RepoRoot

# 1. Produce the container-ready MLflow DB (idempotent).
Write-Host "==> Generating portable MLflow DB (mlflow.container.db)" -ForegroundColor Cyan
& $Python scripts\make_mlflow_portable.py --src mlflow.db --dest mlflow.container.db --new-base "file:///app/mlruns"
if ($LASTEXITCODE -ne 0) { throw "make_mlflow_portable.py failed." }

# 2. Resolve the ECR repository URL.
if (-not $RepoUrl) {
    Write-Host "==> Reading ECR repository URL from Terraform output" -ForegroundColor Cyan
    $RepoUrl = (terraform -chdir="$TfDir" output -raw ecr_repository_url).Trim()
}
if (-not $RepoUrl) { throw "Could not determine ECR repository URL." }
$Registry = $RepoUrl.Split("/")[0]
$Image = "${RepoUrl}:${Tag}"

# 3. Log Docker in to ECR.
Write-Host "==> Logging in to ECR: $Registry" -ForegroundColor Cyan
(aws ecr get-login-password --region $Region) | docker login --username AWS --password-stdin $Registry
if ($LASTEXITCODE -ne 0) { throw "ECR login failed." }

# 4. Build and push.
Write-Host "==> Building image $Image" -ForegroundColor Cyan
docker build -f docker\Dockerfile -t $Image .
if ($LASTEXITCODE -ne 0) { throw "docker build failed." }

Write-Host "==> Pushing image $Image" -ForegroundColor Cyan
docker push $Image
if ($LASTEXITCODE -ne 0) { throw "docker push failed." }

Write-Host "OK: pushed $Image" -ForegroundColor Green
