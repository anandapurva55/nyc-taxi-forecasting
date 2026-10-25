<#
.SYNOPSIS
  One-command AWS deploy: provision infra, push the image, roll out the API.

.DESCRIPTION
  Handles the ECR chicken-and-egg problem by applying in two phases:
    1. terraform apply (ECR repository only)
    2. build + push the self-contained image into ECR (build_push.ps1)
    3. terraform apply (ECS service, ALB, everything else)
    4. force a new ECS deployment so the task picks up the pushed image
    5. print the public ALB URL

  Requires: Terraform, AWS CLI, Docker, and AWS credentials in the environment
  (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, or `aws configure`).

  WARNING: this creates billable resources (Fargate + ALB, ~$25-35/month).
  Run scripts/aws/destroy.ps1 to tear everything down.

.EXAMPLE
  $env:AWS_ACCESS_KEY_ID="..."; $env:AWS_SECRET_ACCESS_KEY="..."
  .\scripts\aws\deploy.ps1 -Region us-east-1 -TerraformVarFile infra\terraform\staging.tfvars
#>

[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [string]$Tag = "latest",
    [string]$TerraformVarFile,
    [switch]$AllowDestroy
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TfDir = Join-Path $RepoRoot "infra\terraform"

function Assert-Tool($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required tool '$name' not found on PATH."
    }
}
"terraform", "aws", "docker" | ForEach-Object { Assert-Tool $_ }

$tfArgs = @("-var", "aws_region=$Region", "-var", "image_tag=$Tag")
if ($TerraformVarFile) {
  $resolvedVarFile = Resolve-Path (Join-Path $RepoRoot $TerraformVarFile) -ErrorAction Stop
  $tfArgs += "-var-file=$resolvedVarFile"
}

Write-Host "==> terraform init" -ForegroundColor Cyan
terraform -chdir="$TfDir" init -input=false
if ($LASTEXITCODE -ne 0) { throw "terraform init failed." }

Write-Host "==> Step 1: create ECR repository" -ForegroundColor Cyan
terraform -chdir="$TfDir" apply -input=false -auto-approve -target=aws_ecr_repository.api @tfArgs
if ($LASTEXITCODE -ne 0) { throw "terraform apply (ECR) failed." }

$RepoUrl = (terraform -chdir="$TfDir" output -raw ecr_repository_url).Trim()
Write-Host "==> Step 2: build & push image to $RepoUrl" -ForegroundColor Cyan
& (Join-Path $PSScriptRoot "build_push.ps1") -Region $Region -Tag $Tag -RepoUrl $RepoUrl

Write-Host "==> Step 3: apply full infrastructure" -ForegroundColor Cyan
$PlanPath = Join-Path $env:TEMP "taxi-forecasting-deploy.tfplan"
Remove-Item $PlanPath -Force -ErrorAction SilentlyContinue
try {
  terraform -chdir="$TfDir" plan -input=false -out="$PlanPath" @tfArgs
  if ($LASTEXITCODE -ne 0) { throw "terraform plan failed." }

  $PlanJson = terraform -chdir="$TfDir" show -json "$PlanPath" | ConvertFrom-Json
  $Deletes = @(
    $PlanJson.resource_changes | Where-Object {
      $_.change.actions -contains "delete"
    }
  )
  if ($Deletes.Count -gt 0 -and -not $AllowDestroy) {
    $Addresses = $Deletes.address -join ", "
    throw "Refusing a plan with $($Deletes.Count) delete action(s): $Addresses. Review variables and state; use -AllowDestroy only for an intentional teardown."
  }

  terraform -chdir="$TfDir" apply -input=false -auto-approve "$PlanPath"
  if ($LASTEXITCODE -ne 0) { throw "terraform apply (full) failed." }
}
finally {
  Remove-Item $PlanPath -Force -ErrorAction SilentlyContinue
}

# Force a new deployment so an unchanged tag (e.g. 'latest') is re-pulled.
$Cluster = (terraform -chdir="$TfDir" output -raw ecs_cluster_name).Trim()
$Service = (terraform -chdir="$TfDir" output -raw ecs_service_name).Trim()
Write-Host "==> Step 4: force new ECS deployment" -ForegroundColor Cyan
aws ecs update-service --cluster $Cluster --service $Service --force-new-deployment --region $Region | Out-Null

$Url = (terraform -chdir="$TfDir" output -raw api_url).Trim()
Write-Host ""
Write-Host "Deployed. API URL: $Url" -ForegroundColor Green
Write-Host "It may take 1-3 minutes for the task to become healthy." -ForegroundColor Yellow
Write-Host "Check:  curl $Url/health" -ForegroundColor Yellow
Write-Host "Teardown: .\scripts\aws\destroy.ps1 -Region $Region" -ForegroundColor Yellow
