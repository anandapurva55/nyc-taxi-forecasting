<#
.SYNOPSIS
  Tear down all AWS resources created by the deploy (stops all charges).

.DESCRIPTION
  Runs `terraform destroy`. ECR (force_delete) and ECS are removed, so no
  Fargate/ALB charges continue. Requires Terraform, AWS CLI and credentials.

.EXAMPLE
  .\scripts\aws\destroy.ps1 -Region us-east-1
#>

[CmdletBinding()]
param(
    [string]$Region = "us-east-1",
    [string]$Tag = "latest"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$TfDir = Join-Path $RepoRoot "infra\terraform"

if (-not (Get-Command terraform -ErrorAction SilentlyContinue)) {
    throw "Required tool 'terraform' not found on PATH."
}

Write-Host "==> terraform destroy (region $Region)" -ForegroundColor Cyan
terraform -chdir="$TfDir" destroy -input=false -auto-approve `
    -var "aws_region=$Region" -var "image_tag=$Tag"
if ($LASTEXITCODE -ne 0) { throw "terraform destroy failed." }

Write-Host "All resources destroyed. Charges stopped." -ForegroundColor Green
