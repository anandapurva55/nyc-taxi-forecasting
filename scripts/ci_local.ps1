<#
.SYNOPSIS
    Local quality runner with optional Terraform checks.

.DESCRIPTION
  Runs the CodeBuild Python checks plus optional Terraform validation locally
    before submitting a CodeBuild run:
    1. Ruff lint            (ruff check .)
    2. Pytest + coverage    (pytest --cov=src)
    3. Terraform fmt/validate (infra/terraform)  - skipped if terraform is absent

  Exits non-zero if any check fails, mirroring CI.

.EXAMPLE
  .\scripts\ci_local.ps1
#>

[CmdletBinding()]
param(
    # Skip the Terraform checks even if terraform is installed.
    [switch]$NoTerraform
)

$ErrorActionPreference = "Continue"

# Repo root = parent of this script's directory.
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# Prefer the project venv's Python if present, else fall back to `python`.
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$failures = @()

function Invoke-Step {
    param([string]$Name, [scriptblock]$Action)
    Write-Host ""
    Write-Host "=== $Name ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit $LASTEXITCODE)" -ForegroundColor Red
        $script:failures += $Name
    }
    else {
        Write-Host "OK: $Name" -ForegroundColor Green
    }
}

Invoke-Step "Ruff lint" { & $Python -m ruff check . }

Invoke-Step "Pytest + coverage" {
    & $Python -m pytest --cov=src --cov-report=term-missing --cov-report=xml
}

$tf = Get-Command terraform -ErrorAction SilentlyContinue
if ($NoTerraform) {
    Write-Host "`nSkipping Terraform checks (-NoTerraform)." -ForegroundColor Yellow
}
elseif (-not $tf) {
    Write-Host "`nSkipping Terraform checks: 'terraform' not found on PATH." -ForegroundColor Yellow
}
else {
    Push-Location (Join-Path $RepoRoot "infra\terraform")
    Invoke-Step "Terraform fmt" { terraform fmt -check -recursive }
    Invoke-Step "Terraform init" { terraform init -backend=false -input=false | Out-Null }
    Invoke-Step "Terraform validate" { terraform validate }
    Pop-Location
}

Write-Host ""
if ($failures.Count -gt 0) {
    Write-Host "CI FAILED - $($failures.Count) check(s) failed: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "All CI checks passed." -ForegroundColor Green
exit 0
