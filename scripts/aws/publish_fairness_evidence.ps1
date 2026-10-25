<#
.SYNOPSIS
  Publish fairness evidence to S3 and retain object metadata for review.

.DESCRIPTION
  Uploads each fairness report, SHAP summary, and model card to both a stable
  latest prefix and an immutable UTC-stamped prefix in the CI/CD artifacts
  bucket. S3 head-object responses for the immutable objects are saved under
  artifacts/verification/.
#>
param(
  [string]$Region = "us-east-1",
  [string]$Bucket,
  [string]$Version = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tfDir = Join-Path $repoRoot "infra\terraform"
$verificationDir = Join-Path $repoRoot "artifacts\verification"
$items = @(
  @{ Problem = "demand"; Kind = "fairness-report"; Path = "artifacts\responsible\demand_fairness.json" },
  @{ Problem = "fare"; Kind = "fairness-report"; Path = "artifacts\responsible\fare_fairness.json" },
  @{ Problem = "demand"; Kind = "shap-summary"; Path = "artifacts\responsible\demand_shap_summary.png" },
  @{ Problem = "fare"; Kind = "shap-summary"; Path = "artifacts\responsible\fare_shap_summary.png" },
  @{ Problem = "demand"; Kind = "model-card"; Path = "docs\model_cards\demand.md" },
  @{ Problem = "fare"; Kind = "model-card"; Path = "docs\model_cards\fare.md" }
)

if (-not $Bucket) {
  Write-Host "==> Reading CI/CD artifacts bucket from Terraform output"
  $Bucket = (terraform -chdir="$tfDir" output -raw cicd_artifacts_bucket).Trim()
}
if (-not $Bucket) { throw "Could not determine the CI/CD artifacts bucket." }
if ($Version -notmatch '^[A-Za-z0-9._-]+$') {
  throw "Version may contain only letters, numbers, periods, underscores, and hyphens."
}

New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null
$manifest = @()

foreach ($item in $items) {
  $source = Join-Path $repoRoot $item.Path
  if (-not (Test-Path $source -PathType Leaf)) {
    throw "Required fairness artifact is missing: $($item.Path)"
  }

  $fileName = Split-Path $source -Leaf
  $latestKey = "dashboard/responsible/latest/$fileName"
  $versionedKey = "dashboard/responsible/versions/$Version/$fileName"
  $metadata = "evidence-set=fairness,problem=$($item.Problem),evidence-kind=$($item.Kind),evidence-version=$Version"

  Write-Host "==> Uploading $fileName"
  aws s3 cp $source "s3://$Bucket/$latestKey" --region $Region --metadata $metadata --only-show-errors
  if ($LASTEXITCODE -ne 0) { throw "Failed to upload s3://$Bucket/$latestKey" }
  aws s3 cp $source "s3://$Bucket/$versionedKey" --region $Region --metadata $metadata --only-show-errors
  if ($LASTEXITCODE -ne 0) { throw "Failed to upload s3://$Bucket/$versionedKey" }

  $head = aws s3api head-object --bucket $Bucket --key $versionedKey --region $Region --output json | ConvertFrom-Json
  if ($LASTEXITCODE -ne 0) { throw "Failed to read metadata for s3://$Bucket/$versionedKey" }
  $manifest += [pscustomobject]@{
    problem = $item.Problem
    evidence_kind = $item.Kind
    s3_uri = "s3://$Bucket/$versionedKey"
    etag = $head.ETag
    content_length = $head.ContentLength
    last_modified = $head.LastModified
    version_id = $head.VersionId
    metadata = $head.Metadata
  }
}

$manifestPath = Join-Path $verificationDir "fairness-s3-$Version.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding utf8

Write-Host "OK: published fairness evidence version $Version to s3://$Bucket/dashboard/responsible/"
Write-Host "OK: retained object metadata in $manifestPath"