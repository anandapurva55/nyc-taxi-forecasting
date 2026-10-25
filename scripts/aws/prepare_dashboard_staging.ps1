<#
.SYNOPSIS
  Prepare the local dashboard to use the staging AWS API and optional S3 evidence.

.DESCRIPTION
  Reads Terraform outputs, verifies the public API health endpoint, and sets
  API_BASE_URL for the current PowerShell process. With -DownloadS3Artifacts,
  it also downloads fairness and drift evidence and sets RESPONSIBLE_DIR and
  DRIFT_DIR to the isolated local cache.
#>
param(
  [string]$Region = "us-east-1",
  [switch]$DownloadS3Artifacts
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$tfDir = Join-Path $repoRoot "infra\terraform"

$apiUrl = (terraform -chdir="$tfDir" output -raw api_url).Trim().TrimEnd("/")
if (-not $apiUrl) { throw "Terraform output api_url is empty." }

Write-Host "==> Verifying staging API at $apiUrl"
$health = Invoke-RestMethod -Uri "$apiUrl/health" -TimeoutSec 60
if ($health.status -ne "ok" -or -not $health.demand_model_loaded -or -not $health.fare_model_loaded) {
  throw "The staging API responded but did not report both models ready."
}
$env:API_BASE_URL = $apiUrl

if ($DownloadS3Artifacts) {
  $cicdBucket = (terraform -chdir="$tfDir" output -raw cicd_artifacts_bucket).Trim()
  $monitoringBucket = (terraform -chdir="$tfDir" output -raw monitoring_bucket).Trim()
  if (-not $cicdBucket) { throw "Terraform output cicd_artifacts_bucket is empty." }
  if (-not $monitoringBucket) { throw "Terraform output monitoring_bucket is empty." }

  $cacheRoot = Join-Path $repoRoot "artifacts\aws-showcase"
  $responsibleCache = Join-Path $cacheRoot "responsible"
  $driftCache = Join-Path $cacheRoot "drift"
  $verificationDir = Join-Path $repoRoot "artifacts\verification"
  New-Item -ItemType Directory -Force -Path $responsibleCache, $driftCache | Out-Null
  New-Item -ItemType Directory -Force -Path $verificationDir | Out-Null

  Write-Host "==> Downloading latest fairness evidence"
  aws s3 sync "s3://$cicdBucket/dashboard/responsible/latest" $responsibleCache --region $Region --only-show-errors --delete
  if ($LASTEXITCODE -ne 0) { throw "Failed to download the latest fairness evidence." }

  Write-Host "==> Downloading drift evidence"
  aws s3 sync "s3://$monitoringBucket/drift-reports" $driftCache --region $Region --only-show-errors --delete
  if ($LASTEXITCODE -ne 0) { throw "Failed to download the drift evidence." }

  $sources = @()
  $prefixes = @(
    @{ Bucket = $cicdBucket; Prefix = "dashboard/responsible/latest/"; Kind = "fairness" },
    @{ Bucket = $monitoringBucket; Prefix = "drift-reports/"; Kind = "drift" }
  )
  foreach ($source in $prefixes) {
    $listing = aws s3api list-objects-v2 --bucket $source.Bucket --prefix $source.Prefix --region $Region --output json | ConvertFrom-Json
    if ($LASTEXITCODE -ne 0) { throw "Failed to list s3://$($source.Bucket)/$($source.Prefix)" }
    foreach ($object in $listing.Contents) {
      $head = aws s3api head-object --bucket $source.Bucket --key $object.Key --region $Region --output json | ConvertFrom-Json
      if ($LASTEXITCODE -ne 0) { throw "Failed to read metadata for s3://$($source.Bucket)/$($object.Key)" }
      $evidenceKind = if ($head.Metadata.'evidence-kind') { $head.Metadata.'evidence-kind' } else { $source.Kind }
      $sources += [pscustomobject]@{
        evidence_kind = $evidenceKind
        s3_uri = "s3://$($source.Bucket)/$($object.Key)"
        etag = $head.ETag
        content_length = $head.ContentLength
        last_modified = $head.LastModified
        version_id = $head.VersionId
        metadata = $head.Metadata
      }
    }
  }

  $checkedAt = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssZ")
  $manifestPath = Join-Path $verificationDir "dashboard-s3-sources-$checkedAt.json"
  [pscustomobject]@{
    api_url = $apiUrl
    checked_at = $checkedAt
    sources = $sources
  } | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding utf8

  $env:RESPONSIBLE_DIR = $responsibleCache
  $env:MODEL_CARDS_DIR = $responsibleCache
  $env:DRIFT_DIR = $driftCache
}

$health | Select-Object status, demand_model_loaded, fare_model_loaded, history_hours, zones | Format-List
Write-Host "API_BASE_URL=$env:API_BASE_URL"
if ($DownloadS3Artifacts) {
  Write-Host "RESPONSIBLE_DIR=$env:RESPONSIBLE_DIR"
  Write-Host "MODEL_CARDS_DIR=$env:MODEL_CARDS_DIR"
  Write-Host "DRIFT_DIR=$env:DRIFT_DIR"
  Write-Host "S3_SOURCE_MANIFEST=$manifestPath"
}
Write-Host "OK: start Streamlit from this PowerShell process."