#!/usr/bin/env bash
# Publish responsible AI evidence to the versioned S3 data lake.
#
# Uploads each subgroup report, SHAP summary and model card to a stable
# latest/ prefix and an immutable UTC-stamped versions/<stamp>/ prefix, with
# metadata on every object. The data lake bucket has S3 versioning enabled, so
# each upload also receives an S3 version ID. A manifest of keys, ETags and
# version IDs is written to logs/fairness_evidence_<stamp>.json.
#
# Usage (from the repository root, AWS_PROFILE set):
#   bash scripts/aws/publish_fairness_evidence.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REGION="${AWS_REGION:-us-east-1}"
BUCKET="${BUCKET:-$(terraform -chdir=infra/terraform output -raw datalake_bucket)}"
STAMP="${STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
COMMIT="$(git rev-parse --short HEAD)"
PREFIX="evidence/responsible"
MANIFEST="logs/fairness_evidence_${STAMP}.json"
mkdir -p logs

ITEMS=(
    "demand fairness-report artifacts/responsible/demand_fairness.json"
    "fare fairness-report artifacts/responsible/fare_fairness.json"
    "demand shap-summary artifacts/responsible/demand_shap_summary.png"
    "fare shap-summary artifacts/responsible/fare_shap_summary.png"
    "demand model-card docs/model_cards/demand.md"
    "fare model-card docs/model_cards/fare.md"
)

entries=()
for item in "${ITEMS[@]}"; do
    read -r problem kind path <<<"$item"
    [[ -s "$path" ]] || { echo "missing evidence file: $path"; exit 1; }
    name="$(basename "$path")"
    meta="evidence-set=fairness,problem=$problem,evidence-kind=$kind,evidence-version=$STAMP,git-commit=$COMMIT"
    versioned_key="$PREFIX/versions/$STAMP/$name"

    aws s3 cp "$path" "s3://$BUCKET/$PREFIX/latest/$name" --region "$REGION" --metadata "$meta" --only-show-errors
    aws s3 cp "$path" "s3://$BUCKET/$versioned_key" --region "$REGION" --metadata "$meta" --only-show-errors

    head="$(aws s3api head-object --bucket "$BUCKET" --key "$versioned_key" --region "$REGION" \
        --query '{etag: ETag, version_id: VersionId, bytes: ContentLength, last_modified: LastModified}' --output json)"
    entries+=("{\"problem\": \"$problem\", \"evidence_kind\": \"$kind\", \"key\": \"$versioned_key\", \"s3\": $head}")
    echo "published $name"
done

{
    echo "{"
    echo "  \"evidence_version\": \"$STAMP\","
    echo "  \"git_commit\": \"$COMMIT\","
    echo "  \"objects\": ["
    for i in "${!entries[@]}"; do
        sep=","
        [[ $i -eq $((${#entries[@]} - 1)) ]] && sep=""
        echo "    ${entries[$i]}$sep"
    done
    echo "  ]"
    echo "}"
} >"$MANIFEST"

echo "manifest: $MANIFEST"
