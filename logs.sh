#!/usr/bin/env bash
set -e

if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
JOB_NAME="agent-peptide-sota-runner"
REGION="us-central1"

# Determine execution name (passed as argument or dynamically resolved to latest)
EXECUTION_NAME="$1"
if [ -z "${EXECUTION_NAME}" ]; then
  echo "🔍 Detecting latest execution for job: ${JOB_NAME}..."
  EXECUTION_NAME=$(gcloud run jobs executions list --job="${JOB_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --limit=1 --format="value(metadata.name)")
fi

if [ -z "${EXECUTION_NAME}" ]; then
  echo "❌ No executions found for job: ${JOB_NAME}"
  exit 1
fi

echo "============================================================"
echo "📋 Showing logs for Cloud Run Execution: ${EXECUTION_NAME}"
echo "   (Project: ${PROJECT_ID} | Region: ${REGION})"
echo "============================================================"
echo ""
echo "--- 📜 Recent Logs (Last 30 lines) ---"
gcloud beta run jobs executions logs read "${EXECUTION_NAME}" --region="${REGION}" --project="${PROJECT_ID}" --limit=30 || true

echo ""
echo "--- ⚡ Live Streaming Logs (Press Ctrl+C to stop) ---"
gcloud beta run jobs executions logs tail "${EXECUTION_NAME}" --region="${REGION}" --project="${PROJECT_ID}"

