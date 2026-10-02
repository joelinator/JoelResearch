#!/usr/bin/env bash
set -e

# Load environment variables from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"
JOB_NAME="agent-peptide-sota-runner"
REGION="us-central1"
IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT_ID}/cloud-run-source-deploy/${JOB_NAME}:latest"
CPU="4"
MEMORY="8Gi"
TIMEOUT="86400"

echo "============================================================"
echo "🚀 [1/3] Building container with Docker BuildKit cache..."
echo "============================================================"
gcloud builds submit --config=cloudbuild.yaml --project="${PROJECT_ID}" .

echo "============================================================"
echo "📦 [2/3] Deploying Cloud Run Job: ${JOB_NAME}"
echo "============================================================"
gcloud run jobs deploy "${JOB_NAME}" \
    --image "${IMAGE_URI}" \
    --region "${REGION}" \
    --tasks 1 \
    --max-retries 0 \
    --cpu "${CPU}" \
    --memory "${MEMORY}" \
    --task-timeout "${TIMEOUT}" \
    --set-env-vars="GCP_PROJECT_ID=${PROJECT_ID},A100_VM_NAME=${A100_VM_NAME},A100_VM_ZONE=${A100_VM_ZONE},GITHUB_REPO_URL=${GITHUB_REPO_URL},GITHUB_TOKEN=${GITHUB_TOKEN},GOOGLE_GENAI_USE_VERTEXAI=true,VERTEXAI_PROJECT=${PROJECT_ID},VERTEXAI_LOCATION=${REGION},GOOGLE_CLOUD_PROJECT=${PROJECT_ID}" \
    --project="${PROJECT_ID}"

echo "============================================================"
echo "⚡ [3/3] Triggering Cloud Run Job Execution"
echo "============================================================"
gcloud run jobs execute "${JOB_NAME}" --region "${REGION}" --project="${PROJECT_ID}"
