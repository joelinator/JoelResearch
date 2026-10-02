#!/usr/bin/env bash
set -e

# Load project ID from .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

PROJECT_ID="${GCP_PROJECT_ID:-$(gcloud config get-value project)}"

if [ -z "$PROJECT_ID" ]; then
  echo "❌ Error: GCP_PROJECT_ID is not set."
  exit 1
fi

echo "============================================================"
echo "🔧 Configuring GCP APIs and Permissions for Project: $PROJECT_ID"
echo "============================================================"

# 1. Enable Required Google Cloud APIs
echo "➡️ [1/3] Enabling required Google Cloud APIs..."
gcloud services enable \
    run.googleapis.com \
    artifactregistry.googleapis.com \
    cloudbuild.googleapis.com \
    compute.googleapis.com \
    aiplatform.googleapis.com \
    iam.googleapis.com \
    iap.googleapis.com \
    logging.googleapis.com \
    monitoring.googleapis.com \
    cloudtrace.googleapis.com \
    storage.googleapis.com \
    --project="${PROJECT_ID}"

# 2. Retrieve Project Number and Service Accounts
echo "➡️ [2/3] Retrieving project details..."
PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")
COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
CLOUDBUILD_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"

echo "Project Number: ${PROJECT_NUMBER}"
echo "Compute / Cloud Run SA: ${COMPUTE_SA}"
echo "Cloud Build SA: ${CLOUDBUILD_SA}"

# 3. Assign Complete IAM Permissions
echo "➡️ [3/3] Assigning IAM roles..."

# Roles for Cloud Run Runtime & VM Compute Service Account
RUN_ROLES=(
  "roles/compute.instanceAdmin.v1"
  "roles/compute.osAdminLogin"
  "roles/compute.osLogin"
  "roles/compute.viewer"
  "roles/iam.serviceAccountUser"
  "roles/iap.tunnelResourceAccessor"
  "roles/aiplatform.user"
  "roles/storage.objectAdmin"
  "roles/storage.admin"
  "roles/logging.logWriter"
  "roles/monitoring.metricWriter"
  "roles/cloudtrace.agent"
  "roles/artifactregistry.writer"
  "roles/artifactregistry.reader"
)

for role in "${RUN_ROLES[@]}"; do
  echo "Assigning $role to $COMPUTE_SA..."
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${COMPUTE_SA}" \
    --role="${role}" \
    --condition=None || true
done

# Roles for Cloud Build Service Account
BUILD_ROLES=(
  "roles/cloudbuild.builds.builder"
  "roles/run.admin"
  "roles/iam.serviceAccountUser"
  "roles/artifactregistry.writer"
  "roles/artifactregistry.admin"
  "roles/storage.admin"
  "roles/logging.logWriter"
)

for role in "${BUILD_ROLES[@]}"; do
  echo "Assigning $role to $CLOUDBUILD_SA..."
  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CLOUDBUILD_SA}" \
    --role="${role}" \
    --condition=None || true
done

echo "============================================================"
echo "✅ All comprehensive APIs and IAM permissions have been assigned."
echo "============================================================"
