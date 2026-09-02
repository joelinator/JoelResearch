#!/usr/bin/env bash
# Sync local project to the GCP VM (rsync over gcloud ssh).
#
# Usage:
#   source config/gcp_vm.env
#   bash scripts/gcp_sync_project.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
elif [[ -f "${ROOT_DIR}/config/gcp_vm.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config/gcp_vm.env"
fi

: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_ZONE:?Set GCP_ZONE}"
: "${GCP_VM_NAME:?Set GCP_VM_NAME}"

REMOTE_DIR="${GCP_REMOTE_DIR:-~/dfm-joelresearch}"

gcloud config set project "${GCP_PROJECT}" >/dev/null

echo "Syncing ${ROOT_DIR} -> ${GCP_VM_NAME}:${REMOTE_DIR}"

# Ensure remote directory exists.
gcloud compute ssh "${GCP_VM_NAME}" --zone="${GCP_ZONE}" --command="mkdir -p ${REMOTE_DIR}"

gcloud compute scp --recurse \
  --zone="${GCP_ZONE}" \
  "${ROOT_DIR}/config" \
  "${ROOT_DIR}/scripts" \
  "${ROOT_DIR}/src" \
  "${ROOT_DIR}/tests" \
  "${ROOT_DIR}/requirements.txt" \
  "${ROOT_DIR}/report.md" \
  "${GCP_VM_NAME}:${REMOTE_DIR}/"

echo "Sync complete."
