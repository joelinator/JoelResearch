#!/usr/bin/env bash
# Create the training VM if it does not exist; start it if stopped.
#
# Required env (or set in config/gcp_vm.env):
#   GCP_PROJECT, GCP_ZONE, GCP_VM_NAME
#
# Usage:
#   source config/gcp_vm.env
#   bash scripts/gcp_create_vm.sh

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

GCP_MACHINE_TYPE="${GCP_MACHINE_TYPE:-a2-highgpu-1g}"
GCP_ACCELERATOR_TYPE="${GCP_ACCELERATOR_TYPE:-nvidia-tesla-a100}"
GCP_ACCELERATOR_COUNT="${GCP_ACCELERATOR_COUNT:-1}"
GCP_BOOT_DISK_GB="${GCP_BOOT_DISK_GB:-500}"
GCP_IMAGE_FAMILY="${GCP_IMAGE_FAMILY:-common-cu124-ubuntu-2204}"
GCP_IMAGE_PROJECT="${GCP_IMAGE_PROJECT:-deeplearning-platform-release}"

gcloud config set project "${GCP_PROJECT}" >/dev/null

if gcloud compute instances describe "${GCP_VM_NAME}" --zone="${GCP_ZONE}" >/dev/null 2>&1; then
  echo "VM '${GCP_VM_NAME}' already exists in ${GCP_ZONE}."
  status="$(gcloud compute instances describe "${GCP_VM_NAME}" --zone="${GCP_ZONE}" --format='get(status)')"
  if [[ "${status}" == "TERMINATED" || "${status}" == "STOPPED" ]]; then
    echo "Starting VM (status was ${status})..."
    gcloud compute instances start "${GCP_VM_NAME}" --zone="${GCP_ZONE}"
  else
    echo "VM status: ${status}"
  fi
else
  echo "Creating VM '${GCP_VM_NAME}' in ${GCP_ZONE}..."
  gcloud compute instances create "${GCP_VM_NAME}" \
    --project="${GCP_PROJECT}" \
    --zone="${GCP_ZONE}" \
    --machine-type="${GCP_MACHINE_TYPE}" \
    --accelerator="type=${GCP_ACCELERATOR_TYPE},count=${GCP_ACCELERATOR_COUNT}" \
    --boot-disk-size="${GCP_BOOT_DISK_GB}GB" \
    --boot-disk-type=pd-balanced \
    --image-family="${GCP_IMAGE_FAMILY}" \
    --image-project="${GCP_IMAGE_PROJECT}" \
    --maintenance-policy=TERMINATE \
    --restart-on-failure \
    --scopes=storage-full,cloud-platform \
    --metadata-from-file=startup-script="${ROOT_DIR}/scripts/gcp_vm_startup.sh" \
    --tags=dfm-train,http-server

  echo "Waiting for VM to become RUNNING..."
  gcloud compute instances wait-until-running "${GCP_VM_NAME}" --zone="${GCP_ZONE}"
fi

EXTERNAL_IP="$(gcloud compute instances describe "${GCP_VM_NAME}" \
  --zone="${GCP_ZONE}" \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)')"
echo "VM ready: ${GCP_VM_NAME} @ ${EXTERNAL_IP} (${GCP_ZONE})"
