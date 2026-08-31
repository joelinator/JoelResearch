#!/usr/bin/env bash
# End-to-end: create/start VM, sync code, run download + training.
#
# Prerequisites:
#   gcloud auth login
#   export HF_TOKEN=hf_...
#   source config/gcp_vm.env
#
# Usage:
#   bash scripts/gcp_provision.sh
#   HF_SUBSET='[:1%]' EPOCHS=2 bash scripts/gcp_provision.sh   # smoke test

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

echo "=== [1/4] Create or start VM ==="
bash "${ROOT_DIR}/scripts/gcp_create_vm.sh"

echo "=== [2/4] Wait for SSH (startup script may still be running) ==="
sleep 30

echo "=== [3/4] Sync project to VM ==="
bash "${ROOT_DIR}/scripts/gcp_sync_project.sh"

echo "=== [4/4] Launch download + training on VM ==="
bash "${ROOT_DIR}/scripts/gcp_run_remote.sh"

echo "Provision complete."
