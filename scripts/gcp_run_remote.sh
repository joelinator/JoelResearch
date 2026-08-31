#!/usr/bin/env bash
# SSH into the VM and run dataset download + training (in tmux).
#
# Usage:
#   export HF_TOKEN=hf_...
#   source config/gcp_vm.env
#   bash scripts/gcp_run_remote.sh            # full pipeline
#   bash scripts/gcp_run_remote.sh --attach   # attach to tmux session
#
# Smoke test on VM:
#   export HF_SUBSET='[:1%]' EPOCHS=2 EVAL_MAX_BATCHES=4
#   bash scripts/gcp_run_remote.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/config/gcp_vm.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config/gcp_vm.env"
fi

: "${GCP_PROJECT:?Set GCP_PROJECT}"
: "${GCP_ZONE:?Set GCP_ZONE}"
: "${GCP_VM_NAME:?Set GCP_VM_NAME}"

REMOTE_DIR="${GCP_REMOTE_DIR:-~/dfm-joelresearch}"
SESSION_NAME="${GCP_TMUX_SESSION:-dfm-train}"

gcloud config set project "${GCP_PROJECT}" >/dev/null

if [[ "${1:-}" == "--attach" ]]; then
  gcloud compute ssh "${GCP_VM_NAME}" --zone="${GCP_ZONE}" -- \
    tmux attach -t "${SESSION_NAME}" || true
  exit 0
fi

if [[ "${1:-}" == "--logs" ]]; then
  gcloud compute ssh "${GCP_VM_NAME}" --zone="${GCP_ZONE}" -- \
    "tail -n 200 -f ${REMOTE_DIR}/train.log"
  exit 0
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "Warning: HF_TOKEN is not set. Public dataset download may be slower or rate-limited."
fi

REMOTE_ENV=(
  "export HF_TOKEN='${HF_TOKEN:-}'"
  "export HF_DATASETS_CACHE='${HF_DATASETS_CACHE:-${REMOTE_DIR}/data/cache}'"
  "export OUTPUT_DIR='${OUTPUT_DIR:-artifacts}'"
  "export BATCH_SIZE='${BATCH_SIZE:-128}'"
  "export EPOCHS='${EPOCHS:-30}'"
  "export LEARNING_RATE='${LEARNING_RATE:-3e-4}'"
  "export WEIGHT_DECAY='${WEIGHT_DECAY:-0.01}'"
  "export NUM_WORKERS='${NUM_WORKERS:-8}'"
  "export TOP_K_PEAKS='${TOP_K_PEAKS:-200}'"
  "export INFERENCE_STEPS='${INFERENCE_STEPS:-50}'"
  "export EVAL_EVERY='${EVAL_EVERY:-2}'"
  "export EVAL_MAX_BATCHES='${EVAL_MAX_BATCHES:-64}'"
  "export DEVICE='cuda'"
)
if [[ -n "${HF_SUBSET:-}" ]]; then
  REMOTE_ENV+=("export HF_SUBSET='${HF_SUBSET}'")
fi

REMOTE_CMD="$(printf '%s; ' "${REMOTE_ENV[@]}")"
REMOTE_CMD+="cd ${REMOTE_DIR} && bash scripts/launch_gcp.sh"

# Kill old session if present, start fresh tmux with logging.
gcloud compute ssh "${GCP_VM_NAME}" --zone="${GCP_ZONE}" --command="
  set -e
  tmux kill-session -t ${SESSION_NAME} 2>/dev/null || true
  tmux new-session -d -s ${SESSION_NAME} \
    \"${REMOTE_CMD} 2>&1 | tee ${REMOTE_DIR}/train.log\"
  echo 'Started tmux session: ${SESSION_NAME}'
  echo 'Logs: ${REMOTE_DIR}/train.log'
"

echo ""
echo "Training started on ${GCP_VM_NAME} in tmux session '${SESSION_NAME}'."
echo "  View logs:  bash scripts/gcp_run_remote.sh --logs"
echo "  Attach:     bash scripts/gcp_run_remote.sh --attach"
echo "  SSH:        gcloud compute ssh ${GCP_VM_NAME} --zone=${GCP_ZONE}"
