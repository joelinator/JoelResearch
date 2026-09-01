#!/usr/bin/env bash
# Bootstrap a Google Cloud VM (or any Linux host) and start training.
#
# Usage:
#   export HF_TOKEN=hf_...
#   source config/gcp_vm.env   # optional: apply recommended defaults
#   bash scripts/launch_gcp.sh
#
# For smoke tests on a small subset:
#   export HF_SUBSET='[:1%]' EPOCHS=2 EVAL_MAX_BATCHES=4
#   bash scripts/launch_gcp.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/config/gcp_vm.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/config/gcp_vm.env"
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-.venv}"

echo "[1/6] Creating virtual environment at ${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

echo "[2/6] Installing Python dependencies"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${ROOT_DIR}/data/cache}"
export OUTPUT_DIR="${OUTPUT_DIR:-artifacts}"
export DEVICE="${DEVICE:-cuda}"

echo "[3/6] GCP recommendation"
echo "  machine_type=${GCP_MACHINE_TYPE:-a2-highgpu-1g}"
echo "  accelerator=${GCP_ACCELERATOR_TYPE:-nvidia-tesla-a100} x ${GCP_ACCELERATOR_COUNT:-1}"
echo "  boot_disk_gb=${GCP_BOOT_DISK_GB:-500}"

echo "[4/6] Downloading dataset"
python scripts/download_dataset.py \
  --cache-dir "${HF_DATASETS_CACHE}" \
  --splits train validation test \
  ${HF_SUBSET:+--subset "${HF_SUBSET}"}

echo "[5/6] Starting training"
python scripts/train_lightning.py "$@"

echo "[6/6] Done"
