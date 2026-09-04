#!/usr/bin/env bash
# De Novo Generative Evaluation Script for DFM Peptide Sequencing
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

if [[ -f "${ROOT_DIR}/.env" ]]; then
  # shellcheck disable=SC1091
  source "${ROOT_DIR}/.env"
fi

# Locate Python environment
if [[ -d "${ROOT_DIR}/.venv" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python3"
else
  PYTHON_BIN="python3"
fi

DEFAULT_CKPT="artifacts/dfm_pl_run_20260903_092757/checkpoints/last.ckpt"
CHECKPOINT="${1:-${CHECKPOINT:-$DEFAULT_CKPT}}"
SPLIT="${SPLIT:-validation}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-4096}"
NUM_WORKERS="${EVAL_NUM_WORKERS:-12}"
NUM_STEPS="${INFERENCE_STEPS:-20}"
TOP_K_LENGTHS="${TOP_K_LENGTHS:-3}"
LENGTH_BEAM_ALPHA="${LENGTH_BEAM_ALPHA:-0.01}"
GUIDANCE_SCALE="${EVAL_GUIDANCE_SCALE:-1.5}"
CALIBRATE_SUBSET_BATCHES="${CALIBRATE_SUBSET_BATCHES:-5}"
TARGET_PRECISION="${TARGET_PRECISION:-0.80}"
OUTPUT_JSON="${OUTPUT_JSON:-artifacts/dfm_pl_run_20260903_092757/eval_${SPLIT}_cfg1.5_calibrated.json}"
SAVE_PLOT="${SAVE_PLOT:-artifacts/dfm_pl_run_20260903_092757/pauc_${SPLIT}_cfg1.5.png}"
DEVICE="${DEVICE:-cuda}"

echo "=========================================================="
echo "  DFM De Novo Generative Evaluation & Score Calibration"
echo "=========================================================="
echo "Checkpoint:          ${CHECKPOINT}"
echo "Split:               ${SPLIT}"
echo "Batch Size:          ${BATCH_SIZE}"
echo "Num Workers:         ${NUM_WORKERS}"
echo "Inference Steps:     ${NUM_STEPS}"
echo "Top-K Lengths:       ${TOP_K_LENGTHS}"
echo "Beam Alpha (m/z):    ${LENGTH_BEAM_ALPHA}"
echo "Guidance Scale:      ${GUIDANCE_SCALE}"
echo "Calibrate Batches:   ${CALIBRATE_SUBSET_BATCHES}"
echo "Target Precision:    ${TARGET_PRECISION}"
echo "Output JSON:         ${OUTPUT_JSON}"
echo "Save Plot:           ${SAVE_PLOT}"
echo "Device:              ${DEVICE}"
echo "=========================================================="

"${PYTHON_BIN}" scripts/eval.py \
    --checkpoint "${CHECKPOINT}" \
    --split "${SPLIT}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --num-steps "${NUM_STEPS}" \
    --top-k-lengths "${TOP_K_LENGTHS}" \
    --length-beam-alpha "${LENGTH_BEAM_ALPHA}" \
    --guidance-scale "${GUIDANCE_SCALE}" \
    --calibrate-subset-batches "${CALIBRATE_SUBSET_BATCHES}" \
    --target-precision "${TARGET_PRECISION}" \
    --device "${DEVICE}" \
    --output-json "${OUTPUT_JSON}" \
    --save-plot "${SAVE_PLOT}"
