#!/usr/bin/env bash
set -euo pipefail

VENV_PYTHON="/home/joelgedeon_aims_ac_za/dfm-joelresearch/.venv/bin/python"
CKPT="artifacts/canonical_reproduction/stage3_stratified/checkpoints/best-hcpt-stratified-epoch=04-exact=0.3799.ckpt"
OUT_DIR="artifacts/eval_beam_rerank_hcpt_base"
mkdir -p "${OUT_DIR}"

echo "=========================================================================="
echo "STARTING BENCHMARK 1/2: NINE-SPECIES FULL TEST SPLIT (ZERO-SHOT TRANSFER)"
echo "=========================================================================="
${VENV_PYTHON} scripts/eval_hcpt_base_rerank.py \
  --checkpoint "${CKPT}" \
  --dataset-name "InstaDeepAI/ms_ninespecies_benchmark" \
  --split "test" \
  --num-samples -1 \
  --batch-size 256 \
  --num-workers 4 \
  --top-k-lengths 5 \
  --num-samples-per-length 2 \
  --temperature 0.5 \
  --num-steps 20 \
  --scheduler "cosine" \
  --knapsack-tol-da 0.5 \
  --output-json "${OUT_DIR}/ninespecies_full_test_results.json" \
  2>&1 | tee "${OUT_DIR}/ninespecies_full_test.log"

echo ""
echo "=========================================================================="
echo "STARTING BENCHMARK 2/2: HC-PT FULL TEST SPLIT (IN-DOMAIN EVALUATION)"
echo "=========================================================================="
${VENV_PYTHON} scripts/eval_hcpt_base_rerank.py \
  --checkpoint "${CKPT}" \
  --dataset-name "InstaDeepAI/ms_proteometools" \
  --split "test" \
  --num-samples -1 \
  --batch-size 256 \
  --num-workers 4 \
  --top-k-lengths 5 \
  --num-samples-per-length 2 \
  --temperature 0.5 \
  --num-steps 20 \
  --scheduler "cosine" \
  --knapsack-tol-da 0.5 \
  --output-json "${OUT_DIR}/hcpt_full_test_results.json" \
  2>&1 | tee "${OUT_DIR}/hcpt_full_test.log"

echo ""
echo "=========================================================================="
echo "ALL EVALUATIONS COMPLETED SUCCESSFULLY!"
echo "=========================================================================="
