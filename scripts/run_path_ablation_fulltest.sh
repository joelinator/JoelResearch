#!/usr/bin/env bash
set -eo pipefail

export CUDA_MEMORY_FRACTION=0.85
export PYTHONUNBUFFERED=1

CHECKPOINT="models/dfm_massive_kb.ckpt"
DATASET="InstaDeepAI/ms_ninespecies_benchmark"
SPLIT="test"
BATCH_SIZE=512
NUM_WORKERS=8

echo "=========================================================="
echo "Starting Full-Test Probability Path Ablation"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset:    $DATASET (split: $SPLIT)"
echo "Time:       $(date)"
echo "=========================================================="

echo "[1/2] Running Uniform Categorical Path Evaluation..."
.venv/bin/python scripts/eval.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-name "$DATASET" \
  --split "$SPLIT" \
  --noising-scheme uniform \
  --guidance-scale 1.8 \
  --eta 0.15 \
  --knapsack-tol-da 0.002 \
  --top-k-lengths 1 \
  --num-samples-per-length 1 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-json artifacts/eval_massivekb_uniform_fulltest.json

echo "[2/2] Running Masked Absorbing Path Evaluation..."
.venv/bin/python scripts/eval.py \
  --checkpoint "$CHECKPOINT" \
  --dataset-name "$DATASET" \
  --split "$SPLIT" \
  --noising-scheme mask \
  --guidance-scale 1.8 \
  --eta 0.15 \
  --knapsack-tol-da 0.002 \
  --top-k-lengths 1 \
  --num-samples-per-length 1 \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --output-json artifacts/eval_massivekb_masked_fulltest.json

echo "=========================================================="
echo "Probability Path Ablation Finished: $(date)"
echo "=========================================================="
