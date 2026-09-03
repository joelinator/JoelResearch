#!/usr/bin/env bash
python3 scripts/eval.py \
	--checkpoint artifacts/dfm_pl_run_20260902_184635/checkpoints/last.ckpt \
	--split validation \
	--batch-size 7024 \
	--num-steps 20 \
	--device cuda \
	--top-k-lengths 3 \
	--output-json  artifacts/dfm_pl_run_20260902_184635/eval_val_epoch23.json \

