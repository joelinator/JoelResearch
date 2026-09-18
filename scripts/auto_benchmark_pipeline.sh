#!/usr/bin/env bash
set -eo pipefail

PID=84448
echo "Auto Benchmark Pipeline: Monitoring PID ${PID} (Casanovo)..."

while kill -0 "$PID" 2>/dev/null; do
    sleep 20
done

echo "=== Casanovo benchmark finished at $(date) ==="
echo "=== Starting PowerNovo2 benchmark across Nine-Species (104k) and HC-PT (50k) ==="

PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_casanovo_powernovo2_benchmark.py --model powernovo2 --batch-size 128

echo "=== PowerNovo2 benchmark finished at $(date) ==="
echo "=== Generating 4-way comparative publication figures ==="

.venv/bin/python scripts/plot_four_way_benchmark.py

echo "=== Benchmark Pipeline Complete! All artifacts written to artifacts/benchmark_external/ ==="
