#!/usr/bin/env bash
set -e

# Change directory to project root
cd "$(dirname "$0")"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run Gradio Web UI with public share link
echo "Starting DFlowNovo Gradio Web Interface..."
python deployment/huggingface/app.py --share "$@"
