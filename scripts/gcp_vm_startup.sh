#!/usr/bin/env bash
# Runs once on VM first boot (GCP metadata startup-script).
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update -y
apt-get install -y \
  git \
  python3 \
  python3-pip \
  python3-venv \
  build-essential \
  tmux \
  htop \
  rsync

# Deep Learning VM images already ship NVIDIA drivers; verify when present.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

mkdir -p /opt/dfm
chown -R "$(whoami):$(whoami)" /opt/dfm 2>/dev/null || true

echo "DFM VM startup complete at $(date -Is)" > /var/log/dfm-startup.log
