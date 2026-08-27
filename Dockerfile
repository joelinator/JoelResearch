FROM google/cloud-sdk:slim

WORKDIR /workspace

# Install system utilities, Python and Git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    git \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install lightweight CPU PyTorch wheel (~150MB vs ~5GB CUDA wheels) since orchestrator runs on CPU,
# while heavy GPU computing is executed remotely on the A100 VM.
RUN pip install --no-cache-dir --break-system-packages torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --break-system-packages -r requirements.txt

COPY . .

CMD ["python3", "run_job.py"]
