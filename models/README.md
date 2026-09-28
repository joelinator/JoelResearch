# Pretrained Model Checkpoints

This directory houses pretrained DFlowNovo model weights.

### Production Checkpoint

The canonical frozen production checkpoint is:
- **File Name**: `frozen_production_model.ckpt`
- **Architecture**: 6-layer Transformer Spectrum Encoder, 6-layer Discrete Flow Matching Decoder, Exact Dynamic Programming Knapsack Reachability
- **Training**: Joint Multi-Domain (Nine-Species Biological + ProteomeTools Synthetic) with Length-Weighted Loss ($w(L) \propto \sqrt{L}$)
- **Performance**:
  - Nine-Species Benchmark: **68.28%** Strict Exact Match
  - ProteomeTools Benchmark: **56.79%** I/L Exact Match
  - Throughput: **227 – 256 spectra/second** on single GPU

### Setup Instructions

Place `frozen_production_model.ckpt` directly in this directory:
```bash
models/frozen_production_model.ckpt
```

When running inference via the CLI, Python API, or Gradio web app, DFlowNovo automatically detects this model.
