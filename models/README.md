# Pretrained Model Checkpoints

This directory houses pretrained DFlowNovo model weights.

### Production Checkpoint

The canonical frozen production checkpoint is:
- **File Name**: `frozen_production_model.ckpt` (454 MB lean checkpoint, stripped of AdamW optimizer states)
- **Architecture**: 6-layer Transformer Spectrum Encoder, 6-layer Discrete Flow Matching Decoder, Exact Dynamic Programming Knapsack Reachability
- **Training**: Joint Multi-Domain (Nine-Species Biological + ProteomeTools Synthetic) with Length-Weighted Loss ($w(L) \propto \sqrt{L}$)
- **Performance**:
  - Nine-Species Benchmark: **68.28%** Strict Exact Match
  - ProteomeTools Benchmark: **56.79%** I/L Exact Match
  - Throughput: **227 – 256 spectra/second** on single GPU

### Download Checkpoint
You can download the canonical production checkpoint directly via:
- **Hugging Face Hub**: [`joelinator/dflow-novo-model`](https://huggingface.co/joelinator/dflow-novo-model)
- **GitHub Release**: [v0.2.0](https://github.com/joelinator/JoelResearch/releases/tag/v0.2.0)

```bash
# Using huggingface-cli
huggingface-cli download joelinator/dflow-novo-model frozen_production_model.ckpt --local-dir models/

# Using gh cli
gh release download v0.2.0 -p frozen_production_model.ckpt -D models/
```

When running inference via the CLI, Python API, or Gradio web app, DFlowNovo automatically detects this model.
