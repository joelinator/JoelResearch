# DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/pytest-75%20passed-success)](tests/)
[![Reproducibility Notebook](https://img.shields.io/badge/Jupyter-Reproducibility%20Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/reproducibility.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DFlowNovo is a non-autoregressive deep generative framework for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data. By formulating sequence generation as a continuous-time Markov chain (CTMC) probability flow on categorical amino acid probability simplices combined with exact dynamic programming knapsack mass guidance, DFlowNovo achieves high sequencing accuracy with fast parallel decoding throughput.

---

## Method Overview

```
                                    ┌────────────────────────────────────────────────────────┐
   MS/MS Spectrum (m/z, Int) ────► │  Transformer Spectrum Encoder + Complementary b/y Ions  │
                                    └──────────────────────────┬─────────────────────────────┘
                                                               │  Latent Representations (h_spec)
                                                               ▼
   Precursor Mass (M_prec, z) ────► ┌────────────────────────────────────────────────────────┐
                                    │  Adaptive LayerNorm (AdaLN) Conditioned Flow Matching   │ ◄─── Time t ∈ [0, 1]
                                    │  SwiGLU Discrete Denoiser Head                         │
                                    └──────────────────────────┬─────────────────────────────┘
                                                               │  Velocity Fields v_t
                                                               ▼
                                    ┌────────────────────────────────────────────────────────┐
                                    │  Exact Dynamic Programming Knapsack Mass Guidance       │ ──► Predicted Peptide
                                    └────────────────────────────────────────────────────────┘
```

1. **Spectrum Encoder**: Projects peak $m/z$, square-root normalized intensities, and theoretical complementary ion masses $(m/z)_{\text{comp}, i} = M_{\text{prec}} - (m/z)_i$ into a continuous latent space using sinusoidal positional embeddings and multi-head self-attention.
2. **Length Predictor Head**: Predicts the peptide length distribution $p(L \mid \mathcal{S})$ over candidate lengths in parallel.
3. **Continuous-Time Markov Chain (CTMC) Probability Flow**: Frames sequence generation as probability velocity interpolation across discrete categorical amino acid simplices:
   $$\frac{\mathrm{d}p_t(x)}{\mathrm{d}t} = \sum_{y \in \mathcal{V}} \left[ q_t(y \to x) p_t(y) - q_t(x \to y) p_t(x) \right]$$
4. **Exact Knapsack Dynamic Programming Guidance**: Prunes unreachable candidate tokens in $\mathcal{O}(1)$ time by verifying whether the remaining mass budget can be physically partitioned by valid residues within instrument tolerance $\tau$.
5. **Hybrid Beam Decoding & ListNet Reranking**: Combines deterministic MAP exploration ($T = 0.0$) with stochastic sampling ($T = 0.7$), scoring candidate sequences via physical MS/MS features ($b/y$ ion series continuity, ppm mass accuracy, and sequence likelihood).

---

## Installation

### 1. Environment Setup
```bash
git clone https://github.com/joelinator/JoelResearch.git
cd JoelResearch

python3 -m venv .venv
source .venv/bin/activate

pip install -e .
```

### 2. Verify Installation
Run the test suite:
```bash
pytest
```

---

## Quickstart

### 1. Download Datasets
Download the MassIVE-KB dataset shards or the Nine-Species benchmark:
```bash
# MassIVE-KB dataset shards
python scripts/download_massive_kb.py --split train --num-shards 5

# Nine-Species benchmark
python scripts/download_dataset.py \
    --repo-id InstaDeepAI/ms_ninespecies_benchmark \
    --splits train validation test \
    --cache-dir data/cache
```

### 2. Training
Train DFlowNovo on MassIVE-KB or benchmark datasets:
```bash
python scripts/train.py \
    --train-split train \
    --valid-split validation \
    --batch-size 128 \
    --epochs 30 \
    --lr 5e-5 \
    --output-dir checkpoints/
```

### 3. De Novo Sequencing (Inference)
Run inference on an experimental MGF file or test spectra:
```bash
python scripts/infer.py \
    --checkpoint models/dfm_massive_kb.ckpt \
    --input data/sample_spectra.mgf \
    --output predictions.csv \
    --batch-size 128 \
    --num-steps 20 \
    --knapsack-filter \
    --knapsack-tol-da 0.5
```

### 4. Evaluation & Reranking
Evaluate a checkpoint on the held-out test split with KnapsackDP guidance and ListNet candidate reranking:
```bash
python scripts/eval_beam_rerank.py \
    --checkpoint models/dfm_massive_kb.ckpt \
    --dataset-name InstaDeepAI/ms_ninespecies_benchmark \
    --split test \
    --top-k-lengths 3 \
    --num-samples-per-length 2 \
    --batch-size 64
```

---

## Reproducibility Notebook

An end-to-end interactive notebook is available at [`notebooks/reproducibility.ipynb`](notebooks/reproducibility.ipynb):
- **Mini-Training Demo**: Loads a small chunk of MassIVE-KB and runs a 2-epoch training loop to demonstrate the flow matching optimization.
- **Checkpoint Loading**: Loads pre-trained model weights.
- **Batched Inference**: Evaluates 10,000 spectra using exact KnapsackDP guidance, hybrid beam decoding ($T \in \{0.0, 0.7\}$), and ListNet candidate reranking.
- **Metrics Calculation**: Reports peptide accuracy (isobaric I/L match), amino acid recall/F1, precursor mass match, and decoding throughput (spectra/second).

---

## Repository Structure

```text
dfm-joelresearch/
├── src/                        # Core library
│   ├── config/                 # Hyperparameters and physical constants
│   ├── data/                   # MS/MS data parsers and spectrum collators
│   ├── eval/                   # Standardized de novo sequencing evaluation metrics
│   ├── flow_matching/          # CTMC discrete flow schedulers and reverse solvers
│   ├── inference/              # KnapsackDP mass guidance and ListNet candidate reranker
│   ├── model/                  # Spectrum encoder and AdaLN-Zero flow decoder
│   └── train/                  # Training loops, loss functions, and checkpoint I/O
├── scripts/                    # Command-line entry points
│   ├── download_dataset.py     # Benchmark dataset downloader
│   ├── download_massive_kb.py  # MassIVE-KB shard downloader
│   ├── eval.py                 # Standard evaluation runner
│   ├── eval_beam_rerank.py     # Hybrid beam decoding and ListNet evaluation
│   ├── generate_paper_figures.py # Figure generation script
│   ├── infer.py                # Standalone MGF / mzML inference script
│   └── train.py                # Unified model training script
├── notebooks/                  # Reproducibility notebook
│   └── reproducibility.ipynb   # End-to-end training demo, 10k inference, and metrics
├── models/                     # Model weights
│   └── dfm_massive_kb.ckpt     # Pretrained checkpoint
├── tests/                      # Unit and integration test suite
└── pyproject.toml              # Build system and dependencies
```

---

## Citation

```bibtex
@inproceedings{asa2026dflownovo,
  title={DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing with Exact Knapsack Dynamic Programming Guidance},
  author={Asa, Jo{\"e}l G{\'e}d{\'e}on},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2026}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
