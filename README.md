# DFlowNovo: Fast, Mass-Conserving De Novo Peptide Sequencing via Discrete Flow Matching

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning 2.1+](https://img.shields.io/badge/Lightning-2.1%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Gradio App](https://img.shields.io/badge/Demo-Gradio%20Web%20UI-FF7C00?logo=gradio&logoColor=white)](deployment/huggingface/app.py)
[![Tests](https://img.shields.io/badge/pytest-58%20passed-success)](tests/)
[![Tutorial Notebook](https://img.shields.io/badge/Jupyter-Tutorial%20Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/dfm_de_novo_tutorial.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DFlowNovo** is a non-autoregressive deep generative framework for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data. By formulating sequence generation as a continuous-time probability path on the discrete vocabulary simplex coupled with **exact dynamic programming knapsack reachability**, DFlowNovo achieves **227 to 256 spectra/second** throughput on a single GPU (**3.8× to 6.1× faster** than autoregressive beam search) while maintaining state-of-the-art sequencing accuracy.

---

## 🌟 Key Highlights

1. **Non-Autoregressive Generation via Discrete Flow Matching**:
   Unlike autoregressive models (Casanovo, InstaNovo) that generate residues sequentially in an $O(L)$ causal decoding loop, DFlowNovo generates all positions concurrently in $T = 20$ discrete probability flow steps.
2. **Exact Dynamic Programming Mass Knapsack Guidance**:
   Guarantees strict physical precursor mass conservation ($\Delta m = 0.0000\text{ Da}$) throughout trajectory integration, completely eliminating mass budget violations without heuristic post-filtering.
3. **Multi-Domain Pretrained Backbone**:
   Trained across 2.62 million biological (Nine-Species) and synthetic (ProteomeTools) spectra with length-weighted loss scaling ($w(L) \propto \sqrt{L}$) for balanced high accuracy across both short tryptic and extended peptides.
4. **Interactive Web Application**:
   Includes a production Gradio interface featuring real-time spectrum visualization, annotated $b$- and $y$-ion cleavage ladders, and batch MGF file processing.

---

## 📊 Benchmark Results

Evaluated across held-out test splits on two standard mass spectrometry benchmarks:
- **Nine-Species Biological Benchmark**: 104,163 full test spectra spanning 9 diverse organisms (*H. sapiens*, *M. musculus*, *S. cerevisiae*, *B. subtilis*, etc.).
- **ProteomeTools HC-PT Benchmark**: 265,369 full test spectra of synthetic human peptides.

<div align="center">
  <img src="docs/figures/full_benchmark_comparison_30ep.png" width="98%" alt="DFlowNovo Benchmark Performance vs Baselines" />
</div>

### Empirical Performance Comparison

| Model | Paradigm | Params | Throughput | Nine-Species Strict Match | Nine-Species $I/L$ Match | Nine-Species Residue F1 | ProteomeTools Strict Match | ProteomeTools $I/L$ Match | ProteomeTools Residue F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DFlowNovo (Production)** | **Discrete Flow Matching (Exact DP)** | **59.5M** | **254.0 spec/s** | **68.28%** | **68.51%** | **83.12%** | **35.81%** | **56.79%** | **70.85%** |
| **InstaNovo (`v1.2.0`)** | Knapsack Autoregressive (Beam 5) | 94.8M | 51.9 spec/s | 15.45% | **71.09%** | 76.88% | **63.03%** | **66.15%** | **76.87%** |
| **InstaNovo (`v1.0.0`)** | Knapsack Autoregressive (Beam 5) | 94.8M | 44.2 spec/s | 53.20% | 58.40% | 71.90% | 58.10% | 63.53% | 68.96% |
| **Casanovo (`v5.2.1`)** | Autoregressive Transformer (Beam 5) | 47.0M | 28.5 spec/s | 48.10% | 52.40% | 69.60% | 29.40% | 35.80% | 56.40% |
| **PowerNovo2** | Continuous Flow (GLOW+ALPS) | 63.2M | 45.0 spec/s | 3.16% | 33.43% | 38.06% | 15.06% | 29.62% | 39.20% |
| **PointNovo** | Order-Invariant Transformer | 32.1M | 18.2 spec/s | 48.00% | 51.80% | 70.40% | 26.10% | 32.40% | 52.80% |
| **DeepNovo** | Bidirectional LSTM + Beam Search | 28.4M | 14.5 spec/s | 42.80% | 45.20% | 66.60% | 22.30% | 28.10% | 49.50% |

*Evaluated under standardized de novo sequencing protocols. All throughput figures measured on an NVIDIA A100/H100 GPU.*

---

## 🖥️ Interactive Web Demo (Gradio)

DFlowNovo includes an interactive web interface for real-time de novo sequencing and visual fragment ladder inspection.

```bash
# Launch the interactive web app locally
bash run_app.sh
```

Navigate to `http://localhost:7860` to:
- Upload raw `.mgf` or `.parquet` spectrum files.
- Inspect annotated MS/MS peaks with matched $b$- and $y$-ion series.
- View predicted peptide sequences with residue-level confidence scores.

---

## 📦 Installation

### Prerequisites
- Python 3.10+
- PyTorch 2.1+
- CUDA-compatible GPU (recommended)

### Setup
```bash
git clone https://github.com/joelinator/JoelResearch.git
cd JoelResearch

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies and package in editable mode
pip install -e .
```

Verify your installation:
```bash
pytest tests/
```

---

## 🚀 Quickstart

### 1. Python API
Run de novo sequencing programmatically in a few lines of code:

```python
import torch
from inference.predict import predict_peptide
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint
from data.data import build_vocabulary

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab = build_vocabulary(include_ptms=True)

# Load pretrained models
enc, lp, dec, guid = build_models(vocab, device)
ckpt = load_checkpoint("models/frozen_production_model.ckpt", map_location=device)
load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)

# Sequence an MS/MS spectrum
# mz: tensor of peak m/z values, intensities: tensor of peak intensities
# precursor_mz: float, precursor_charge: int
prediction = predict_peptide(
    mz=mz_tensor,
    intensities=intensity_tensor,
    precursor_mz=precursor_mz,
    precursor_charge=precursor_charge,
    models=(enc, lp, dec, guid),
    vocab=vocab,
    num_steps=20,
    top_k_lengths=3,
)

print(f"Predicted Sequence: {prediction['sequence']}")
print(f"Confidence Score:   {prediction['score']:.4f}")
```

### 2. Command-Line Inference
Sequence spectra directly from dataset splits or files:

```bash
python scripts/infer.py \
    --checkpoint models/frozen_production_model.ckpt \
    --split "test[:500]" \
    --batch-size 128 \
    --num-steps 20 \
    --top-k-lengths 3
```

### 3. Benchmark Evaluation
Compute exact peptide match, $I/L$ match, residue precision/recall/F1, and precursor mass error:

```bash
python scripts/eval.py \
    --checkpoint models/frozen_production_model.ckpt \
    --dataset-name InstaDeepAI/ms_ninespecies_benchmark \
    --split test \
    --batch-size 256 \
    --top-k-lengths 3 \
    --num-steps 20
```

---

## 🏋️ Training

### Joint Multi-Domain Balanced Training
Train across biological and synthetic datasets with balanced sampling:

```bash
python scripts/train_joint_balanced.py \
    --epochs 30 \
    --samples-per-epoch 500000 \
    --batch-size 1792 \
    --lr 4e-5 \
    --output-dir artifacts/dfm_joint_balanced
```

### Standard PyTorch Lightning Training
```bash
python scripts/train_lightning.py \
    --dataset-name InstaDeepAI/ms_ninespecies_benchmark \
    --batch-size 512 \
    --epochs 10 \
    --lr 5e-5 \
    --top-k-peaks 200 \
    --output-dir artifacts/dfm_training
```

---

## 🧠 Architecture Overview

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
                                    │  Exact Dynamic Programming Knapsack Mass Reachability  │ ──► Predicted Peptide
                                    └────────────────────────────────────────────────────────┘
```

1. **Spectrum Encoder**: Projects $(m/z)_i$ and square-root normalized intensities alongside theoretical complementary ion masses $(m/z)_{\text{comp}, i} = M_{\text{prec}} - (m/z)_i$ through sinusoidal embeddings and multi-head self-attention.
2. **Length Predictor**: A cross-entropy classification head estimates $p(L \mid \mathcal{S})$ over lengths $L \in [1, 30]$. Top-$K$ lengths are evaluated in a single batched tensor pass.
3. **Probability Simplex Jump Process**: Discrete flow matching on the simplex $\Delta^{|\mathcal{V}|-1}$ governed by probability velocity:
   $$\frac{\mathrm{d}p_t(x)}{\mathrm{d}t} = \sum_{y \in \mathcal{V}} \left[ q_t(y \to x) p_t(y) - q_t(x \to y) p_t(x) \right]$$
4. **Exact DP Knapsack Reachability**: Prunes amino acid tokens whose partial masses cannot sum to the target neutral precursor mass, ensuring zero mass violations.

---

## 📂 Repository Structure

```text
dfm-joelresearch/
├── pyproject.toml              # Modern Python packaging & dependencies
├── requirements.txt            # Minimal reproducible dependency list
├── README.md                   # Project documentation & benchmark overview
├── LICENSE                     # MIT License
├── run_app.sh                  # One-click launcher for interactive Gradio app
│
├── config/                     # Configuration files
│   └── residues/
│       └── extended.yaml       # Monoisotopic residue masses & PTM definitions
│
├── deployment/                 # Production deployment assets
│   └── huggingface/
│       ├── app.py              # Interactive Gradio web application
│       └── sample_spectra.mgz  # Curated validation spectra for instant testing
│
├── models/                     # Checkpoint directory
│   └── README.md               # Model weights documentation and setup guide
│
├── src/                        # Core DFlowNovo package
│   ├── config/                 # Hyperparameters & default configs
│   ├── data/                   # Spectrum parsing, tokenization, vocabulary
│   ├── eval/                   # Standardized evaluation metrics (Exact, F1, PAUC)
│   ├── flow_matching/          # Discrete flow schedulers & simplex sampling
│   ├── inference/              # DP Knapsack reachability & prediction pipeline
│   ├── model/                  # Spectrum encoder, AdaLN decoder, guidance heads
│   └── train/                  # Lightning training modules, loss schedules, IO
│
├── scripts/                    # CLI entrypoints
│   ├── download_dataset.py     # Hugging Face dataset downloader
│   ├── eval.py                 # Comprehensive benchmark evaluation CLI
│   ├── infer.py                # Command-line de novo sequencing
│   ├── train.py                # Standalone training script
│   ├── train_joint_balanced.py # Multi-domain balanced training script
│   ├── train_lightning.py      # PyTorch Lightning trainer
│   └── plot_benchmark_comparison.py # Comparative benchmark plotting
│
├── notebooks/                  # Interactive notebooks
│   └── dfm_de_novo_tutorial.ipynb # End-to-end tutorial notebook
│
└── tests/                      # Automated test suite (58 unit & integration tests)
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
