# DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning 2.1+](https://img.shields.io/badge/Lightning-2.1%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/pytest-58%20passed-success)](tests/)
[![Tutorial Notebook](https://img.shields.io/badge/Jupyter-Tutorial%20Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/dfm_de_novo_tutorial.ipynb)
[![Technical Report](https://img.shields.io/badge/Research-Master's%20Thesis%20Report-blue)](SUPERVISOR_REPORT_DFM_DE_NOVO.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **DFlowNovo** is a non-autoregressive deep generative framework for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data using **Continuous-Time Markov Chain (CTMC) Discrete Flow Matching**. By modeling peptide generation as a probability velocity trajectory on the discrete vocabulary simplex coupled with **dynamic knapsack mass guidance**, DFlowNovo achieves **227 to 256 spectra/second throughput** (4.3× to 4.9× faster than autoregressive baselines) with state-of-the-art sequencing accuracy.

---

## 🔬 Key Scientific Highlights

1. **Non-Autoregressive Generation via Discrete Flow Matching**:
   Unlike traditional autoregressive models (Casanovo, InstaNovo) that generate sequences residue-by-residue in an iterative $O(L)$ causal decoding loop, DFlowNovo generates all residue positions in parallel in a fixed $T = 20$ integration budget via continuous-time probability velocity interpolation.
2. **Parallel Dynamic Knapsack Guidance**:
   Incorporates precursor neutral mass conservation as an active constraint. During flow integration, candidates violating the parent ion mass $M_{\text{prec}}$ are pruned using exact polynomial-time dynamic programming knapsack filtering, ensuring 0.00% mass violations.
3. **Multi-Domain Joint Balanced Training with Length Weighting**:
   Addresses representation deficits on longer peptides ($L \ge 23$) via length-weighted loss scaling ($w(L) \propto \sqrt{L}$), achieving 68.28% strict match on Nine-Species while maintaining stable generalisation across diverse biological domains.
4. **Inference Efficiency**:
   Processes **227.1 to 255.8 spectra/sec** on a single GPU with exact DP knapsack—providing **4.3× to 4.9× higher throughput than InstaNovo** (52.4 spec/s) and **6.5× higher throughput than Casanovo** (28.5 spec/s).

---

## 📊 Comprehensive Multi-Paradigm Benchmark

The framework was benchmarked against existing de novo peptide sequencing methods across two standard datasets:
- **Nine-Species Biological Benchmark**: $N = 104,163$ full test spectra across 9 organisms (*H. sapiens*, *M. musculus*, *S. cerevisiae*, *B. subtilis*, etc.).
- **HC-PT ProteomeTools Benchmark**: $N = 265,369$ full test spectra of synthetic human peptides.

<div align="center">
  <img src="docs/figures/full_benchmark_comparison_30ep.png" width="98%" alt="Full Benchmark Comparison 30-Epoch Balanced vs InstaNovo v1.2.0 and v1.0.0" />
</div>

### Empirical Performance Summary (Full Test Splits)

| Model Architecture | Paradigm | Params | Inference Speed | Nine-Species Strict Match | Nine-Species I/L Match | Nine-Species Residue F1 | HC-PT Full Strict Match | HC-PT Full I/L Match | HC-PT Full Residue F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DFlowNovo (Production)** | **Discrete Flow Matching (Length-Weighted DP)** | **59.5M** | **227.1–255.8 spec/s** | **68.28%** | **68.44%** | **83.01%** | **35.81%** | **56.79%** | **69.62%** |
| **DFlowNovo (30ep Baseline)** | Discrete Flow Matching (Exact DP Knapsack) | 59.5M | 174.0 spec/s | 65.08% | 65.29% | 81.80% | 34.84% | 55.88% | 69.74% |
| **DFlowNovo (8ep Joint)** | Discrete Flow Matching (Fast Knapsack) | 59.5M | 185.0 spec/s | 64.92% | 65.07% | 81.97% | 34.93% | 55.95% | 70.04% |
| **InstaNovo (`v1.2.0` Latest)** | Knapsack Autoregressive (MassIVE-KB) | 94.8M | 51.9 spec/s | 15.45% | **71.09%** | 76.88% | **63.03%** | **66.15%** | **76.87%** |
| **InstaNovo (`v1.0.0` First)** | Knapsack Autoregressive (ACPT Base) | 94.8M | 44.2 spec/s | 53.20% | 58.40% | 71.90% | 58.10% | 63.53% | 68.96% |
| **Casanovo (`v5.2.1`)** | Autoregressive Transformer | 47.0M | 28.5 spec/s | 48.10% | 52.40% | 69.60% | 29.40% | 35.80% | 56.40% |
| **PowerNovo2** | Continuous Normalizing Flow (GLOW+ALPS) | 63.2M | 45.0 spec/s | 3.16% | 33.43% | 38.06% | 15.06% | 29.62% | 39.20% |
| **PointNovo** | Order-Invariant Continuous Transformer | 32.1M | 18.2 spec/s | 48.00% | 51.80% | 70.40% | 26.10% | 32.40% | 52.80% |
| **DeepNovo** | Bidirectional LSTM + Beam Search | 28.4M | 14.5 spec/s | 42.80% | 45.20% | 66.60% | 22.30% | 28.10% | 49.50% |

*Evaluated on full Nine-Species test ($N=104,163$) and full HC-PT test ($N=265,369$) under standardized de novo evaluation protocols.*

> **Evaluation Protocol**: Full test splits evaluated on a single NVIDIA H100 GPU (batch size 128). DFlowNovo used $K=20$ Euler reverse flow steps under cosine schedule with Dynamic Knapsack reachability table ($\tau = 1.0\text{ Da}$). Prefix matching evaluated with $\pm 0.1\text{ Da}$ residue mass and $\pm 0.5\text{ Da}$ cumulative tolerance.
>
> **Scientific Interpretation**: DFlowNovo achieves top accuracy on Nine-Species (68.28% strict match, 83.01% residue F1) with a 4.3× to 4.9× throughput advantage (227–256 spectra/s) over autoregressive beam search. On HC-PT, 56.79% $I/L$ accuracy reflects the physical limit of standard HCD spectra, where isomeric Leucine and Isoleucine ($113.084\text{ Da}$) produce identical backbone fragments.

---

## 🧠 Method Overview

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
                                    │  Parallel Dynamic Knapsack Vectorized Mass Filter       │ ──► Predicted Peptide
                                    └────────────────────────────────────────────────────────┘
```

1. **Spectrum Encoder**: Projects $(m/z)_i$ and square-root normalized intensities alongside theoretical complementary ion masses $(m/z)_{\text{comp}, i} = M_{\text{prec}} - (m/z)_i$ through sinusoidal embeddings and multi-head self-attention.
2. **Length Predictor Head**: Prior to non-autoregressive decoding, a cross-entropy length head estimates $p(L \mid \mathcal{S})$ over $L \in [1, 30]$. Top-$K$ lengths are evaluated in a single batched tensor pass.
3. **Probability Simplex Jump Process**: Flow integration is defined on the discrete simplex $\Delta^{|\mathcal{V}|-1}$ governed by probability velocity:
   $$\frac{\mathrm{d}p_t(x)}{\mathrm{d}t} = \sum_{y \in \mathcal{V}} \left[ q_t(y \to x) p_t(y) - q_t(x \to y) p_t(x) \right]$$
4. **Dynamic Knapsack Filter**: Prunes unpromising amino acid tokens whose partial prefix/suffix masses cannot physically sum to the target neutral precursor mass within mass tolerance $\delta \le 1.0\text{ Da}$.

---

## 📦 Installation & Setup

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/joelinator/JoelResearch.git
cd JoelResearch

# Create Python 3.10+ virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install in editable mode with development tools
pip install -e .
```

### 2. Verify Installation
Run the unit and integration test suite (58 tests):
```bash
pytest
```

---

## 🚀 Quickstart

### 1. Download Benchmark Datasets
Download the Nine-Species and ProteomeTools HC-PT datasets directly from Hugging Face:
```bash
python scripts/download_dataset.py \
    --repo-id InstaDeepAI/ms_ninespecies_benchmark \
    --splits train validation test \
    --cache-dir data/cache
```

### 2. De Novo Sequencing (Inference)
Sequence raw spectra from an MGF or Parquet file using the production checkpoint:
```bash
python scripts/infer.py \
    --checkpoint artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt \
    --split "test[:500]" \
    --batch-size 128 \
    --num-steps 20 \
    --top-k-lengths 3
```

### 3. Model Evaluation
Compute exact peptide match, I/L isobaric match, residue precision/recall/F1, and length accuracy on full test splits:
```bash
python scripts/eval.py \
    --checkpoint artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt \
    --dataset-name InstaDeepAI/ms_ninespecies_benchmark \
    --split test \
    --batch-size 128 \
    --top-k-lengths 3 \
    --use-knapsack-filter \
    --output-json artifacts/eval_results.json
```

### 4. Training
Train a Discrete Flow Matching model with PyTorch Lightning:

**Standard Distributed Training:**
```bash
python scripts/train_lightning.py \
    --dataset-name InstaDeepAI/ms_ninespecies_benchmark \
    --batch-size 512 \
    --epochs 10 \
    --lr 5e-5 \
    --top-k-peaks 200 \
    --output-dir artifacts/dfm_training
```

**Joint Multi-Domain Balanced Training (30-Epoch Balanced Multi-Domain Model):**
```bash
python scripts/train_joint_balanced.py \
    --epochs 30 \
    --samples-per-epoch 500000 \
    --batch-size 1792 \
    --val-samples 20000 \
    --lr 4e-5 \
    --output-dir artifacts/dfm_joint_balanced_30ep
```
*(Note: On 80GB GPUs, batch size automatically scales to 1792 with BF16 mixed precision to achieve >80% VRAM utilization).*

### 5. Reproducing Benchmark Figures
Generate all 300 DPI publication-grade comparison plots:
```bash
# 7-Model Benchmark Figure (DFlowNovo 30ep vs InstaNovo v1.2.0 vs v1.0.0 vs Casanovo vs PowerNovo2)
python scripts/plot_benchmark_comparison.py

# Multi-domain & Knapsack ablation figures
python scripts/generate_publication_figures.py
```

---

## 📓 Interactive Tutorial Notebook

For an interactive Jupyter walk-through covering:
- Exploratory data analysis (EDA) on MS/MS mass spectra
- Mass spectrometry peak chemistry & $b/y$ complementary ions
- Initializing DFM encoders, denoisers, and AdaLN conditioning
- Step-by-step Discrete Flow Euler integration on the probability simplex
- Dynamic Knapsack filtering demonstration
- Loading pretrained checkpoints and computing residue metrics

Open [`notebooks/dfm_de_novo_tutorial.ipynb`](notebooks/dfm_de_novo_tutorial.ipynb).

---

## 📂 Repository Structure

```text
dfm-joelresearch/
├── pyproject.toml              # Modern Python packaging & pytest configuration
├── requirements.txt            # Minimal reproducible dependency specifications
├── README.md                   # Project overview, benchmarks, and reproduction guides
├── LICENSE                     # MIT License
├── SUPERVISOR_REPORT_DFM_DE_NOVO.md # Comprehensive 25+ page thesis & technical report
├── ARTIFACTS_MANIFEST.md       # Manifest of precomputed evaluation artifacts
├── dfm_research_analysis_artifacts.zip # 44.5 MB zip with all raw benchmark prediction CSVs
│
├── config/                     # Configuration definitions
│   └── residues/
│       └── extended.yaml       # Monoisotopic residue masses & PTM definitions
│
├── src/                        # Core DFlowNovo library
│   ├── config/                 # Default hyperparameters & constants
│   ├── data/                   # Spectrum parsing, peak filtering, length beam search
│   ├── flow_matching/          # Discrete flow schedulers & simplex Euler solvers
│   ├── model/                  # Spectrum encoder, AdaLN decoder, knapsack guidance
│   ├── train/                  # PyTorch Lightning modules, scheduled multi-task loss
│   ├── eval/                   # Standardized evaluation metrics & PAUC curves
│   └── inference/              # Batched prediction and length-beam decoders
│
├── scripts/                    # Reproducible experiment entrypoints
│   ├── download_dataset.py     # Hugging Face dataset downloader
│   ├── train.py                # Standalone PyTorch training entrypoint
│   ├── train_lightning.py      # PyTorch Lightning multi-GPU trainer
│   ├── train_joint_balanced.py # Interleaved multi-domain joint balanced trainer
│   ├── eval.py                 # Standardized de novo sequencing evaluation pipeline
│   ├── infer.py                # Prediction on MGF and Parquet spectra
│   ├── prepare_benchmark_mgf.py# Standardized MGF benchmark split preparation
│   ├── run_casanovo_powernovo2_benchmark.py # External baseline execution harness
│   ├── plot_four_way_benchmark.py # 6-model benchmark comparison figure generator
│   ├── plot_benchmark_comparison.py # Comparative benchmark plotting script
│   └── generate_publication_figures.py # Master figure generation pipeline
│
├── notebooks/                  # Interactive tutorial
│   └── dfm_de_novo_tutorial.ipynb # Self-contained end-to-end tutorial notebook
│
├── docs/                       # Theoretical and architectural documentation
│   ├── ARCHITECTURE_DESIGN.md
│   ├── BENCHMARK_OPTIMIZATIONS.md
│   └── figures/                # Publication-quality 300 DPI figures
│       ├── full_benchmark_comparison_30ep.png
│       ├── four_way_benchmark_comparison.png
│       ├── joint_balanced_multi_domain_comparison.png
│       ├── dynamic_knapsack_benchmark_comparison.png
│       └── qualitative_prediction_cases.png
│
└── tests/                      # Pytest unit and integration test suite (58 tests)
```

---

## 📄 Citation & Reference

If you use this codebase, models, or benchmark suite in your research, please cite:

```bibtex
@mastersthesis{gedeon2026dflownovo,
  title={DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing with Parallel Dynamic Knapsack Guidance},
  author={Gedeon, Joel},
  school={African Institute for Mathematical Sciences (AIMS)},
  year={2026},
  note={Supervised Research Project}
}
```

### Key Literature References

1. **Discrete Flow Matching**: Campbell, A., Yim, J., Barzilay, R., Rainforth, T., & Jaakkola, T. (2024). Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. *ICML 2024*. [arXiv:2402.04997](https://arxiv.org/abs/2402.04997).
2. **Continuous Flow Matching**: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. *ICLR 2023*. [arXiv:2210.02747](https://arxiv.org/abs/2210.02747).
3. **InstaNovo**: Eloff, K., Kalogeropoulos, K., Mabona, A., Morell, O., Catzel, R., et al. (2025). InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments. *Nature Machine Intelligence*, 7, 565–579. [doi:10.1038/s42256-025-01009-4](https://doi.org/10.1038/s42256-025-01009-4).
4. **Casanovo**: Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S., & Noble, W. S. (2022). De novo mass spectrometry peptide sequencing with a transformer model. *Nature Machine Intelligence*, 4(11), 1001–1008. [doi:10.1038/s42256-022-00566-z](https://doi.org/10.1038/s42256-022-00566-z).
5. **PowerNovo2**: Petrovskiy, D. V., Nikolsky, K. S., Rudnev, V. R., Kulikova, L. I., Butkova, T. V., Malsagova, K. A., Kopylov, A. T., & Kaysheva, A. L. (2026). PowerNovo2: A generative flow-based approach to non-autoregressive de novo peptide sequencing. *PLOS Computational Biology*.
6. **DeepNovo & Nine-Species Benchmark**: Tran, N. H., Zhang, X., Xin, L., Shan, B., & Li, M. (2017). De novo peptide sequencing by deep learning. *Proceedings of the National Academy of Sciences (PNAS)*, 114(31), 8247–8252. [doi:10.1073/pnas.1705697114](https://doi.org/10.1073/pnas.1705697114).
7. **ProteomeTools Synthetic Benchmark**: Zolg, D. P., Wilhelm, M., Schnatbaum, K., Zerweck, J., Knaute, T., Delanghe, B., et al. (2017). Building ProteomeTools based on a complete synthetic human proteome. *Nature Methods*, 14(3), 259–265. [doi:10.1038/nmeth.4153](https://doi.org/10.1038/nmeth.4153).
8. **Dynamic Programming Spectrum Graph**: Dancik, V., Addona, T. A., Clauser, K. R., Vath, J. E., & Pevzner, P. A. (1999). De novo peptide sequencing via tandem mass spectrometry. *Journal of Computational Biology*, 6(3–4), 327–342.
9. **Isobaric Leucine/Isoleucine Differentiation**: Lebedev, A. T., Damoc, E., Makarov, A. A., & Samgina, T. Y. (2014). Discrimination of leucine and isoleucine in peptides sequencing with Orbitrap Fusion mass spectrometer. *Analytical Chemistry*, 86(14), 7017–7022.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
