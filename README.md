# DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning 2.1+](https://img.shields.io/badge/Lightning-2.1%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/pytest-68%20passed-success)](tests/)
[![Tutorial Notebook](https://img.shields.io/badge/Jupyter-Tutorial%20Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/dfm_de_novo_tutorial.ipynb)
[![Reproducibility Guide](https://img.shields.io/badge/Reproducibility-Verified%20Pipeline-success)](REPRODUCIBILITY.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DFlowNovo is a non-autoregressive deep generative framework for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data. By formulating sequence generation as a continuous-time probability flow over categorical amino acid states coupled with dynamic programming mass knapsack guidance, DFlowNovo achieves high sequencing accuracy with 227–256 spectra/second throughput (4.3× to 4.9× faster than autoregressive beam search).

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
                                    │  Parallel Dynamic Knapsack Vectorized Mass Filter       │ ──► Predicted Peptide
                                    └────────────────────────────────────────────────────────┘
```

1. **Spectrum Encoder**: Maps peak $m/z$, square-root normalized intensities, and theoretical complementary ion masses $(m/z)_{\text{comp}, i} = M_{\text{prec}} - (m/z)_i$ into a continuous latent representation using sinusoidal positional embeddings and multi-head self-attention.
2. **Length Predictor Head**: Prior to non-autoregressive decoding, an auxiliary cross-entropy head predicts the peptide length distribution $p(L \mid \mathcal{S})$ over $L \in [1, 30]$. The top-$K$ candidate lengths are evaluated concurrently in a single batched tensor pass.
3. **Continuous-Time Markov Chain (CTMC) Probability Flow**: Sequence generation is framed as probability velocity interpolation on the discrete vocabulary simplex:
   $$\frac{\mathrm{d}p_t(x)}{\mathrm{d}t} = \sum_{y \in \mathcal{V}} \left[ q_t(y \to x) p_t(y) - q_t(x \to y) p_t(x) \right]$$
4. **Dynamic Programming Knapsack Mass Guidance**: Enforces precursor neutral mass conservation by pruning partial candidate sequences that cannot physically sum to the target precursor mass within tolerance $\tau \le 1.0\text{ Da}$.
5. **Peak-Evidence Unmasking & Detailed Balance**: Schedules position unmasking by factoring in observed fragment peak evidence and applies stochastic error correction ($\eta$) to prevent premature token commitment.

---

## Benchmark Results

Evaluated across standard benchmarks under identical evaluation protocols on a single NVIDIA GPU:
- **Nine-Species Benchmark**: $N = 104,163$ full test spectra across 9 organisms (*H. sapiens*, *M. musculus*, *S. cerevisiae*, *B. subtilis*, etc.).
- **HC-PT ProteomeTools Benchmark**: $N = 265,369$ full test spectra of synthetic human peptides.

| Model Architecture | Paradigm | Params | Throughput | Nine-Species Strict Match | Nine-Species I/L Match | Nine-Species Residue F1 | HC-PT Strict Match | HC-PT I/L Match | HC-PT Residue F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DFlowNovo (Production)** | **Discrete Flow Matching (DP Knapsack)** | **59.5M** | **227–256 spec/s** | **68.28%** | **68.44%** | **83.01%** | **35.81%** | **56.79%** | **69.62%** |
| **DFlowNovo (30ep Baseline)** | Discrete Flow Matching (Exact DP) | 59.5M | 174 spec/s | 65.08% | 65.29% | 81.80% | 34.84% | 55.88% | 69.74% |
| **DFlowNovo (8ep Joint)** | Discrete Flow Matching (Fast Knapsack) | 59.5M | 185 spec/s | 64.92% | 65.07% | 81.97% | 34.93% | 55.95% | 70.04% |
| **InstaNovo (`v1.2.0`)** | Knapsack Autoregressive (MassIVE-KB) | 94.8M | 52 spec/s | 15.45% | **71.09%** | 76.88% | **63.03%** | **66.15%** | **76.87%** |
| **InstaNovo (`v1.0.0`)** | Knapsack Autoregressive (ACPT Base) | 94.8M | 44 spec/s | 53.20% | 58.40% | 71.90% | 58.10% | 63.53% | 68.96% |
| **Casanovo (`v5.2.1`)** | Autoregressive Transformer | 47.0M | 28 spec/s | 48.10% | 52.40% | 69.60% | 29.40% | 35.80% | 56.40% |
| **PowerNovo2** | Continuous Normalizing Flow (GLOW) | 63.2M | 45 spec/s | 3.16% | 33.43% | 38.06% | 15.06% | 29.62% | 39.20% |
| **PointNovo** | Order-Invariant Transformer | 32.1M | 18 spec/s | 48.00% | 51.80% | 70.40% | 26.10% | 32.40% | 52.80% |
| **DeepNovo** | Bidirectional LSTM + Beam Search | 28.4M | 14 spec/s | 42.80% | 45.20% | 66.60% | 22.30% | 28.10% | 49.50% |

*Prefix matching evaluated at $\pm 0.1\text{ Da}$ residue mass and $\pm 0.5\text{ Da}$ cumulative precursor tolerance.*

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
Download benchmark datasets from Hugging Face:
```bash
python scripts/download_dataset.py \
    --repo-id InstaDeepAI/ms_ninespecies_benchmark \
    --splits train validation test \
    --cache-dir data/cache
```

### 2. De Novo Sequencing (Inference)
Run inference on an MGF or Parquet file:
```bash
python scripts/infer.py \
    --checkpoint artifacts/dfm_joint_balanced_30ep/checkpoints/dfm_balanced_best.ckpt \
    --input data/sample.mgf \
    --output predictions.csv \
    --batch-size 128 \
    --num-steps 20 \
    --top-k-lengths 3 \
    --strategy confidence \
    --guidance-scale 1.8 \
    --knapsack-filter \
    --knapsack-tol-da 1.0 \
    --peak-evidence \
    --eta 0.15
```

#### Inference Flags & Strategies
| Flag | Default | Description |
| :--- | :---: | :--- |
| `--strategy` | `confidence` | Unmasking schedule: `confidence` (entropy-guided) or `random` (uniform). |
| `--guidance-scale` | `1.0` | Classifier-free mass guidance scale ($s > 1.0$ emphasizes precursor conditioning). |
| `--knapsack-filter` / `--no-knapsack-filter` | Enabled | Prunes unreachable amino acid tokens via dynamic programming mass table. |
| `--knapsack-tol-da` | `1.0` | Maximum precursor mass tolerance in Daltons for knapsack reachability. |
| `--peak-evidence` / `--no-peak-evidence` | Enabled | Boosts decoding confidence at positions supported by observed $b/y$ ion peaks. |
| `--eta` | `0.15` | Stochastic correction noise parameter for detailed-balance reverse sampling. |
| `--num-steps` | `20` | Number of Euler flow integration steps $T$. |
| `--top-k-lengths` | `3` | Number of candidate peptide lengths evaluated per spectrum. |
| `--temperature` | `1.0` | Softmax temperature for token sampling. |

### 3. Model Evaluation
Evaluate a checkpoint on benchmark splits with standardized residue precision, recall, and exact match metrics:
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

**Single / Multi-GPU PyTorch Lightning Training:**
```bash
python scripts/train_lightning.py \
    --dataset-name InstaDeepAI/ms_ninespecies_benchmark \
    --batch-size 512 \
    --epochs 30 \
    --lr 5e-5 \
    --precision bf16-mixed \
    --top-k-peaks 200 \
    --scheduler-type wsd \
    --output-dir artifacts/dfm_training
```

**Stage-by-Stage Curriculum Reproduction:**
```bash
# Stage 1: HC-PT Pre-Training
python scripts/train_lightning.py --dataset-name InstaDeepAI/ms_proteometools --no-use-ptm --epochs 20

# Stage 2: Weight Surgery & Vocabulary Expansion
python -m src.train.surgery --checkpoint artifacts/stage1/best.ckpt --output artifacts/stage2/expanded.ckpt

# Stage 3: Length-Stratified Fine-Tuning
python scripts/train_hcpt_length_stratified.py --init-checkpoint artifacts/stage2/expanded.ckpt

# Stage 4: Joint Balanced Multi-Domain Fine-Tuning
python scripts/train_joint_balanced.py --epochs 30 --batch-size 1792 --lr 4e-5
```

Alternatively, orchestrate the entire pipeline:
```bash
python scripts/reproduce_canonical_pipeline.py --stage all --dry-run
```

### 5. Publication Figures
Regenerate all figures from cached evaluation artifacts:
```bash
python scripts/generate_publication_figures.py
```

---

## Interactive Tutorial

An interactive walkthrough is provided in [`notebooks/dfm_de_novo_tutorial.ipynb`](notebooks/dfm_de_novo_tutorial.ipynb), covering:
- MS/MS spectrum preprocessing and complementary ion computation.
- Discrete flow trajectory visualization on the probability simplex.
- Dynamic programming knapsack filtering.
- End-to-end peptide prediction and residue alignment.

---

## Repository Structure

```text
dfm-joelresearch/
├── pyproject.toml              # Build system, dependencies, and pytest configuration
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview and reproduction instructions
├── REPRODUCIBILITY.md          # End-to-end training curriculum and verification guide
├── LICENSE                     # MIT License
│
├── config/                     # Configuration definitions
│   └── residues/
│       └── extended.yaml       # Monoisotopic residue masses & PTM definitions
│
├── src/                        # Core library
│   ├── config/                 # Hyperparameters and physical constants
│   ├── data/                   # Data parsers, peak filters, and length samplers
│   ├── flow_matching/          # Discrete flow matching schedulers and samplers
│   ├── model/                  # Spectrum encoder, AdaLN decoder, and knapsack guidance
│   ├── inference/              # DP knapsack reachability and beam decoding
│   ├── train/                  # Lightning module, loss functions, and weight surgery
│   └── eval/                   # Standardized evaluation metrics and PAUC curves
│
├── scripts/                    # CLI entry points
│   ├── infer.py                # Standalone de novo inference CLI
│   ├── train.py                # Standalone PyTorch training entry point
│   ├── train_lightning.py      # PyTorch Lightning multi-GPU trainer
│   ├── train_hcpt_length_stratified.py # Length-stratified domain adaptation
│   ├── train_joint_balanced.py # Interleaved multi-domain joint balanced trainer
│   ├── eval.py                 # Evaluation pipeline with standardized metrics
│   ├── download_dataset.py     # Dataset download utility
│   ├── prepare_benchmark_mgf.py# Benchmark MGF conversion utility
│   ├── reproduce_canonical_pipeline.py # End-to-end curriculum orchestrator
│   ├── run_casanovo_powernovo2_benchmark.py # External baseline runner
│   ├── benchmark_compomics_generalization.py # Cross-project generalization benchmark
│   ├── run_comprehensive_ablation_study.py # Ablation experiments
│   ├── generate_publication_figures.py # Master figure generator from artifacts
│   ├── generate_architecture_diagram.py # Architecture visualization
│   ├── generate_ctmc_dynamics_figures.py # Flow dynamics visualization
│   └── deploy_huggingface.py   # Hugging Face Spaces deployment
│
├── docs/                       # Technical documentation
│   ├── ARCHITECTURE_DESIGN.md  # Detailed architectural specifications
│   ├── BENCHMARK_OPTIMIZATIONS.md # Decoding optimizations and mathematical derivations
│   ├── CODEBASE_GUIDE.md       # Implementation guide and algorithm reference
│   └── figures/                # Publication figures (300 DPI)
│
├── notebooks/                  # Interactive notebooks
│   └── dfm_de_novo_tutorial.ipynb
│
└── tests/                      # Unit and integration tests (68 tests, 100% passing)
```

---

## Citation

If you build upon this work, please cite:

```bibtex
@article{gedeon2026dflownovo,
  title={DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing with Parallel Dynamic Knapsack Guidance},
  author={Gedeon, Joel},
  year={2026}
}
```

### Key References

1. **Discrete Flow Matching**: Campbell et al., *Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design*, ICML 2024. [arXiv:2402.04997](https://arxiv.org/abs/2402.04997).
2. **InstaNovo**: Eloff et al., *InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments*, Nature Machine Intelligence, 2025. [doi:10.1038/s42256-025-01009-4](https://doi.org/10.1038/s42256-025-01009-4).
3. **Casanovo**: Yilmaz et al., *De novo mass spectrometry peptide sequencing with a transformer model*, Nature Machine Intelligence, 2022. [doi:10.1038/s42256-022-00566-z](https://doi.org/10.1038/s42256-022-00566-z).
4. **PowerNovo2**: Petrovskiy et al., *PowerNovo2: A generative flow-based approach to non-autoregressive de novo peptide sequencing*, PLOS Computational Biology, 2026.
5. **Nine-Species Benchmark**: Tran et al., *De novo peptide sequencing by deep learning*, PNAS, 2017. [doi:10.1073/pnas.1705697114](https://doi.org/10.1073/pnas.1705697114).
6. **ProteomeTools Benchmark**: Zolg et al., *Building ProteomeTools based on a complete synthetic human proteome*, Nature Methods, 2017. [doi:10.1038/nmeth.4153](https://doi.org/10.1038/nmeth.4153).

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
