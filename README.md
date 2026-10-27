# DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Lightning 2.1+](https://img.shields.io/badge/Lightning-2.1%2B-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![Tests](https://img.shields.io/badge/pytest-68%20passed-success)](tests/)
[![Tutorial Notebook](https://img.shields.io/badge/Jupyter-Tutorial%20Notebook-F37626?logo=jupyter&logoColor=white)](notebooks/dfm_de_novo_tutorial.ipynb)
[![Reproducibility Guide](https://img.shields.io/badge/Reproducibility-Verified%20Pipeline-success)](REPRODUCIBILITY.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

DFlowNovo is a non-autoregressive deep generative framework for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data. The model uses Continuous-Time Markov Chains (CTMC) Discrete Flow Matching coupled with dynamic programming mass knapsack guidance to generate peptide sequences in parallel, achieving competitive state-of-the-art performance and high inference throughput on standard proteomics benchmarks (Nine-Species and HC-PT ProteomeTools).

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
```bash
pytest
```

---

## Quickstart

### 1. Download Datasets
```bash
python scripts/download_dataset.py \
    --repo-id InstaDeepAI/ms_ninespecies_benchmark \
    --splits train validation test \
    --cache-dir data/cache
```

### 2. De Novo Sequencing (Inference)
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

### 3. Model Evaluation
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

**Full Pipeline Orchestration:**
```bash
python scripts/reproduce_canonical_pipeline.py --stage all --dry-run
```

### 5. Publication Figures
```bash
python scripts/generate_publication_figures.py
```

---

## Interactive Tutorial

See [`notebooks/dfm_de_novo_tutorial.ipynb`](notebooks/dfm_de_novo_tutorial.ipynb) for an interactive walkthrough of MS/MS preprocessing, discrete flow matching sampling, and peptide prediction.

---

## Citation

```bibtex
@article{gedeon2026dflownovo,
  title={DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing with Parallel Dynamic Knapsack Guidance},
  author={Gedeon, Joel},
  year={2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
