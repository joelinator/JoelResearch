---
title: DFlowNovo De Novo Peptide Sequencing
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: static
pinned: false
license: apache-2.0
short_description: De Novo Peptide Sequencing with Flow Matching
---

# 🧬 DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/joelinator/dflow-novo)
[![Model Weights](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model%20Weights-green)](https://huggingface.co/joelinator/dflow-novo-model)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**DFlowNovo** is a de novo peptide sequencing system based on **Continuous-Time Markov Chain Discrete Flow Matching (CTMC-DFM)**. By formulating de novo sequencing as continuous probability flux over discrete amino acid sequences and coupling it with precomputed dynamic programming (`KnapsackDP`) reachability bitmasks, DFlowNovo achieves:

- **68.28% Strict Exact Match** on the Nine-Species biological benchmark.
- **56.79% I/L Exact Match** on the ProteomeTools synthetic benchmark.
- **227–256 spectra/second** parallel GPU decoding, and interactive CPU sequencing.
- Native support for **Post-Translational Modifications (PTMs)** with standard UNIMOD annotations (`M(ox)` $\to$ `M[UNIMOD:35]`, `C(cam)` $\to$ `C[UNIMOD:4]`, `N(deam)` $\to$ `N[UNIMOD:7]`, `Q(deam)` $\to$ `Q[UNIMOD:7]`, `S(ph)` $\to$ `S[UNIMOD:21]`, `T(ph)` $\to$ `T[UNIMOD:21]`, `Y(ph)` $\to$ `Y[UNIMOD:21]`).

---

## 🚀 Live Gradio Application Guide

1. **Upload Spectrum File**:
   - Accepts `.mgz`, `.mgf`, `.gz`, or plain text mass spectrometry files.
   - Built-in 1-click test file available under **Clickable Example (.mgz)**.
2. **Model Selection**:
   - `Frozen Production Model (Nine-Species SOTA, 68.28% Strict Exact Match)`: Recommended canonical checkpoint.
   - `PTM Extended Warmstart Model`: PTM-extended checkpoint.
3. **Advanced Parameters**:
   - **Flow Matching Steps ($T$)**: Number of discrete Euler steps (default: 20).
   - **Top-$k$ Length Candidates ($K$)**: Bayesian candidate lengths sampled in parallel (default: 5).
   - **Classifier-Free Guidance ($s$)**: Conditioning strength multiplier (default: 1.5).
   - **Fragment Matching Weight ($\beta$)**: Weight of theoretical $b$/$y$ ion matching in joint posterior scoring (default: 0.5).
   - **Trypsin Cleavage Prior**: Enforce C-terminal Lysine (K) or Arginine (R) preference.
4. **Export Results**:
   - Interactive table with per-spectrum confidence scores, precursor error (ppm), and UNIMOD representations.
   - Download as CSV or TSV with one click.

---

## 📖 Citation

```bibtex
@mastersthesis{gedeon2026dflownovo,
  title={Discrete Flow Matching for De Novo Peptide Sequencing in Mass Spectrometry},
  author={G{\'e}d{\'e}on, Jo{\"e}l},
  school={African Institute for Mathematical Sciences (AIMS South Africa) / InstaDeep},
  year={2026}
}
```
