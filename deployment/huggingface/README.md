---
title: DFlowNovo De Novo Peptide Sequencing
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: Fast De Novo Peptide Sequencing with Flow Matching
---

# 🧬 DFlowNovo: De Novo Peptide Sequencing

Fast *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) using discrete flow matching over Continuous-Time Markov Chains (CTMC).

DFlowNovo performs fast parallel decoding across discrete amino acid states with exact dynamic programming reachability constraints.

## 🚀 How to Use

1. **Upload Spectrum File**:
   - Accepts `.mgz`, `.mgf`, `.gz`, or `.txt` mass spectrometry files.
   - Use the built-in sample spectrum under **Clickable Example (.mgz)** for a quick test.
2. **Select Model Checkpoint**:
   - `Production Model (HC-PT + Nine-Species)`: Canonical production checkpoint.
   - `MassIVE-KB Pretrained Model (Human 28M)`: MassIVE-KB pretrained checkpoint.
   - `PTM Extended Model`: Pretrained checkpoint with extended post-translational modification vocabulary.
3. **Run Sequencing**:
   - Click **Run De Novo Sequencing** to generate peptide predictions, calculated masses, and confidence scores.
   - Download results as CSV or TSV.

## 📖 Citation

```bibtex
@mastersthesis{gedeon2026dflownovo,
  title={Discrete Flow Matching for De Novo Peptide Sequencing in Mass Spectrometry},
  author={G{\'e}d{\'e}on, Jo{\"e}l},
  school={African Institute for Mathematical Sciences (AIMS South Africa)},
  year={2026}
}
```
