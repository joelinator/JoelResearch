---
title: DFlowNovo De Novo Peptide Sequencing
emoji: 🧬
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
pinned: false
license: apache-2.0
short_description: De Novo Peptide Sequencing with PTMs via Flow Matching
---

# 🧬 DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

**DFlowNovo** is a state-of-the-art de novo peptide sequencing framework powered by **Discrete Flow Matching (DFM)** with full support for Post-Translational Modifications (PTMs) and UNIMOD standard formatting.

---

### 🚀 Key Features

- **Discrete Flow Matching**: Continuous-time generative modeling on discrete token sequences for accurate and coherent peptide reconstruction.
- **Top-k Length Bayesian Beam Decoding**: Dynamic length exploration over the top predicted candidate lengths ranked by joint posterior likelihood.
- **Post-Translational Modifications (PTMs)**:
  - Methionine Oxidation: `M(ox)` / `M[UNIMOD:35]` (+15.9949 Da)
  - Cysteine Carboxyamidomethylation: `C(cam)` / `C[UNIMOD:4]` (+57.0215 Da)
  - Asparagine / Glutamine Deamidation: `N(deam)` / `Q(deam)` / `[UNIMOD:7]` (+0.9840 Da)
  - Serine / Threonine / Tyrosine Phosphorylation: `S(ph)` / `T(ph)` / `Y(ph)` / `[UNIMOD:21]` (+79.9663 Da)
  - N-terminal Acetylation: `(+42.01)` / `[UNIMOD:1]` (+42.0106 Da)
  - N-terminal Carbamylation: `(+43.01)` / `[UNIMOD:5]` (+43.0058 Da)
  - N-terminal Ammonia Loss: `(-17.03)` / `[UNIMOD:385]` (-17.0265 Da)
- **Input Formats**: Mascot Generic Format (`.mgf`), Compressed MGF (`.mgz` / `.mgf.gz`).
- **Interactive Outputs**: Real-time tabular output with confidence scores, precursor and calculated peptide masses, mass errors in ppm, and export to CSV/TSV.

---

### 📖 How to Use

1. **Upload** an `.mgz` or `.mgf` file containing MS/MS spectra (or click the example file).
2. **Adjust** inference parameters (number of integration steps, length candidates, guidance scale) if desired.
3. Click **"Run De Novo Sequencing"**.
4. **View and download** the predicted peptide sequences and spectrum metrics in the interactive results table.
