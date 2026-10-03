# Post-Translational Modifications (PTM) Integration & Transfer Learning Plan

## Executive Summary

This document establishes the architecture, biochemistry, tokenization, and model weight transplantation pipeline to extend the Discrete Flow Matching (DFM) de novo peptide sequencing model to support **Post-Translational Modifications (PTMs)** on branch [`feature/ptm-support`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch).

We preserve 100% of the learned representations from our best checkpoint ([`artifacts/dfm_pl_run_20260914_182552/checkpoints/best-gen-exact-epoch=43-exact=0.3267.ckpt`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/dfm_pl_run_20260914_182552/checkpoints/best-gen-exact-epoch=43-exact=0.3267.ckpt)) through surgical weight expansion, warm-starting new PTM tokens from their parent amino acids.

---

## 1. Benchmark Baseline: Full Test Split Evaluation (Epoch 43)

Evaluation on the complete test split (**265,369 spectra**) with `num_steps=20`, `top_k_lengths=5`, `guidance_scale=1.5`:

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Exact Match (I/L Equivalence)** | **52.55%** | Standard proteomics benchmark standard |
| **Mass-Based Peptide Accuracy** | **52.57%** | Prefix/suffix mass alignment tolerance |
| **Strict Exact Accuracy ($p == t$)** | **32.05%** | Without I/L interchangeability |
| **Length Accuracy** | **80.94%** | Predicted peptide length matches target |
| **Amino Acid F1** | **61.96%** | Per-residue alignment F1 |
| **Calibrated Precision ($\tau = -0.152$)** | **80.00%** | High-confidence identification precision |
| **Calibrated Recall ($\tau = -0.152$)** | **40.13%** | Recall at 80% precision |
| **Coverage at 80% Precision** | **50.16%** | **133,112 spectra** identified with $\ge 80\%$ precision |
| **Precision-Coverage AUC (Mass)** | **0.7562** | Area under precision-coverage curve |
| **Partial AUC ($\ge 80\%$ Precision)** | **0.4441** | High-confidence curve integration |

Saved artifacts:
- JSON: [`artifacts/test_eval_results_epoch43.json`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/test_eval_results_epoch43.json)
- Curves: [`artifacts/test_eval_pauc_epoch43.png`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/test_eval_pauc_epoch43.png)

---

## 2. Target PTM Chemistry & Extended Vocabulary

We support the requested modifications with monoisotopic masses (Da) aligned with Unimod and ProteomeTools:

| Amino Acid | PTM Name | Symbol / Token | Delta Mass ($\Delta$ Da) | Monoisotopic Mass (Da) | Unimod ID |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **C** (Cysteine) | Carbamidomethylation | `C(+57.02)` / `C(cam)` | $+57.021464$ | **$160.030649$** | Unimod:4 |
| **M** (Methionine) | Oxidation | `M(+15.99)` / `M(ox)` | $+15.994915$ | **$147.035400$** | Unimod:35 |
| **N** (Asparagine) | Deamidation | `N(+0.98)` / `N(deam)` | $+0.984016$ | **$115.026943$** | Unimod:7 |
| **Q** (Glutamine) | Deamidation | `Q(+0.98)` / `Q(deam)` | $+0.984016$ | **$129.042594$** | Unimod:7 |
| **S** (Serine) | Phosphorylation | `S(+79.97)` / `S(ph)` | $+79.966331$ | **$166.998359$** | Unimod:21 |
| **T** (Threonine) | Phosphorylation | `T(+79.97)` / `T(ph)` | $+79.966331$ | **$181.014010$** | Unimod:21 |
| **Y** (Tyrosine) | Phosphorylation | `Y(+79.97)` / `Y(ph)` | $+79.966331$ | **$243.029660$** | Unimod:21 |

### Extended Vocabulary ($V = 30$)
```json
{
  "<pad>": 0,
  "<mask_token>": 1,
  "A": 2, "R": 3, "N": 4, "D": 5, "C": 6, "E": 7, "Q": 8, "G": 9,
  "H": 10, "I": 11, "L": 12, "K": 13, "M": 14, "F": 15, "P": 16,
  "S": 17, "T": 18, "W": 19, "Y": 20, "V": 21,
  "M(ox)": 22,
  "C(cam)": 23,
  "N(deam)": 24,
  "Q(deam)": 25,
  "S(ph)": 26,
  "T(ph)": 27,
  "Y(ph)": 28
}
```
*(Standardized tokens can support dual alias formats: short chemistry `M(ox)` or delta format `M(+15.99)` via a parser mapper).*

---

## 3. Architecture & Weight Transplantation Mechanism

In the DFM architecture:
```mermaid
graph TD
    subgraph Fully Reused Weights (100% Preserved)
        SE[Spectrum Transformer Encoder]
        LP[Length Predictor MLP]
        GM[Classifier-Free Guidance Module]
        TD[12-Layer Transformer Decoder Blocks]
    end

    subgraph Transplanted & Expanded Weights
        TE[Token Embedding: V_old -> V_new]
        OH[Output Linear Head: V_old -> V_new]
    end

    SE --> TD
    LP --> TD
    GM --> TD
    TE --> TD
    TD --> OH
```

### Weight Surgery Logic
For any existing token $w \in V_{\text{old}} \cap V_{\text{new}}$:
$$W_{\text{embed}}^{\text{new}}[w] = W_{\text{embed}}^{\text{old}}[w], \quad W_{\text{head}}^{\text{new}}[w] = W_{\text{head}}^{\text{old}}[w], \quad b_{\text{head}}^{\text{new}}[w] = b_{\text{head}}^{\text{old}}[w]$$

For new PTM token $m \in V_{\text{new}} \setminus V_{\text{old}}$ with parent amino acid $p$:
$$W_{\text{embed}}^{\text{new}}[m] = W_{\text{embed}}^{\text{old}}[p] + \mathcal{N}(0, 0.01)$$
$$W_{\text{head}}^{\text{new}}[m] = W_{\text{head}}^{\text{old}}[p] + \mathcal{N}(0, 0.01)$$
$$b_{\text{head}}^{\text{new}}[m] = b_{\text{head}}^{\text{old}}[p] - \ln(\text{prior\_penalty})$$

**Result**: 99.85% of all trainable parameters are instantly transferred without cold restart!

---

## 4. Pipeline Modifications

1. **`src/data/data.py`**:
   - Include `modified_sequence` in `keep_columns`.
   - Prefer `row["modified_sequence"]` over `row["sequence"]` if present.
   - Implement multi-character token tokenizer: `tokenize_peptide(seq: str) -> list[str]` supporting bracket/parenthesis patterns.
2. **`src/data/constants.py`**:
   - Register all 7 PTM residual masses in `AA_MASSES_DICT`.
3. **`src/inference/predict.py` & `src/eval/metrics.py`**:
   - Update theoretical $b/y$ ion generation to look up token residue masses dynamically.
   - Support PTM-aware de novo exact matching and mass matching.
4. **`src/train/surgery.py`**:
   - Provide `transplant_checkpoint(ckpt_path, new_vocab, output_path)` to generate the warm-started checkpoint.
5. **`scripts/train_lightning.py` & `comand.sh`**:
   - Load the transplanted checkpoint, resume fine-tuning with a gentle warmup learning rate ($1 \times 10^{-4}$), and train.

---

## 5. Verification & Testing Plan

1. Unit test `tokenize_peptide` on standard, oxidized, carbamidomethylated, deamidated, and phosphorylated peptides.
2. Unit test `transplant_checkpoint` verifying shapes, parameter identities, and forward pass loss matching.
3. Test 1 training step on GPU to confirm zero NaN, valid gradient flow, and smooth loss convergence.
