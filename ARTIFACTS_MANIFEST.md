# Research Artifacts Archive Manifest

**Archive File:** `dfm_research_analysis_artifacts.zip`  
**Size:** 44.53 MB (Compressed) / 156.58 MB (Uncompressed)  
**Total Files:** 84  
**Integrity:** Verified (Zero CRC checksum errors)  
**Purpose:** Enables offline analysis, plotting, and metric verification across 369,532 test spectra without requiring GPU hardware, model checkpoints, or re-running inference/training.

---

## 1. Quick Extraction

To unpack all artifacts into the standard directory structure:
```bash
unzip dfm_research_analysis_artifacts.zip
```

---

## 2. Archive Contents Breakdown

### A. Full Test Split Predictions (CSV)
Contains per-spectrum predictions across all **369,532 held-out test spectra**:

1. **Nine-Species Full Test Split ($N = 104,163$ spectra)**:
   - `artifacts/eval_joint_balanced/joint_ninespecies_full_test_preds.csv`: Predictions from the Balanced Joint model (8 epochs).
   - `artifacts/eval_sequential_finetune/base_ninespecies_full_test_preds.csv`: Predictions from the Base model (HC-PT trained only).
   - `artifacts/eval_sequential_finetune/finetuned_ninespecies_full_test_preds.csv`: Predictions from the Finetuned model (Nine-Species Phase 2).
   - `artifacts/instanovo_eval/ninespecies_full_test_preds.csv`: InstaNovo baseline predictions (`v1.2.0`, 5-beam search). Includes `predictions`, `log_probs`, individual candidate beams (`predictions_beam_0` to `4`), `delta_mass_ppm`, `scan_number`, and `spectrum_id`.

2. **HC-PT ProteomeTools Full Test Split ($N = 265,369$ spectra)**:
   - `artifacts/eval_joint_balanced/joint_hcpt_full_test_preds.csv`: Predictions from the Balanced Joint model.
   - `artifacts/eval_sequential_finetune/base_hcpt_full_test_preds.csv`: Predictions from the Base model.
   - `artifacts/eval_sequential_finetune/finetuned_hcpt_full_test_preds.csv`: Predictions from the Finetuned model (illustrating catastrophic forgetting).
   - `artifacts/instanovo_eval/hcpt_full_test_preds.csv`: InstaNovo baseline predictions (`v1.2.0`, 5-beam search).

*DFM CSV schema:* `target,prediction,score,exact_match,mass_match,target_length,pred_length`  
*InstaNovo CSV schema:* `experiment_name,evidence_index,scan_number,spectrum_id,precursor_mz,precursor_charge,predictions,log_probs,predictions_beam_0..4,delta_mass_ppm`

---

### B. Benchmark Evaluation Metrics (JSON)
Complete summary metrics (strict exact match, I/L exact match, residue precision/recall/F1, length accuracy, precursor mass match, coverage @ 80% precision):

- **Frozen Production Model (Length-Weighted)**:
  - Canonical Checkpoint: `models/frozen_production_model.ckpt` (pointing to `artifacts/dfm_length_weighted_10ep/checkpoints/best-joint-gen-exact-epoch=01-exact=0.4746.ckpt`)
  - `artifacts/dfm_length_weighted_10ep/hcpt_50k_evaluation.json`: HC-PT 50k stratified metrics (35.81% strict exact match, 56.79% I/L match).
  - `artifacts/dfm_length_weighted_10ep/ninespecies_50k_evaluation.json`: Nine-Species 50k stratified metrics (68.28% strict exact match).
  - `artifacts/reports/LENGTH_WEIGHTED_50K_EVALUATION_REPORT.md`: Comprehensive 50,000-spectra length-weighted benchmark report.
  - `artifacts/reports/REFINED_DECODING_50K_EVALUATION_REPORT.md`: Full 50,000-spectra decoding refinements and physical error analysis.
- **30-Epoch Baseline Multi-Domain Model**:
  - `artifacts/eval_dfm_30ep_ninespecies_test_metrics.json`: Nine-Species full test metrics (65.08% strict exact match).
  - `artifacts/eval_dfm_30ep_hcpt_test_metrics.json`: HC-PT full test metrics (55.88% I/L exact match).
  - `artifacts/full_benchmark_comparison.json`: 7-model cross-paradigm benchmark summary metrics.
  - `docs/figures/full_benchmark_comparison_30ep.png`: Publication-grade comparative figure.
- **Joint Balanced Model (8ep)**:
  - `artifacts/eval_joint_balanced/joint_ninespecies_full_test_metrics.json`
  - `artifacts/eval_joint_balanced/joint_hcpt_full_test_metrics.json`
- **InstaNovo Baseline**:
  - `artifacts/instanovo_eval/ninespecies_full_test_metrics.json`
  - `artifacts/instanovo_eval/hcpt_full_test_metrics.json`
- **Multi-Step Dynamic Knapsack Filter Ablation**:
  - `artifacts/eval_sequential_finetune/base_ninespecies_full_test_metrics.json`
  - `artifacts/eval_sequential_finetune/base_hcpt_full_test_metrics.json`
  - `artifacts/eval_sequential_finetune/finetuned_ninespecies_full_test_metrics.json`
  - `artifacts/eval_sequential_finetune/finetuned_hcpt_full_test_metrics.json`
- **Per-Species & Scheduling Studies**:
  - `artifacts/ninespecies_full_benchmark_results.json`
  - `artifacts/ninespecies_full_benchmark_summary.json`
  - `artifacts/ninespecies_post_finetune_results.json`
  - `artifacts/ninespecies_post_finetune_summary.json`
  - `artifacts/benchmark_3species_cosine.json`
  - `artifacts/benchmark_3species_power1_5.json`
  - `artifacts/benchmark_3species_improved_linear.json`
  - `artifacts/calibrated_threshold.json`
- **Validation Checkpoint Metrics**:
  - `artifacts/val_metrics/metrics_val_epoch_*.json` (epochs 00 to 49)
  - `artifacts/dfm_joint_balanced_8ep/val_metrics/metrics_val_epoch_*.json`
  - `artifacts/dfm_joint_balanced_30ep/val_metrics/pauc_val_epoch_*.png`

---

### C. Ground Truth Data (JSON)
- `artifacts/instanovo_eval/ninespecies_full_test_ground_truth.json`: Ground-truth peptide sequences and metadata for 104,163 spectra.
- `artifacts/instanovo_eval/hcpt_full_test_ground_truth.json`: Ground-truth peptide sequences and metadata for 265,369 spectra.

---

### D. Training Logs (CSV)
Epoch-by-epoch and step-by-step training losses, learning rates, and validation curves:
- `artifacts/dfm_joint_balanced_30ep/logs_csv/version_0/metrics.csv`
- `artifacts/dfm_joint_balanced_8ep/logs_csv/version_0/metrics.csv`
- `artifacts/dfm_pl_ninespecies_finetune_phase2_10ep/version_0/metrics.csv`
- `artifacts/dfm_pl_ninespecies_finetune_10ep/version_0/metrics.csv`
- `artifacts/dfm_pl_run_20260914_182552/version_0/metrics.csv`
- `artifacts/dfm_pl_run_20260914_220152/version_0/metrics.csv`
- `artifacts/dfm_pl_run_20260915_000148/version_0/metrics.csv`

---

### E. Vocabularies (JSON)
Token-to-index mappings defining the 32-token alphabet (including PTMs and structural tokens):
- `artifacts/dfm_joint_balanced_30ep/vocabulary.json`
- `artifacts/dfm_joint_balanced_8ep/vocabulary.json`
- `artifacts/ptm_extended_baseline_val40_exact=0.3557.vocabulary.json`
- `artifacts/dfm_pl_ninespecies_finetune_10ep/vocabulary.json`
- `artifacts/dfm_pl_ninespecies_finetune_phase2_10ep/vocabulary.json`

---

## 3. Example Analysis Snippet

```python
import pandas as pd

# Load DFM and InstaNovo predictions on Nine-Species full test
df_dfm = pd.read_csv("artifacts/eval_joint_balanced/joint_ninespecies_full_test_preds.csv")
df_in = pd.read_csv("artifacts/instanovo_eval/ninespecies_full_test_preds.csv")

# Compute I/L equivalent match rate
def canonicalize_il(s):
    return str(s).replace("L", "I")

dfm_il_match = (df_dfm["target"].apply(canonicalize_il) == df_dfm["prediction"].apply(canonicalize_il)).mean()
in_il_match = (df_dfm["target"].apply(canonicalize_il) == df_in["predictions"].apply(canonicalize_il)).mean()

print(f"DFM Nine-Species I/L Exact Match: {dfm_il_match * 100:.2f}%")
print(f"InstaNovo Nine-Species I/L Exact Match: {in_il_match * 100:.2f}%")
```
