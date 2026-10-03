# Full Benchmark Report: InstaNovo vs. DFM on HC-PT Test Split

**Evaluation Scope**: Official Test Split of the High-Confidence ProteomeTools Benchmark ([`InstaDeepAI/ms_proteometools`](https://huggingface.co/datasets/InstaDeepAI/ms_proteometools), `split="test"`).  
**Total Test Spectra Evaluated**: **265,369 spectra**.  
**Hardware Platform**: Dedicated NVIDIA H100 80GB HBM3 GPU.

---

## 1. Executive Summary & Head-to-Head Results

The official pre-trained **InstaNovo** model (`instanovo-v1.2.0`, autoregressive transformer with 5-beam knapsack beam search) was evaluated end-to-end across the **complete 265,369 spectra** of the HC-PT test split. Below is the head-to-head comparison against the **DFM Model** evaluated under identical conditions.

| Evaluation Metric | DFM (Fine-Tuned Model) | InstaNovo (`instanovo-v1.2.0`) | Margin / Winner |
| :--- | :---: | :---: | :---: |
| **Strict Exact Match** | 13.16% (34,921) | **63.03%** (167,270) | **InstaNovo** (+49.87%) |
| **I/L-Equivalent Exact Match** | 49.69% (131,862) | **66.15%** (175,548) | **InstaNovo** (+16.46%) |
| **Precursor Mass Match** | 55.17% (146,394) | **73.20%** (194,260) | **InstaNovo** (+18.03%) |
| **Length Accuracy** | 77.53% (205,729) | **78.27%** (207,704) | **InstaNovo** (+0.74%) |
| **Residue Amino Acid Precision** | 69.24% | **75.65%** | **InstaNovo** (+6.41%) |
| **Residue Amino Acid Recall** | 69.27% | **78.14%** | **InstaNovo** (+8.87%) |
| **Residue Amino Acid F1** | 69.26% | **76.87%** | **InstaNovo** (+7.61%) |
| **Precision-Coverage AUC (PC-AUC)** | 0.8064 | **0.9386** | **InstaNovo** (+0.1322) |
| **Coverage @ $\ge$ 80% Precision** | 64.11% (170,131) | **91.47%** (242,746) | **InstaNovo** (+27.36%) |
| **Inference Throughput** | **120.6 spectra/s** | 51.9 spectra/s | **DFM (2.32x Faster)** |
| **Total Test Inference Time** | **36.6 min** | 85.2 min | **DFM saves 48.6 min** |

---

## 2. Key Insights & Architectural Analysis

![HC-PT Full Test Benchmark: DFM vs InstaNovo](instanovo_vs_dfm_hcpt_full_test_comparison.png)

### Insight A: Definitively Resolving the "Performance Drop" on HC-PT
Earlier, we observed that when DFM was fine-tuned on Nine-Species, its strict match on HC-PT dropped drastically to 13.16%. These InstaNovo results provide the definitive mathematical proof:
1. **The Label Canonicalization Asymmetry**:
   - The **Nine-Species benchmark** canonicalized all Leucine (`L`) residues to Isoleucine (`I`) (`L` count = 0 in labels).
   - In contrast, **HC-PT contains natural biological distributions of both `L` and `I`**.
2. **Behavior on Nine-Species**:
   - InstaNovo (trained on natural `L`/`I`) achieved only **15.45% strict match**, but **71.09% I/L match**.
   - DFM (fine-tuned on Nine-Species) adapted to predict `I`, achieving **66.62% strict match**.
3. **Behavior on HC-PT**:
   - InstaNovo achieves **63.03% strict match** and **66.15% I/L match** (a tiny 3.12% gap, because it naturally models both `L` and `I`).
   - DFM crashed to **13.16% strict match**, but retained **49.69% I/L match**!
4. **Conclusion**:
   The "drop" on HC-PT was **not** a collapse of de novo peptide sequencing capabilities, but an artifact of DFM learning the Nine-Species labeling bias (predicting `I` for isobaric 113.084 Da residues) which penalized it under strict matching against HC-PT's uncanonicalized labels.

### Insight B: InstaNovo's Dominance on HC-PT
- **Sequence Identification**: InstaNovo recovers **66.15%** of peptide sequences (I/L equivalent) and **73.20%** within mass tolerance, outperforming DFM's 49.69% and 55.17%.
- **Residue-Level Metrics**: InstaNovo achieves **76.87% AA F1** compared to DFM's **69.26%** (+7.61% absolute).
- **Proteomic Yield**: At $\ge 80\%$ precision, InstaNovo validates **91.47% coverage** (242,746 spectra), yielding an additional **72,615 high-confidence PSMs** over DFM (64.11% coverage, 170,131 spectra).
- **Why**: ProteomeTools (HC-PT) was one of the primary datasets on which InstaNovo's base model was trained, making HC-PT in-domain for InstaNovo.

### Insight C: DFM's 2.32x Throughput Advantage
- Even with scaled batch sizes (`batch_size=2048` allocating 37.4 GB VRAM on H100):
  - **InstaNovo** averaged **51.9 spectra/sec** (85.2 minutes total).
  - **DFM** averaged **120.6 spectra/sec** (36.6 minutes total).
- Non-autoregressive discrete flow matching decodes peptides in parallel, providing a **2.32x speedup** on large-scale proteomic repositories.

---

## 3. Artifacts and Reproducibility

- **InstaNovo Predictions**: [`artifacts/instanovo_eval/hcpt_full_test_preds.csv`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/instanovo_eval/hcpt_full_test_preds.csv)
- **InstaNovo Metrics JSON**: [`artifacts/instanovo_eval/hcpt_full_test_metrics.json`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/instanovo_eval/hcpt_full_test_metrics.json)
- **DFM HC-PT Metrics JSON**: [`artifacts/eval_hcpt_full_test.json`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/eval_hcpt_full_test.json)
- **Publication Comparison Figure**: [`artifacts/instanovo_vs_dfm_hcpt_full_test_comparison.png`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/instanovo_vs_dfm_hcpt_full_test_comparison.png)
- **Evaluation Runner Script**: [`scripts/eval_instanovo_hcpt_full_test.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/scripts/eval_instanovo_hcpt_full_test.py)
