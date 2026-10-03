# Full Benchmark Report: InstaNovo vs. DFM on Nine-Species Test Split

**Evaluation Scope**: Official Test Split of the Nine-Species Benchmark ([`InstaDeepAI/ms_ninespecies_benchmark`](https://huggingface.co/datasets/InstaDeepAI/ms_ninespecies_benchmark), `split="test"`), filtering valid peptide lengths ($1 \le L \le 30$).  
**Total Test Spectra Evaluated**: **104,163 spectra** across all 9 organisms.  
**Hardware Platform**: Dedicated NVIDIA H100 80GB HBM3 GPU.

---

## 1. Executive Summary & Head-to-Head Results

The pre-trained **InstaNovo** model (`instanovo-v1.2.0`, autoregressive transformer with 5-beam knapsack beam search) was evaluated end-to-end across the **complete 104,163 spectra** of the Nine-Species benchmark test set. Below is the standardized head-to-head comparison against both the **Zero-Shot Base DFM Model** and the **Fine-Tuned DFM Model (20 Epochs)** evaluated under identical conditions.

| Evaluation Metric | DFM Base (Zero-Shot) | DFM Fine-Tuned (20ep) | InstaNovo (`v1.2.0`) | Winner & Margin |
| :--- | :---: | :---: | :---: | :---: |
| **Strict Exact Match** | 12.26% (12,770) | **66.62%** (69,396) | 15.45% (16,093) | **DFM Fine-Tuned** (+51.17%) |
| **I/L-Equivalent Exact Match** | 51.64% (53,786) | 66.62% (69,396) | **71.09%** (74,051) | **InstaNovo** (+4.47%) |
| **Precursor Mass Match** | 58.01% (60,421) | 68.07% (70,899) | **72.90%** (75,931) | **InstaNovo** (+4.83%) |
| **Length Accuracy** | 76.87% (80,068) | **84.38%** (87,897) | 80.65% (84,006) | **DFM Fine-Tuned** (+3.73%) |
| **Residue Amino Acid Precision** | 71.49% | **81.42%** | 75.36% | **DFM Fine-Tuned** (+6.06%) |
| **Residue Amino Acid Recall** | 70.76% | **81.30%** | 78.46% | **DFM Fine-Tuned** (+2.84%) |
| **Residue Amino Acid F1** | 71.12% | **81.36%** | 76.88% | **DFM Fine-Tuned** (+4.48%) |
| **Precision-Coverage AUC (PC-AUC)** | 0.8355 | 0.8971 | **0.9083** | **InstaNovo** (+0.0112) |
| **Coverage @ $\ge$ 80% Precision** | 66.01% (68,757) | 82.49% (85,925) | **90.97%** (94,755) | **InstaNovo** (+8.48%) |
| **Inference Throughput** | **120.6 spectra/s** | **120.6 spectra/s** | 53.3 spectra/s | **DFM (2.26x Faster)** |
| **Full 104k Evaluation Time** | **14.4 min** | **14.4 min** | 32.6 min | **DFM (18.2 min faster)** |

---

## 2. Key Insights & Architectural Analysis

![Nine-Species Full Test Benchmark: DFM vs InstaNovo](instanovo_vs_dfm_full_test_comparison.png)

### Insight A: DFM Outperforms InstaNovo on Residue-Level Fidelity (Amino Acid F1)
- **DFM Fine-Tuned achieves 81.36% AA F1**, exceeding InstaNovo's **76.88% AA F1** by **+4.48% absolute**.
- **DFM Precision is 81.42%**, outperforming InstaNovo's **75.36%** by **+6.06%**.
- **DFM Length Accuracy is 84.38%**, outperforming InstaNovo's **80.65%** by **+3.73%**.
- **Architectural Rationale**: Non-autoregressive discrete flow matching considers the entire sequence globally at every generation step with bidirectional attention, avoiding the cumulative exposure bias and error compounding that autoregressive beam search models experience on longer peptides.

### Insight B: The Strict Exact Match Anomaly — Leucine vs. Isoleucine Canonicalization
- In the Nine-Species dataset, the dataset curators **canonicalized 100% of Leucine (`L`) residues to Isoleucine (`I`)** in ground truth labels (`L` count = 0, `I` count = 20,773).
- **InstaNovo** was pre-trained on diverse natural datasets containing both `L` and `I`. Consequently, InstaNovo predicts `L` whenever the biological prior dictates it. Because the Nine-Species benchmark labels contain zero `L`s, InstaNovo's **strict exact match plummets to 15.45%**, even though its **I/L-equivalent exact match is 71.09%**.
- **DFM Fine-Tuned** adapted to the Nine-Species labeling distribution during fine-tuning, predicting `I` for isobaric 113.084 Da mass jumps, allowing its strict and I/L match to coincide at **66.62%**.
- Under standard MS/MS convention where `I` and `L` are indistinguishable by low-energy collision-induced dissociation (CID), **InstaNovo achieves 71.09%** and **DFM achieves 66.62%** (a small 4.47% margin).

### Insight C: Inference Speed & Computational Efficiency (2.26x Speedup)
- On an identical NVIDIA H100 GPU:
  - **InstaNovo (Autoregressive 5-Beam Knapsack)**: 53.3 spectra/sec (1,955 seconds = **32.6 minutes** for 104,163 spectra).
  - **DFM (Non-Autoregressive Discrete Flow Matching)**: 120.6 spectra/sec (864 seconds = **14.4 minutes** for 104,163 spectra).
- **DFM provides a 2.26x wall-clock speedup** while delivering higher amino acid residue precision (+6.06%) and higher length accuracy (+3.73%).

### Insight D: Confidence Scoring & FDR Filtering (PC-AUC)
- InstaNovo's beam search log-probabilities provide smooth confidence ranking across the dataset:
  - InstaNovo reaches **90.97% coverage at $\ge 80\%$ precision** (threshold $\tau = -11.71$, yielding 94,755 high-confidence PSMs).
  - DFM Fine-Tuned reaches **82.49% coverage at $\ge 80\%$ precision** (threshold $\tau = -0.827$, yielding 85,925 high-confidence PSMs).
  - DFM Base (Zero-Shot) reaches **66.01% coverage at $\ge 80\%$ precision** (threshold $\tau = -0.422$, yielding 68,757 high-confidence PSMs).

---

## 3. Artifacts and Reproducibility

- **InstaNovo Predictions**: [`artifacts/instanovo_eval/ninespecies_full_test_preds.csv`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/instanovo_eval/ninespecies_full_test_preds.csv)
- **InstaNovo Metrics JSON**: [`artifacts/instanovo_eval/ninespecies_full_test_metrics.json`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/instanovo_eval/ninespecies_full_test_metrics.json)
- **DFM Fine-Tuned Metrics JSON**: [`artifacts/eval_ninespecies_full_test.json`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/eval_ninespecies_full_test.json)
- **DFM Zero-Shot Metrics JSON**: [`artifacts/eval_ninespecies_full_test_zeroshot_baseline.json`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/eval_ninespecies_full_test_zeroshot_baseline.json)
- **Comprehensive Benchmark Plot**: [`artifacts/instanovo_vs_dfm_full_test_comparison.png`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/artifacts/instanovo_vs_dfm_full_test_comparison.png)
- **Evaluation Runner Script**: [`scripts/eval_instanovo_full_test.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/scripts/eval_instanovo_full_test.py)
