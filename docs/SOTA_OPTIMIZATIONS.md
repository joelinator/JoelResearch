# SOTA Algorithmic and Decoding Optimizations for Discrete Flow Matching De Novo Peptide Sequencing

**Author:** Joelinator / Google DeepMind Pair Programming  
**Branch:** `feature/sota-architecture-opt`  
**Target:** Closing the Performance Gap to SOTA De Novo Sequencing (InstaNovo, Casanovo)  
**Date:** September 2026  

---

## 1. Executive Summary

This document details the design, mathematical formulation, and implementation of six algorithmic and decoding optimizations integrated into the **Discrete Flow Matching (DFM)** peptide sequencing framework. 

Following the architectural streamlining to **59.47M parameters** and Bayesian length beam decoding, these optimizations target the primary failure modes of generative flow matching:
1. **Order-Agnostic Sampling Deficiencies**: Uniform random unmasking forces difficult, low-information positions to be decoded simultaneously with easy anchors. Resolved via **Confidence-Based Flow Matching Decoding**.
2. **Disregard for Fragment Spectrum Evidence in Beam Selection**: Generated candidates previously relied only on precursor neutral mass matching. Resolved via **GPU-Vectorized Theoretical $b/y$ Fragment Ion Matching**.
3. **Overconfidence and Fragile Representations**: Resolved via **Label Smoothing** ($0.05$) and **Spectral Peak Dropout Augmentation** ($0.15$).
4. **SGD Weight Jitter and Generalization Gaps**: Resolved via **Exponential Moving Average (EMA)** shadow weights ($\beta = 0.999$).
5. **Isobaric Amino Acid Mispenalization**: Isoleucine (I) and Leucine (L) share identical mass ($113.084064\text{ Da}$). Resolved via **I/L Equivalence Tracking** in benchmark metrics.
6. **Padding-Safe Cumulative Mass Loss**: Guaranteed zero-padding invariance and dynamic vocabulary index alignment.

All components are fully tested (`tests/test_sota_enhancements.py`, `tests/test_user_modifications.py`, and the full unit test suite passing 100%).

---

## 2. Mathematical Formulations & Implementations

### 2.1 Confidence-Based Flow Matching Decoding & Low-Temperature Sampling

#### Problem
In standard discrete flow matching, reverse sampling from $t=0 \to 1$ unmasks corrupted tokens uniformly at random based solely on the transition rate:
$$P(\text{unmask position } i \text{ at step } t) \propto \frac{\kappa'(t) \Delta t}{1 - \kappa(t)}$$
Uniform selection can prematurely unmask positions where the model has high entropy and low confidence, leading to error propagation in subsequent flow steps.

#### Solution
Under `strategy="confidence"`, the model prioritizes unmasking positions where it is most certain of its prediction:
1. At each reverse step $t$, compute token posterior probabilities:
   $$P_t(Y_i = a) = \text{Softmax}\left(\frac{\mathbf{z}_{i, a}}{T}\right)$$
   where $T$ is the sampling temperature. When $T \le 0.05$, greedy argmax MAP sampling is used: $Y_i^* = \arg\max_a \mathbf{z}_{i, a}$.
2. Measure confidence at each masked position:
   $$c_i = \max_{a \in \mathcal{V}} P_t(Y_i = a)$$
3. If $k$ tokens are scheduled to be unmasked at this step based on the scheduler transition budget:
   $$k = \min\left(N_{\text{masked}}, \operatorname{round}\left(N_{\text{active}} \cdot \frac{\kappa'(t)\Delta t}{1 - \kappa(t)}\right)\right)$$
   unmask the top-$k$ positions with the highest confidence scores $c_i$.

**File Reference:** [`src/flow_matching/sampling.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/flow_matching/sampling.py#L90-L160)

---

### 2.2 Theoretical Fragment Ion ($b/y$-ion) Peak Matching & Beam Re-ranking

#### Problem
Candidate peptide sequences generated across candidate lengths in beam decoding previously competed only on precursor mass error and length prior:
$$\text{Score}(Y) = \log P(L \mid \mathcal{S}) - \alpha \cdot \text{PPM}(Y)^2$$
Two different sequence permutations with identical amino acid composition have the exact same precursor mass, but produce completely different MS/MS fragment ion patterns.

#### Solution
For each candidate sequence $Y = (y_1, y_2, \dots, y_L)$, we compute all theoretical singly charged $b$-ions and $y$-ions directly on GPU:
$$b_i = \sum_{j=1}^i m(y_j) + M_{\text{H}^+}, \quad i \in \{1, \dots, L-1\}$$
$$y_k = M_{\text{prec}} - \left(\sum_{j=1}^{L-k} m(y_j)\right) + 2M_{\text{H}^+} + M_{\text{H}_2\text{O}}, \quad k \in \{1, \dots, L-1\}$$

We broadcast theoretical ions against experimental spectrum peaks $\mathcal{S} = \{(m_j^{\text{exp}}, I_j^{\text{exp}})\}_{j=1}^P$ and evaluate matches within instrument tolerance ($\tau_{\text{ppm}} = 20\text{ ppm}$ or $\tau_{\text{Da}} = 0.05\text{ Da}$):
$$\text{match}(i, j) = \mathbb{I}\left( \frac{|m_i^{\text{theo}} - m_j^{\text{exp}}|}{m_i^{\text{theo}}} \times 10^6 \le \tau_{\text{ppm}} \lor |m_i^{\text{theo}} - m_j^{\text{exp}}| \le \tau_{\text{Da}} \right)$$

The fragment matching score represents the fraction of total experimental intensity explained by the candidate sequence:
$$\text{FragScore}(Y) = \frac{\sum_{j \in \text{matched}} I_j^{\text{exp}}}{\sum_{j=1}^P I_j^{\text{exp}}}$$

The composite beam selection score is:
$$\text{Score}_{\text{final}}(Y) = \log P(L \mid \mathcal{S}) - \alpha \cdot \text{PPM}(Y)^2 + \beta \cdot \text{FragScore}(Y) + \text{Bonus}_{\text{trypsin}}(Y)$$
where $\text{Bonus}_{\text{trypsin}} = 0.2$ if the C-terminal amino acid is Lysine (K) or Arginine (R) (standard tryptic cleavage).

**File Reference:** [`src/inference/predict.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/inference/predict.py#L64-L130)

---

### 2.3 Label Smoothing in Cross-Entropy Loss

#### Formulation
Discrete flow matching requires predicting the clean token $y_1$ given partially corrupted tokens $x_t$. Standard one-hot cross-entropy encourages logits to approach $\infty$, leading to overconfident predictions and poor calibration during multi-step reverse sampling.

With label smoothing parameter $\epsilon = 0.05$, the target distribution over vocabulary size $V$ becomes:
$$q(y \mid x_t) = (1 - \epsilon)\mathbb{I}(y = y^*) + \frac{\epsilon}{V - 1}\mathbb{I}(y \ne y^*)$$
Padded positions (`<pad>`) are masked out from loss computation and normalization.

**File References:**
- [`src/train/loss.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/train/loss.py#L190-L210)
- [`src/train/lightning.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/train/lightning.py#L125-L132)

---

### 2.4 Exponential Moving Average (EMA) Callback

#### Formulation
Stochastic Gradient Descent and AdamW induce weight fluctuations across training iterations. EMA maintains a running shadow copy of the network parameters:
$$\theta_{\text{shadow}}^{(t)} = \beta \cdot \theta_{\text{shadow}}^{(t-1)} + (1 - \beta) \cdot \theta^{(t)}$$
with decay rate $\beta = 0.999$.

#### Seamless PyTorch Lightning Integration
The custom `EMACallback(pl.Callback)`:
1. Updates shadow weights at the end of every training batch: `on_train_batch_end`.
2. Swaps shadow weights into the active model before validation or testing begins: `on_validation_start`, `on_test_start`.
3. Restores raw training weights after validation completes: `on_validation_end`, `on_test_end`.
4. Serializes `ema_shadow_params` into model checkpoints: `on_save_checkpoint`, `on_load_checkpoint`.

**File Reference:** [`src/train/callbacks.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/train/callbacks.py)

---

### 2.5 Spectral Peak Dropout Data Augmentation

#### Formulation
Experimental tandem mass spectra suffer from instrument noise, missing peaks, and dynamic range limitations. To prevent the model from over-relying on any single dominant peak, peak dropout randomly zeroes out peak intensities during training with probability $p = 0.15$:
$$I_j \leftarrow I_j \cdot m_j, \quad m_j \sim \text{Bernoulli}(1 - p_{\text{dropout}})$$
To preserve spectrum identity, if all peaks happen to be masked, the base peak ($\arg\max_j I_j$) is unconditionally retained. When `is_train=False` (validation and testing), peak dropout is disabled ($p = 0$).

**File Reference:** [`src/data/data.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/data/data.py#L125-L140)

---

### 2.6 Isobaric I/L Equivalence in Benchmark Evaluation

#### Chemical Equivalence
Isoleucine (I) and Leucine (L) have the exact same chemical formula ($C_6 H_{13} N O_2$) and monoisotopic mass ($113.084064\text{ Da}$). In standard low-energy collision-induced dissociation (CID/HCD), they cannot be distinguished.

#### Dual Evaluation Metrics
[`src/eval/metrics.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/src/eval/metrics.py) now reports both:
1. `exact_peptide_accuracy`: Strict string equality (`pred == target`).
2. `exact_peptide_accuracy_il`: Standard proteomics benchmark equality (`pred.replace("I", "L") == target.replace("I", "L")`), matching InstaNovo and Casanovo published protocols.
3. Associated curves: `auc_exact_il`, `pauc80_exact_il`, and `pr_auc_exact_il`.

---

### 2.7 Cumulative Prefix Mass Hubert Loss & Padding Invariance

The user-introduced cumulative mass loss:
$$\mathcal{L}_{\text{cum}} = \frac{1}{|\mathcal{A}|} \sum_{i \in \mathcal{A}} \text{Huber}\left( \sum_{j=1}^i \hat{m}_j, \sum_{j=1}^i m(y_j) \right)$$
was verified to:
1. Mask out `<pad>` tokens so that padded positions cannot corrupt prefix mass sums.
2. Align dynamic vocabulary indices so that non-standard or pad tokens safely map to 0.0 mass without index out-of-bounds exceptions.

---

## 3. CLI Configuration Reference

All optimizations are exposed as command-line arguments in [`scripts/train_lightning.py`](file:///home/joelgedeon_aims_ac_za/dfm-joelresearch/scripts/train_lightning.py):

| Flag | Type | Default | Description |
|---|---|---|---|
| `--use-ema` / `--no-ema` | `bool` | `True` | Maintain EMA shadow weights ($\beta = 0.999$) for validation and checkpoints |
| `--ema-decay` | `float` | `0.999` | Decay factor for EMA |
| `--label-smoothing` | `float` | `0.05` | Label smoothing factor for cross-entropy peptide loss |
| `--peak-dropout` | `float` | `0.15` | Spectral peak dropout probability during training |
| `--decoding-strategy` | `str` | `confidence` | Reverse flow unmasking strategy (`confidence` or `random`) |
| `--decoding-temperature`| `float` | `0.0` | Sampling temperature ($0.0 = $ greedy argmax MAP) |
| `--fragment-matching-weight` | `float` | `0.5` | Weight $\beta$ for theoretical $b/y$ fragment ion matching in beam scoring |
| `--trypsin-prior` / `--no-trypsin-prior` | `bool` | `True` | Apply +0.2 candidate bonus for tryptic C-terminal residues (K/R) |

---

## 4. Test Verification Summary

The test suite validates every component:

```bash
PYTHONPATH=src .venv/bin/python tests/test_sota_enhancements.py
```
- `test_confidence_based_unmasking`: Verified top-$k$ confidence ranking unmasks highest probability tokens first.
- `test_fragment_ion_matching_scores`: Verified theoretical $b/y$ ion generation, tolerance matching, and explained intensity ratio computation.
- `test_peak_dropout_augmentation`: Verified stochastic dropout in training mode and strict preservation in evaluation mode.
- `test_label_smoothing_loss`: Verified smoothing loss computation and gradient backpropagation.
- `test_ema_callback_mechanics`: Verified parameter tracking, validation weight swapping, and restoration.
- `test_il_equivalence_metric`: Verified 100% accuracy on I/L substituted peptides.

Full repository test suite (`tests/test_*.py`): **All 9 test suites passed with exit code 0.**
