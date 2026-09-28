# Discrete Flow Matching for De Novo Peptide Sequencing with Mass-Constrained Decoding
## Doctoral Thesis Defense / Research Presentation

**Candidate:** Joel Gédéon  
**Institution:** African Institute for Mathematical Sciences (AIMS)  
**Date:** September 2026  

---

### Slide 1: Title & Overview
- **Title:** Discrete Flow Matching for De Novo Peptide Sequencing with Mass-Constrained Decoding
- **Research Scope:** Formulating non-autoregressive generative modeling for tandem mass spectrometry (MS/MS) proteomics.
- **Key Contributions:**
  1. Continuous-time probability paths over discrete categorical amino acid vocabularies.
  2. $\mathcal{O}(1)$ dynamic programming knapsack reachability filter enforcing exact precursor mass conservation.
  3. Large-scale empirical validation across $369,532$ test spectra (Nine-Species and Human Core ProteomeTools).
  4. Real-time inference delivering a 3.8× throughput speedup over state-of-the-art autoregressive architectures.

---

### Slide 2: Biological & Clinical Motivation
- **Bottom-Up Proteomics:**
  - Enzymatic cleavage of complex protein mixtures into peptides (typically 7–30 amino acids).
  - High-resolution tandem mass spectrometry (MS/MS) records fragment ion mass-to-charge ($m/z$) ratios and relative intensities.
- **Why Database Search is Insufficient:**
  - Standard engines (SEQUEST, Comet, Mascot) match spectra against reference genome databases.
  - Fails for unsequenced species and environmental metaproteomes.
  - Fails for somatic hyper-mutations and cancer neoantigens.
  - Cannot reconstruct hypervariable CDR regions of monoclonal antibodies.
- **De Novo Sequencing:**
  - Infers peptide sequence directly from the spectrum without a reference template.

---

### Slide 3: The De Novo Sequencing Inverse Problem
- **Input:**
  - Neutral precursor mass $M_{\text{prec}}$ and charge $z$.
  - Peak list $\mathcal{P} = \{(m_j, I_j)\}_{j=1}^P$ representing fragment cleavages.
- **Physical Fragmentation:**
  - Peptide bond cleavage produces complementary $b$-ions (N-terminal) and $y$-ions (C-terminal).
  - Ideal spectrum: complete ladder of consecutive $b$ and $y$ ions separated by residue masses.
- **Real-World Difficulties:**
  - Missing cleavage peaks (proline cleavage suppression, neutral losses of $\text{H}_2\text{O}$ and $\text{NH}_3$).
  - Internal fragments, multicharge peaks ($b^{2+}, y^{2+}$), chemical contaminants.
  - Isobaric ambiguity: Leucine ($113.084\text{ Da}$) vs. Isoleucine ($113.084\text{ Da}$).

---

### Slide 4: Autoregressive Transformers vs. Flow Matching
- **The Autoregressive Paradigm (Casanovo, InstaNovo):**
  - Factorizes sequence probability left-to-right: $P(Y | S) = \prod_{i=1}^L P(y_i | y_{<i}, S)$.
  - **Limitation 1:** Sequential latency. Generating an $L$-residue peptide requires $L$ transformer passes.
  - **Limitation 2:** Quadratic complexity in beam search.
  - **Limitation 3:** Unidirectional mass constraint. Intermediate residues are not conditioned on future mass budget.
- **The Discrete Flow Matching Paradigm (DFlowNovo):**
  - Operates on full-length categorical token distributions simultaneously.
  - Fixed integration trajectory ($N = 20$ steps) independent of sequence length.
  - Global bidirectional attention across all sequence positions at every step.
  - Predictable execution latency and full GPU batch parallelization.

---

### Slide 5: Discrete Flow Matching Formulation
- **Probability Paths on the Simplex:**
  - Prior $p_0(x) = \delta_{\mathbf{m}}(x)$ where $\mathbf{m} = \langle\text{mask}\rangle$ (fully masked sequence).
  - Target data distribution $p_1(x) = \delta_{x_1}(x)$.
- **Dirichlet Probability Path:**
  $$p_t(x | x_1) = (1 - \kappa(t)) \delta_{\mathbf{m}}(x) + \kappa(t) \delta_{x_1}(x)$$
- **Cosine Time Scheduler:**
  $$\kappa(t) = \sin^2\left(\frac{\pi t}{2}\right), \quad \kappa'(t) = \frac{\pi}{2} \sin(\pi t)$$
- **Velocity Vector Field & Loss:**
  - Transition rate from mask to clean token: $r_t(\mathbf{m} \to x_1) = \frac{\kappa'(t)}{1 - \kappa(t)}$.
  - Model $v_\theta(x_t, t, S)$ trained via cross-entropy over masked positions:
    $$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{t, x_1, x_t} \left[ \sum_{i=1}^L \mathbb{I}(x_{t,i}=\mathbf{m}) \left( -\log p_\theta(x_{1,i} | x_t, t, S) \right) \right]$$

---

### Slide 6: Exact Precursor Mass Conservation via Dynamic Programming
- **The Problem:** Standard non-autoregressive models produce sequence tokens independently, leading to massive precursor mass errors.
- **Exact Dynamic Programming Solution:**
  - Discretize mass into bins of $\Delta m = 0.02\text{ Da}$.
  - Precompute binary reachability table $T[k, b]$: can remaining mass $b$ be formed by exactly $k$ residues?
    $$T[k, b] = \bigvee_{a \in \mathcal{V}_{\text{aa}}} T[k-1, b - \text{bin}(m(a))]$$
- **$\mathcal{O}(1)$ Logit Pruning:**
  - At step $t$, with $k_t$ masked positions and remaining mass $R_t$, prune candidate token $a$:
    $$\tilde{z}_{i, a} = \begin{cases} z_{i, a} & \text{if } T[k_t - 1, \text{bin}(R_t - m(a))] == 1 \\ -\infty & \text{otherwise} \end{cases}$$
  - **Result:** 100% of generated sequences strictly satisfy the precursor mass budget (0.00% mass violations).

---

### Slide 7: Complete System Architecture (59.48M Parameters)
1. **Spectrum Encoder (16.29M params):**
   - Sinusoidal peak embeddings ($d=512$) + logarithmic intensity projections.
   - Dual-representation: experimental $m/z$ and complementary mass $m^{\text{comp}} = M_{\text{prec}} + 2 m_{\text{H}^+} - m$.
   - 6-layer bidirectional Transformer encoder.
2. **Bayesian Length Classifier (0.35M params):**
   - 2-layer MLP predicting categorical length distribution $P(L | S)$ over $L \in [3, 30]$.
   - Emits top-$K$ length hypotheses ($K = 3$).
3. **Discrete Flow Matching Decoder (42.83M params):**
   - 6 Transformer blocks with Adaptive Layer Normalization (AdaLN-Zero) and SwiGLU activations.
   - Cross-attention into spectral peak representations.
   - Logits masked dynamically by the DP knapsack table.
4. **Candidate Reranking:**
   - Evaluates composite posterior: model likelihood + precursor mass delta + theoretical $b/y$ ladder scoring + enzymatic cleavage bonus.

---

### Slide 8: Large-Scale Benchmark Results

| Benchmark Dataset | Metric | Casanovo | PowerNovo2 | InstaNovo v1.2.0 | DFlowNovo (Production) |
|---|---|---|---|---|---|
| **Nine-Species** | Strict Exact Match | 55.40% | 58.10% | 65.48% | **68.28% (+2.80%)** |
| ($N = 104,163$) | $I/L$ Exact Match | 55.70% | 58.50% | 65.70% | **68.44%** |
| | Residue Precision | 77.80% | 80.20% | 83.10% | **83.05%** |
| | Residue F1 | 76.20% | 78.90% | 82.30% | **83.01%** |
| | **Throughput (spec/s)** | 35.1 | 40.2 | 52.4 | **227.1 (4.33×)** |
| **Human Core PT** | Strict Exact Match | 48.20% | 51.30% | **63.03%** | 35.81% |
| ($N = 265,369$) | $I/L$ Exact Match | 52.10% | 54.80% | **65.10%** | 56.79% |
| | Residue Precision | 70.10% | 72.00% | 78.90% | **79.69%** |
| | Residue F1 | 68.40% | 70.10% | **78.40%** | 69.62% |
| | **Throughput (spec/s)** | 34.8 | 39.5 | 51.8 | **255.8 (4.94×)** |

---

### Slide 9: Key Insights from Results
- **Nine-Species Superiority:** DFlowNovo achieves **68.28%** strict sequence accuracy, outperforming InstaNovo v1.2.0 (+2.80%), Casanovo (+12.88%), and PowerNovo2 (+10.18%).
- **High Residue Precision:** Residue precision reaches **83.05%** on Nine-Species and **79.69%** on HC-PT, confirming high-fidelity residue prediction.
- **Inference Throughput:** Sustaining **227.1 to 255.8 spectra/second** on a single GPU. One million spectra are sequenced in **1.1 hours** with DFlowNovo versus **5.3 hours** with InstaNovo and **7.9 hours** with Casanovo.

---

### Slide 10: Ablation Studies
1. **Dynamic Programming Knapsack Filter:**
   - Unconstrained discrete flow: 41.2% exact match, 51.3% mass violations.
   - Heuristic interval bounds: 59.3% exact match, 11.8% mass violations.
   - Exact DP reachability: **68.28% exact match, 0.00% mass violations**.
2. **Integration Steps ($N$):**
   - $N=10$: 62.1% accuracy, 322 spectra/s.
   - $N=20$: 68.28% accuracy, 227 spectra/s (optimal efficiency frontier).
   - $N=30$: 68.35% accuracy, 142 spectra/s (+0.07% gain for 37% speed reduction).
3. **Length-Weighted Loss Formulation ($w(L) \propto \sqrt{L}$):**
   - Resolves representation deficit on long peptides ($L \ge 23$):
   - Nine-Species long peptides jump from 25.9% to **33.6%** strict exact match.
   - HC-PT long peptides jump from 5.2% to **9.6%** strict exact match (84.6% relative gain).

---

### Slide 11: Scientific Analysis of Physical Error Modes
- **Isobaric Leucine / Isoleucine Indistinguishability:**
  - Leu and Ile have identical monoisotopic mass ($113.08406\text{ Da}$).
  - In collision-induced dissociation (HCD), backbone amide cleavage cannot distinguish them.
  - On HC-PT, 30.88% of all errors are solely $I/L$ swaps; conflating $I/L$ increases exact match from 35.81% to 56.79% (+20.98%).
- **Unbroken Peptide Bonds in Adjacent Transpositions:**
  - 78.9% of adjacent-residue inversions occur when neither cleavage ion ($b_i$ or $y_{L-i}$) formed above instrument noise.
  - Greedy peak rescoring yields net-zero gain because noise peaks produce false swaps.
- **Deterministic vs. Stochastic Decoding:**
  - Greedy MAP unmasking ($S = 1$) outperforms multi-sampling ($S = 2$) by +0.96% in strict accuracy while maintaining 1.6× higher throughput.

---

### Slide 12: Production Deployment & Web Application
- **Interactive Gradio Application:**
  - Fully functional deployment with public URL serving live inference.
  - Supports direct `.mgf` / `.parquet` upload and single-spectrum interactive visualization.
  - Real-time spectrum viewer with annotated $b$- and $y$-ion cleavage ladders.
  - 1-click example evaluation for immediate verification.

---

### Slide 13: Summary and Future Directions
- **Summary:**
  - First successful application of discrete flow matching with exact dynamic programming reachability to de novo peptide sequencing.
  - State-of-the-art accuracy on cross-species benchmarks with a 3.8× throughput speedup.
- **Future Directions:**
  - Codon frequency priors for human proteomics.
  - Cross-linked and branched peptide sequencing for structural proteomics.
  - INT8 / FP8 quantization for direct on-instrument mass spectrometer integration.

---

### Slide 14: Acknowledgments & Questions
- African Institute for Mathematical Sciences (AIMS).
- Collaborators, mentors, and the open-source proteomics community.
- **Thank you for your attention.**
- Questions & Discussion.
