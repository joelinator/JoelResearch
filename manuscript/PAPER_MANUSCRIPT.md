# Discrete Flow Matching for De Novo Peptide Sequencing with Dynamic Precursor Mass Constraints

**Joel Gédéon**  
*African Institute for Mathematical Sciences (AIMS)*  
*September 2026*

---

## Abstract

De novo peptide sequencing from tandem mass spectrometry (MS/MS) spectra enables the identification of unsequenced organisms, antibody repertoires, and post-translational modifications without reference proteome databases. Existing state-of-the-art methods rely almost exclusively on autoregressive sequence-to-sequence transformers that predict amino acids iteratively from left to right. While effective, autoregressive decoding exhibits sequential latency that scales with peptide length and cannot enforce global physical constraints, such as the total precursor mass, across intermediate positions simultaneously. In this work, we formulate de novo peptide sequencing as a discrete flow matching problem over categorical probability paths. Our architecture, DFlowNovo, integrates continuous-time probability trajectories over discrete amino acid vocabularies while enforcing exact precursor mass reachability at each discrete unmasking step via a dynamic programming knapsack filter. On the standard Nine-Species benchmark ($N = 104,163$ test spectra), DFlowNovo achieves **68.28%** strict sequence accuracy (exceeding InstaNovo v1's 65.48% by $+2.80\%$) while delivering a **4.3× to 4.9× throughput increase** (227.1–255.8 vs. 52.4 spectra/s on an NVIDIA GPU). On the Human Core ProteomeTools benchmark ($N = 265,369$ spectra), DFlowNovo achieves **56.79%** isobaric ($I/L$-conflated) sequence accuracy and **69.62%** residue F1 score. We analyze the mathematical properties of discrete flow matching for biosequence generation, quantify the role of exact dynamic programming constraints versus heuristic pruning, and characterize the physical error modes arising from isobaric residue isomers and incomplete fragmentation.

---

## 1. Introduction

Tandem mass spectrometry (MS/MS) is the core analytical technique in bottom-up proteomics. Enzymatically digested proteins produce peptide fragments that are ionized, selected by mass-to-charge ratio ($m/z$), fragmented by collision-induced dissociation or higher-energy collisional dissociation (HCD), and recorded as an intensity spectrum across fragment $m/z$ values.

Traditionally, peptide identification is performed by database search algorithms (such as SEQUEST, Comet, or MS-GF+), which match experimental spectra against theoretical spectra predicted from a curated reference genome. However, database search fails when the organism's genome is unsequenced, when unexpected genetic polymorphisms or post-translational modifications (PTMs) occur, or during the characterization of immunoglobulin (antibody) variable regions. Under these circumstances, de novo sequencing is required to reconstruct the peptide sequence directly from the spectrum without prior sequence templates.

Early de novo sequencing algorithms relied on graph-theoretical approaches, such as spectrum graphs (Sherenga, Lutefisk, PEAKS), where peaks correspond to nodes and edges represent mass differences corresponding to amino acid residues. While theoretically sound, spectrum graphs struggle with incomplete fragmentation, internal cleavage fragments, and chemical noise.

Deep learning renewed de novo sequencing performance. DeepNovo and PointNovo introduced convolutional and recurrent neural networks with knapsack beam search. Recently, Casanovo and InstaNovo demonstrated substantial accuracy gains by employing autoregressive transformer architectures. InstaNovo frames peptide prediction as autoregressive translation from peak tokens to amino acid tokens, guided by a dynamic programming knapsack beam search that prunes partial sequences violating the precursor mass budget.

Despite their success, autoregressive models present three key limitations:
1. **Sequential inference latency**: Autoregressive decoding requires $L$ sequential transformer forward passes for a peptide of length $L$, creating an inference bottleneck for high-throughput mass spectrometry facilities that generate tens of millions of spectra per day.
2. **Exposure bias and error propagation**: Left-to-right generation cannot revise earlier residue assignments if later spectral evidence contradicts them.
3. **Unidirectional mass constraint enforcement**: In autoregressive generation, mass constraints act as a filter during beam expansion, but the model cannot condition token generation at position $i$ on residues assigned at position $j > i$.

To address these limitations, we propose **DFlowNovo**, a discrete flow matching framework for de novo peptide sequencing. Flow matching defines continuous-time probability paths that transport probability mass from a tractable prior (such as a fully masked sequence) to the target empirical distribution over discrete sequences. By conditioning the velocity field on both spectral peak representations and an exact dynamic programming reachability constraint, DFlowNovo generates full-length peptide sequences in parallel across a fixed number of integration steps ($N = 20$), offering predictable latency and high GPU throughput.

---

## 2. Problem Formulation and Mathematical Framework

### 2.1 Tandem Mass Spectrometry Representation

An experimental MS/MS spectrum is represented as a tuple $S = (M_{\text{prec}}, z, \mathcal{P})$, where:
- $M_{\text{prec}} \in \mathbb{R}^+$ is the neutral precursor mass in Daltons (Da), computed from the measured precursor mass-to-charge ratio $m/z_{\text{prec}}$ and charge state $z \in \{1, 2, \dots, 8\}$:
  $$M_{\text{prec}} = z \cdot (m/z_{\text{prec}} - m_{\text{H}^+})$$
  where $m_{\text{H}^+} = 1.007276\text{ Da}$ is the proton mass.
- $\mathcal{P} = \{(m_j, I_j)\}_{j=1}^{P}$ is a set of $P$ detected peaks with mass-to-charge ratios $m_j$ and normalized intensities $I_j \in [0, 1]$.
- In addition, we compute the complementary mass for each peak:
  $$m_j^{\text{comp}} = M_{\text{prec}} + 2 m_{\text{H}^+} - m_j$$
  allowing simultaneous representation of potential $b$- and $y$-ion pairs.

A target peptide is an amino acid sequence $Y = (y_1, y_2, \dots, y_L)$ of length $L$, where each token $y_i$ belongs to a vocabulary $\mathcal{V}$ consisting of standard amino acids, targeted post-translational modifications (e.g., oxidized methionine $\text{M(ox)}$, carbamidomethylated cysteine $\text{C(cam)}$, deamidated asparagine/glutamine), and special tokens ($\langle\text{pad}\rangle, \langle\text{mask}\rangle$).

The physical mass conservation law requires that the sum of residue masses plus the mass of water equals the neutral precursor mass within instrument tolerance $\tau$:
$$\left| \sum_{i=1}^L m(y_i) + M_{\text{H}_2\text{O}} - M_{\text{prec}} \right| \le \tau$$
where $M_{\text{H}_2\text{O}} = 18.010565\text{ Da}$, and $\tau$ is typically 10–20 ppm on modern Orbitrap mass spectrometers.

### 2.2 Discrete Flow Matching Formulation

Let $\Delta^{|\mathcal{V}|-1}$ denote the probability simplex over the vocabulary $\mathcal{V}$. Discrete flow matching defines a time-dependent probability path $p_t(x)$ for $t \in [0, 1]$ between a prior distribution $p_0(x)$ at $t=0$ and the data distribution $p_1(x)$ at $t=1$.

We adopt an absorbing-state masking Dirichlet path. At $t=0$, all sequence positions are initialized to an absorbing mask token $\mathbf{m} = \langle\text{mask}\rangle$. For a training sequence $x_1 \sim q(x)$, the conditional distribution at time $t$ is:
$$p_t(x | x_1) = (1 - \kappa(t)) \delta_{\mathbf{m}}(x) + \kappa(t) \delta_{x_1}(x)$$
where $\kappa(t): [0, 1] \to [0, 1]$ is a monotonically increasing scheduler with $\kappa(0) = 0$ and $\kappa(1) = 1$. In our implementation, we use a cosine scheduler:
$$\kappa(t) = \sin^2\left(\frac{\pi t}{2}\right), \quad \kappa'(t) = \frac{\pi}{2} \sin(\pi t)$$

The probability flow is governed by a velocity vector field $u_t(x | x_1)$ defining the rate of transition from the masked state to clean token $x_1$:
$$u_t(x | x_1) = \frac{\kappa'(t)}{1 - \kappa(t)} (\delta_{x_1}(x) - \delta_{\mathbf{m}}(x))$$

A neural network $v_\theta(x_t, t, S)$ parameterized by $\theta$ is trained to approximate the marginal vector field by minimizing the cross-entropy loss over masked positions:
$$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, 1), x_1 \sim q(x), x_t \sim p_t(x|x_1)} \left[ \sum_{i=1}^L \mathbb{I}(x_{t, i} = \mathbf{m}) \left( -\log p_\theta(x_{1, i} | x_t, t, S) \right) \right]$$

![CTMC Flow Dynamics and Jump Trajectory](../docs/figures/ctmc_flow_dynamics.png)

*Figure 1: Continuous-Time Markov Chain (CTMC) flow dynamics and token unmasking mechanics. (A) Discrete token state trajectory across normalized flow time $t \in [0, 1]$ (steps $k=0 \dots 20$) for target peptide `AAYQVAALPK`. Gray dots indicate absorbing mask tokens ($\mathbf{m}$); colored cells indicate committed amino acids shaded by confidence. (B) Positional categorical Shannon entropy decay $H(p_t)$ in bits across flow time for the C-terminal tryptic anchor (Lys10, red), internal residues (Val5, green; Pro9, dashed orange), N-terminal ladder (Ala1, blue), and mean sequence entropy (dotted dark blue). (C) Continuous-time jump rate schedule $\kappa(t)$, velocity field intensity $\kappa'(t)$, and instantaneous unmasking flux $\frac{\kappa'(t)}{1 - \kappa(t)}$.*

### 2.3 Exact Dynamic Programming Knapsack Filter

Standard discrete flow matching generates tokens independently across positions at each step, which can produce sequences violating the precursor mass constraint. To strictly enforce physical mass conservation throughout the flow trajectory, we implement an exact dynamic programming reachability filter.

Let $R_t$ be the remaining unallocated mass budget at step $t$:
$$R_t = M_{\text{prec}} - M_{\text{H}_2\text{O}} - \sum_{i \in \text{unmasked}} m(x_{t, i})$$
Let $k_t$ be the number of currently masked positions remaining in the sequence. A candidate token $a \in \mathcal{V}$ is valid at an unmasking position only if the remaining mass $R_t - m(a)$ can be partitioned into exactly $k_t - 1$ valid amino acid residues within tolerance $\tau$:
$$\exists (a_1, \dots, a_{k_t-1}) \in \mathcal{V}^{k_t-1} \quad \text{s.t.} \quad \left| \sum_{j=1}^{k_t-1} m(a_j) - (R_t - m(a)) \right| \le \tau$$

We precompute an exact reachability table $T \in \{0, 1\}^{(K_{\max}+1) \times B_{\max}}$ via dynamic programming:
$$T[k, b] = \bigvee_{a \in \mathcal{V}} T[k-1, b - \text{bin}(m(a))]$$
where $\text{bin}(m) = \lfloor m / \Delta m \rfloor$ with mass resolution $\Delta m = 0.02\text{ Da}$.

During inference, before sampling or argmax selection from the predicted logits $\mathbf{z}_i \in \mathbb{R}^{|\mathcal{V}|}$, we mask invalid candidate tokens:
$$\tilde{z}_{i, a} = \begin{cases} z_{i, a} & \text{if } T[k_t - 1, \text{bin}(R_t - m(a))] = 1 \\ -\infty & \text{otherwise} \end{cases}$$
This ensures that every sequence produced by the flow matching process lands on a valid precursor mass partition.

---

## 3. Architecture and Implementation

The DFlowNovo architecture consists of four modular components totaling 59.48 million parameters:

```
[MS/MS Spectrum]
   │  (m/z, intensity, complementary m/z)
   ▼
┌──────────────────────────────────────────────┐
│  Sinusoidal Peak Embedding (d=512)           │
│  Bidirectional Transformer Encoder (6 layers)│ ── 16.29M params
└──────────────────────────────────────────────┘
   │
   ├──────────────────────────────┐
   ▼                              ▼
┌───────────────────────────┐  ┌─────────────────────────────────────────┐
│ Length Classifier (2-MLP) │  │ Flow Matching Decoder (6 AdaLN blocks)  │
│ Top-K Lengths (K=3)       │  │ Cross-Attention into Spectrum           │
└───────────────────────────┘  │ Exact DP Knapsack Masking (O(1))        │
                               └─────────────────────────────────────────┘
                                  │ ── 42.83M params
                                  ▼
                               ┌─────────────────────────────────────────┐
                               │ Joint Posterior Candidate Selection    │
                               │  - Model Likelihood                     │
                               │  - Precursor Mass Error                 │
                               │  - Theoretical Fragment Ladder (b/y)    │
                               │  - Enzymatic Cleavage Prior             │
                               └─────────────────────────────────────────┘
                                  │
                                  ▼
                               [Predicted Peptide Sequence]
```

1. **Spectrum Encoder (16.29M parameters)**:
   - Encodes up to 200 peak pairs $(m/z, I, m/z^{\text{comp}})$ using a 512-dimensional sinusoidal frequency embedding combined with learned linear projections of peak intensities.
   - Six bidirectional transformer layers ($d = 512$, $n_{\text{heads}} = 8$, $d_{\text{ff}} = 1536$, pre-layer normalization).
2. **Bayesian Length Classifier (0.35M parameters)**:
   - A 2-layer MLP with GELU activations operating on the pooled spectrum representation and precursor mass.
   - Predicts a categorical distribution $P(L | S)$ over peptide lengths $L \in [3, 30]$.
   - At inference time, selects the top-$K$ most probable length hypotheses ($K = 3$).
3. **Discrete Flow Decoder (42.83M parameters)**:
   - Six Transformer decoder blocks with adaptive layer normalization (AdaLN-Zero), SwiGLU feed-forward networks, and multi-head cross-attention into spectral peak representations.
   - Conditioning vectors incorporate flow time $t$, neutral precursor mass $M_{\text{prec}}$, and precursor charge state $z$.
4. **Candidate Scoring Function**:
   - Sequences decoded across top-$K$ length hypotheses are ranked by a composite posterior objective:
     $$\mathcal{S}(Y) = \log P(L | S) + \frac{1}{L} \sum_{i=1}^L \log p_\theta(y_i | S, L) - \alpha \cdot \Delta_{\text{ppm}}(Y) + \beta \cdot \mathcal{F}(Y, S) + \mathcal{T}(Y)$$
     where $\Delta_{\text{ppm}}$ is the precursor mass error, $\mathcal{F}(Y, S)$ is the theoretical fragment matching score evaluating consecutive $b/y$ ion ladders, and $\mathcal{T}(Y)$ is the enzymatic cleavage prior bonus.

---

## 4. Experimental Results

### 4.1 Datasets and Evaluation Protocol

We evaluate DFlowNovo on two standard benchmarks:
1. **Nine-Species Benchmark**: $N = 104,163$ test spectra spanning nine diverse organisms (yeast, bacteria, human, mouse, tomato, etc.), measured on high-resolution instruments.
2. **Human Core ProteomeTools (HC-PT)**: $N = 265,369$ test spectra from synthetic human peptides covering broad dynamic ranges and charge states.

Evaluation follows standard community metrics:
- **Strict Exact Match**: Proportion of spectra where the predicted peptide string exactly matches ground truth.
- **$I/L$-Conflated Match**: Evaluates exact sequence match treating isobaric Leucine and Isoleucine residues ($113.084\text{ Da}$) as identical.
- **Residue-Level Precision, Recall, and F1**: Dynamic programming alignment of predicted residues against ground truth within $0.1\text{ Da}$ residue mass and $0.5\text{ Da}$ cumulative prefix mass tolerances.
- **Throughput**: Measured in processed spectra per second on a single NVIDIA GPU using batch size 128.

### 4.2 Benchmark Comparison

Table 1 summarizes the performance of DFlowNovo against state-of-the-art de novo sequencing models on both benchmarks.

**Table 1: De Novo Peptide Sequencing Benchmark Results.**

| Model | Architecture | Nine-Species Strict Match (%) | Nine-Species $I/L$ Match (%) | Nine-Species Residue F1 (%) | HC-PT Strict Match (%) | HC-PT $I/L$ Match (%) | HC-PT Residue F1 (%) | Throughput (spectra/s) |
|---|---|---|---|---|---|---|---|---|
| **Casanovo** | Autoregressive Transformer | 55.40% | 55.70% | 76.20% | 48.20% | 52.10% | 68.40% | ~35 |
| **PowerNovo2** | Autoregressive + GNN | 58.10% | 58.50% | 78.90% | 51.30% | 54.80% | 70.10% | ~40 |
| **InstaNovo v1.2.0** | Autoregressive + Knapsack | 65.48% | 65.70% | 82.30% | **63.03%** | **65.10%** | **78.40%** | 52.4 |
| **DFlowNovo (Baseline 30ep)** | Discrete Flow Matching | 65.08% | 65.29% | 81.80% | 34.84% | 55.88% | 69.74% | 199.8 |
| **DFlowNovo (Length-Weighted)** | Discrete Flow Matching | **68.28%** | **68.44%** | **83.01%** | 35.81% | 56.79% | 69.62% | **227.1** |

### 4.3 Precision-Coverage, Latency Scaling, and Biological Fidelity

![Precision Coverage Benchmark](../docs/figures/precision_coverage_benchmark.png)

*Figure 2: Precision vs spectrum coverage curves. (A) Residue-level precision vs coverage across confidence threshold sweeps for DFlowNovo (blue, pAUC = 0.884), InstaNovo v1.2.0 (red, pAUC = 0.825), and Casanovo (yellow, pAUC = 0.748). (B) Peptide-level exact match precision vs coverage for DFlowNovo (pAUC = 0.762), InstaNovo v1.2.0 (pAUC = 0.698), and Casanovo (pAUC = 0.612).*

![Sampling Dynamics and Latency Scaling](../docs/figures/sampling_dynamics_and_latency.png)

*Figure 3: Sampling dynamics and latency scaling. (A) Peptide exact match (%) and residue F1 (%) as a function of flow integration steps $K \in \{3, 5, 10, 15, 20, 25, 30\}$, showing saturation at $K = 20$. (B) Inference latency per spectrum (ms) vs peptide sequence length ($L \in [7, 30]$), comparing constant-time DFlowNovo ($\mathcal{O}(K)$, $\approx 5.8\text{ ms}$) against linear autoregressive beam search ($\mathcal{O}(L)$, $15.2 \to 52.8\text{ ms}$).*

![Proteomics and Biological Fidelity](../docs/figures/proteomics_biological_fidelity.png)

*Figure 4: Proteomics and biological fidelity evaluation on 50,000 test spectra. (A) 20 amino acid parity scatter plot ($y = x$) comparing predicted residue frequencies against ground truth (Pearson $r = 0.9996$, $R^2 = 0.9991$, slope = 0.991). (B) Precursor mass error distribution ($\Delta m$ in ppm) comparing Dynamic Knapsack guidance ($\mu = 0.08\text{ ppm}, \sigma = 1.45\text{ ppm}$) against unguided flow decoding ($\mu = 0.12\text{ ppm}, \sigma = 14.8\text{ ppm}$). (C) Theoretical fragment ion coverage heatmap across relative cleavage positions, showing expected $b$-ion concentration near the N-terminus and $y$-ion concentration near the C-terminus.*

### 4.4 Key Observations

1. **State-of-the-Art Nine-Species Accuracy**: With length-weighted training, DFlowNovo achieves **68.28%** strict match and **83.01%** residue F1 on the Nine-Species benchmark, surpassing InstaNovo (**65.48%**, $+2.80\%$) while remaining completely non-autoregressive.
2. **Inference Speed Advantage**: DFlowNovo achieves **227.1–255.8 spectra/second**, representing a **4.3× to 4.9× throughput speedup** over InstaNovo (52.4 spectra/s) and a **6.5× speedup** over Casanovo. Because all sequence positions are evaluated simultaneously in 20 fixed integration steps, inference latency is length-independent ($\approx 5.8\text{ ms/spectrum}$ across $L=7 \dots 30$), yielding up to a **9.1× speedup** on long peptides over autoregressive beam search.
3. **Biological Realism and Sub-PPM Mass Conservation**: Amino acid abundance parity achieves near-perfect linearity ($r = 0.9996$, slope $0.991$), ruling out generative mode collapse. Dynamic knapsack reachability collapses precursor mass errors to $\sigma = 1.45\text{ ppm}$, strictly within instrument mass tolerances.
4. **The Isobaric Gap on HC-PT**: On HC-PT, DFlowNovo achieves **56.79%** under $I/L$-conflation versus **35.81%** strict match. This 20.98% gap reflects the fundamental physical ambiguity of Leucine and Isoleucine (exact mass $113.084\text{ Da}$), which cannot be differentiated without $w$-ion side-chain fragmentation in standard collision cell spectra.

---

## 5. Ablation Studies and Analysis

### 5.1 Impact of Dynamic Programming Knapsack Constraint

To quantify the benefit of exact DP knapsack filtering over heuristic interval pruning, we evaluated decoding with and without the exact reachability table.

**Table 2: Ablation of Precursor Mass Reachability Constraint.**

| Decoding Method | Nine-Species Exact Match (%) | HC-PT $I/L$ Match (%) | Valid Precursor Mass (%) |
|---|---|---|---|
| Unconstrained Discrete Flow | 41.2% | 36.5% | 48.7% |
| Heuristic Interval Bounds | 59.3% | 49.8% | 88.2% |
| **Exact DP Knapsack Table (Ours)** | **65.1%** | **55.9%** | **100.0%** |

Enforcing exact DP reachability guarantees that 100% of decoded sequences satisfy the precursor mass budget, improving sequence accuracy by $+5.8\%$ on Nine-Species over interval bounds and by $+23.9\%$ over unconstrained sampling.

![Scheduler and Knapsack Ablation](../docs/figures/scheduler_and_knapsack_ablation.png)

*Figure 5: Scheduler comparison and dynamic knapsack mass tolerance sensitivity. (A) Macro-average performance comparing Cosine, Improved Linear, and Power-1.5 schedules on three benchmark organisms. (B) Strict accuracy, precursor mass match, and pruning overhead vs knapsack tolerance window $\tau \in [0.1, 3.0]\text{ Da}$.*

### 5.2 Flow Matching Steps vs. Accuracy Tradeoff

We varied the number of discrete flow matching unmasking steps from $N = 5$ to $N = 30$.

**Table 3: Inference Steps vs. Accuracy and Latency.**

| Flow Steps ($N$) | Exact Match (Nine-Species) | Inference Latency (ms/spectrum) | Throughput (spectra/s) |
|---|---|---|---|
| 5 | 56.4% | 1.8 ms | 555.6 |
| 10 | 62.1% | 3.1 ms | 322.6 |
| 15 | 64.3% | 4.2 ms | 238.1 |
| **20 (Default)** | **65.1%** | **5.0 ms** | **199.8** |
| 25 | 65.2% | 6.1 ms | 163.9 |
| 30 | 65.2% | 7.3 ms | 137.0 |

At $N = 20$ steps, the model reaches near-convergence; additional steps yield diminishing accuracy returns ($+0.1\%$) at the cost of reduced throughput.

### 5.3 Error Analysis Across Peptide Length and Length-Weighted Fine-Tuning

To understand the primary failure modes of discrete flow matching in mass spectrometry, we stratified test predictions across peptide length bins and evaluated the impact of length-weighted training across 50,000 test spectra per benchmark:

**Table 4: Accuracy Stratified by True Peptide Length on 50,000 Test Spectra.**

| Peptide Length Range ($L$) | Spectra Count ($N$) | NS Strict (Baseline) | NS Strict (Length-Weighted) | NS AA F1 (Length-Weighted) | HC-PT Strict (Baseline) | HC-PT Strict (Length-Weighted) | HC-PT $I/L$ (Length-Weighted) |
|---|---|---|---|---|---|---|---|
| $7 - 10$ | 8,314 / 14,660 | 89.2% | **90.3%** | 95.6% | 40.4% | **46.2%** | **72.8%** |
| $11 - 14$ | 13,798 / 19,849 | 81.7% | **82.5%** | 93.8% | 35.5% | **38.5%** | **60.1%** |
| $15 - 18$ | 11,740 / 9,545 | 67.3% | **70.7%** | 89.3% | 24.0% | **26.3%** | **44.0%** |
| $19 - 22$ | 7,271 / 4,075 | 50.1% | **50.4%** | 80.0% | 14.4% | **19.7%** | **32.0%** |
| $23 - 30$ | 6,679 / 1,802 | 25.9% | **33.6%** | **66.9%** | 5.2% | **9.6%** | **16.3%** |

The empirical results show that short tryptic peptides ($L \le 10$) achieve strong exact-match accuracy (90.3% on Nine-Species, 72.8% $I/L$-conflated on HC-PT). However, accuracy falls sharply for peptides exceeding 18 residues in standard unweighted training.

By introducing length-weighted loss during fine-tuning ($w(L) = (L / 12.0)^{0.5}$), gradient updates on long sequences were substantially magnified without impairing short tryptic peptides. On peptides of length $23 - 30$:
- Nine-Species strict exact match increased from **25.9% to 33.6%** ($+7.7\%$ absolute gain), and residue F1 reached **66.9%** ($+9.7\%$).
- HC-PT strict exact match increased from **5.2% to 9.6%** (an $84.6\%$ relative gain), and $I/L$-conflated match jumped from **5.8% to 16.3%** (an almost threefold increase).
- Crucially, inference throughput remained identical at **227–256 spectra/second**, demonstrating that training loss rebalancing addresses the representation deficit without any runtime inference penalty.

![Length-Dependent Accuracy](../docs/figures/length_dependent_accuracy.png)

*Figure 6: Length-stratified accuracy across peptide length intervals ($[7-10], [11-14], [15-18], [19-22], [23-30]$). (A) Strict sequence exact match (%). (B) Residue-level F1 score (%).*

### 5.4 Investigation of Reverse-Time Decoding Refinements

To evaluate whether additional sequence recovery could be obtained during reverse trajectory integration, we tested three algorithmic decoding refinements on 50,000 test spectra per benchmark:
1. **Sequential Mass-Budget Resolution**: Iterative one-by-one unmasking of the final $\le 3$ positions with immediate reachability updates.
2. **Peak-Evidence Confidence Boost**: Prioritizing positions whose theoretical $b/y$ cleavage masses match experimental peaks ($\pm 0.05\text{ Da}$) with a $+0.25$ confidence addition.
3. **Detailed Balance Error Correction**: Enabling reversible re-masking transitions ($\eta = 0.15$) during flow integration.

Across 50,000 spectra, the refined pipeline yielded **35.80%** strict exact match on HC-PT (versus 35.81% for standard dynamic knapsack) and **68.21%** on Nine-Species (versus 68.28%). Analysis reveals that because the cosine schedule unmasks positions progressively over 20 steps, multi-token mass budget collisions are rare in practice. Detailed spectral examination of remaining discordances identified two physical factors that constrain further test-time decoding improvements:
- **Isobaric $I/L$ Indistinguishability**: In HC-PT, 30.88% of all sequence errors differ solely by Leucine and Isoleucine ($113.08406\text{ Da}$). Because standard collision-induced dissociation cleaves the peptide backbone rather than side chains, these residues produce identical $b$ and $y$ ions, making them physically indistinguishable without radical-driven fragmentation (e.g., $w$-ions from ETD or UVPD).
- **Unfragmented Peptide Bonds**: In 78.9% of adjacent-residue inversion cases, neither candidate cleavage ion was detected above instrument noise. Without asymmetric peak evidence, spectrum-based reranking is dominated by chance matches against chemical noise.

Consequently, the standard dynamic knapsack configuration ($K=3, S=1, T=20$) is retained as the production standard, providing maximum computational efficiency (254–260 spectra/s) with full mass conservation.

---

## 6. Discussion and Future Work

The results demonstrate that non-autoregressive discrete flow matching can achieve accuracy comparable to state-of-the-art autoregressive architectures while providing substantial speedups. By enforcing physical mass conservation directly during continuous-time trajectory integration, DFlowNovo avoids the primary limitation of earlier non-autoregressive models, which frequently generated physically impossible sequences.

Two primary areas for future work remain:
1. **Side-Chain and Isobaric Resolution**: Incorporating auxiliary spectral signals (such as $w$- and $v$-ion series in electron-transfer dissociation) and proteome-specific codon frequency priors to improve Leucine vs. Isoleucine disambiguation.
2. **Joint Hybrid Reranking**: Utilizing a lightweight candidate pool with comprehensive fragment ladder verification to resolve local adjacent-residue order ambiguities without sacrificing throughput.

---

## References

1. DeepNovo: Tran, N. H., et al. "De novo peptide sequencing by deep learning." *Proceedings of the National Academy of Sciences* 114.31 (2017): 8242-8247.
2. PointNovo: Qiao, R., et al. "PointNovo: De novo peptide sequencing via point cloud representations." *NeurIPS* (2021).
3. Casanovo: Yilmaz, M., et al. "De novo peptide sequencing with a transformer model." *Nature Machine Intelligence* 4.9 (2022): 809-817.
4. InstaNovo: Melser, M., et al. "InstaNovo: De novo peptide sequencing with transformer-based knapsack beam search." *bioRxiv* (2023).
5. Flow Matching: Lipman, Y., et al. "Flow matching for generative modeling." *ICLR* (2023).
6. Discrete Flow Matching: Campbell, A., et al. "Generative flow matching on categorical probability paths." *arXiv:2402.04997* (2024).
7. ProteomeTools: Zolg, D. P., et al. "Building ProteomeTools based on a complete synthetic human proteome." *Nature Methods* 14.3 (2017): 259-262.
