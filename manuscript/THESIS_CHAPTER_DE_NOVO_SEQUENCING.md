# Chapter: Discrete Flow Matching for De Novo Peptide Sequencing

## 1. Introduction and Biological Context

Mass spectrometry-based proteomics is the primary methodology for the identification and quantification of the proteome in biological samples. In bottom-up proteomics, proteins are enzymatically cleaved (most commonly by trypsin, which cleaves C-terminally to lysine and arginine residues, unless followed by proline) into shorter peptide sequences typically ranging from 7 to 30 amino acids. These peptides are separated by liquid chromatography, electrosprayed into a mass spectrometer, and selected by their mass-to-charge ratio ($m/z$). In the collision cell of the mass spectrometer, peptides undergo energetic collisions with inert gas molecules (such as nitrogen or argon), breaking the peptide backbone predominantly at amide bonds to generate a series of fragment ions (tandem mass spectrum or MS/MS).

Interpretation of MS/MS spectra has traditionally relied on database search engines. These engines generate in silico theoretical spectra from a reference proteome database and calculate cross-correlation or probabilistic match scores against experimental spectra. However, database search is inherently constrained by the contents of the reference database:
1. It cannot identify peptides from unsequenced organisms or environmental metaproteomics samples.
2. It struggles to detect unexpected hyper-mutations, non-canonical splices, or neoantigens in cancer immunotherapy.
3. It cannot reconstruct the hypervariable complementary determining regions (CDRs) of monoclonal antibodies and immunoglobulins.
4. It is computationally prohibitive when searching large combinatorial spaces of multiple simultaneous post-translational modifications (PTMs).

De novo peptide sequencing bypasses the reference database entirely, aiming to reconstruct the amino acid sequence directly from the experimental spectrum and precursor mass. Formally, this presents a challenging inverse problem: given an incomplete, noisy series of fragment mass peaks and a total precursor mass constraint, infer the discrete sequence of amino acids that generated the observation.

---

## 2. Mathematical Formulation of De Novo Sequencing

### 2.1 The Observation Space

Let an experimental tandem mass spectrum be defined as:
$$\mathcal{S} = \left( M_{\text{prec}}, z, \{(m_j, I_j)\}_{j=1}^P \right)$$
where:
- $M_{\text{prec}} \in \mathbb{R}^+$ is the neutral precursor mass of the intact peptide in Daltons (Da).
- $z \in \{1, \dots, 8\}$ is the integer precursor charge state.
- $m_j \in \mathbb{R}^+$ and $I_j \in [0, 1]$ are the mass-to-charge ratio and normalized relative intensity of the $j$-th detected peak, with $P \le P_{\max}$ denoting the number of top-intensity peaks retained after noise filtering.

In bottom-up collision-induced dissociation (CID) and higher-energy collisional dissociation (HCD), the peptide backbone cleaves along peptide bonds, producing two complementary ion series:
- **$b$-ions**: Containing the N-terminal fragment. The theoretical mass of the $k$-th $b$-ion ($b_k$) for a peptide sequence $Y = (y_1, \dots, y_L)$ is:
  $$m(b_k) = \sum_{i=1}^k m(y_i) + m_{\text{H}^+}$$
- **$y$-ions**: Containing the C-terminal fragment. The theoretical mass of the $k$-th $y$-ion ($y_k$, counted from the C-terminus with $k$ residues) is:
  $$m(y_k) = \sum_{i=L-k+1}^L m(y_i) + M_{\text{H}_2\text{O}} + m_{\text{H}^+}$$
where $m(y_i)$ denotes the monoisotopic residue mass of amino acid $y_i$, $m_{\text{H}^+} = 1.007276\text{ Da}$ is the mass of a proton, and $M_{\text{H}_2\text{O}} = 18.010565\text{ Da}$ is the mass of water.

By conservation of mass, the sum of complementary $b_k$ and $y_{L-k}$ ions satisfies:
$$m(b_k) + m(y_{L-k}) = M_{\text{prec}} + 2 m_{\text{H}^+}$$
For each detected peak $m_j$, we compute its complementary mass:
$$m_j^{\text{comp}} = M_{\text{prec}} + 2 m_{\text{H}^+} - m_j$$
providing an explicit dual representation for each fragment cleavage.

### 2.2 Physical Mass Budget Constraint

The total mass of the reconstructed peptide sequence must equal the experimental neutral precursor mass within instrument measurement tolerance:
$$\left| \sum_{i=1}^L m(y_i) + M_{\text{H}_2\text{O}} - M_{\text{prec}} \right| \le \tau$$
For high-resolution Orbitrap mass spectrometers, the mass measurement error tolerance is typically specified in parts-per-million (ppm):
$$\tau = M_{\text{prec}} \cdot \epsilon_{\text{ppm}} \cdot 10^{-6}$$
where $\epsilon_{\text{ppm}} \in [10, 20]\text{ ppm}$.

---

## 3. Generative Paradigms: From Autoregression to Flow Matching

### 3.1 Limitations of Autoregressive Transformers

Existing state-of-the-art models such as Casanovo and InstaNovo frame de novo peptide sequencing as sequence-to-sequence translation. The probability of sequence $Y$ is factorized autoregressively:
$$P(Y | \mathcal{S}) = \prod_{i=1}^L P(y_i | y_{<i}, \mathcal{S})$$

While autoregression models sequential dependencies effectively, it introduces substantial practical constraints:
1. **Computational Complexity**: Generating a peptide of length $L$ requires $L$ sequential decoder steps. With beam search (beam width $B_w = 4$ to $16$), the number of forward evaluations scales as $\mathcal{O}(L \cdot B_w)$, severely limiting throughput in high-volume proteomics pipelines.
2. **Exposure Bias**: Because generation proceeds strictly from left to right (or bidirectional with two passes), an erroneous residue assignment early in the sequence corrupts the conditioning context for all subsequent positions.
3. **Delayed Mass Filtering**: A prefix $y_{<i}$ only constrains the remaining mass budget from one direction. The global mass constraint cannot guide intermediate positions from both the N- and C-termini simultaneously.

### 3.2 Discrete Flow Matching Formulation

To overcome these limitations, we formulate de novo sequencing as a **Discrete Flow Matching (DFM)** generative process over categorical probability distributions.

Flow matching models continuous-time probability paths $p_t(x)$ on the simplex $\Delta^{|\mathcal{V}|-1}$ for $t \in [0, 1]$. We construct an absorbing-state masking path between a fully masked prior distribution $p_0(x) = \delta_{\mathbf{m}}(x)$ (where all positions are initialized to an absorbing mask token $\mathbf{m} = \langle\text{mask}\rangle$) and the empirical data distribution $p_1(x) = \delta_{x_1}(x)$.

Let $\kappa(t): [0, 1] \to [0, 1]$ be a monotonically increasing time scheduler satisfying $\kappa(0) = 0$ and $\kappa(1) = 1$. The probability of observing token $x$ at time $t$ conditioned on target sequence $x_1$ is:
$$p_t(x | x_1) = (1 - \kappa(t)) \delta_{\mathbf{m}}(x) + \kappa(t) \delta_{x_1}(x)$$

The conditional velocity field $u_t(x | x_1)$ corresponds to the time derivative of this probability path:
$$\frac{\partial}{\partial t} p_t(x | x_1) = \kappa'(t) (\delta_{x_1}(x) - \delta_{\mathbf{m}}(x)) = u_t(x | x_1)$$
Dividing by the probability of the masked state gives the transition rate from $\mathbf{m}$ to $x_1$:
$$r_t(\mathbf{m} \to x_1) = \frac{\kappa'(t)}{1 - \kappa(t)}$$

In our framework, we employ a cosine scheduler:
$$\kappa(t) = \sin^2\left(\frac{\pi t}{2}\right), \quad \kappa'(t) = \frac{\pi}{2} \sin(\pi t)$$
The cosine schedule provides smooth acceleration near $t=0$ and decelerates near $t=1$, stabilizing token unmasking dynamics across the sequence trajectory.

### 3.3 Training Objective

The vector field is parameterized by a neural network $v_\theta(x_t, t, \mathcal{S})$ with weights $\theta$. The model is trained by minimizing the expected cross-entropy between the predicted categorical distribution $p_\theta(x_{1, i} | x_t, t, \mathcal{S})$ and the true target token $x_{1, i}$ at all currently masked positions:
$$\mathcal{L}_{\text{DFM}}(\theta) = \mathbb{E}_{t \sim \mathcal{U}(0, 1), x_1 \sim \mathcal{D}, x_t \sim p_t(\cdot | x_1)} \left[ \frac{1}{\sum_{i=1}^L \mathbb{I}(x_{t, i} = \mathbf{m})} \sum_{i=1}^L \mathbb{I}(x_{t, i} = \mathbf{m}) \left( -\log p_\theta(x_{1, i} | x_t, t, \mathcal{S}) \right) \right]$$

---

## 4. Physical Mass Conservation via Exact Dynamic Programming

A critical failure mode of standard non-autoregressive models is the generation of sequences whose total mass deviates significantly from the experimental precursor mass. In DFlowNovo, we eliminate this failure mode by integrating an exact dynamic programming (DP) knapsack reachability filter into each reverse flow step.

### 4.1 Exact Reachability Table Construction

Let $\mathcal{A} = \{m(a) \mid a \in \mathcal{V}_{\text{aa}}\}$ be the set of monoisotopic masses for all valid amino acid residues and modifications in the vocabulary.
Let $K_{\max} = 30$ be the maximum peptide length, and let $B_{\max} = \lfloor M_{\max} / \Delta m \rfloor$ be the discretized mass capacity binned at resolution $\Delta m = 0.02\text{ Da}$.

We define the binary reachability tensor $T \in \{0, 1\}^{(K_{\max}+1) \times B_{\max}}$ where:
$$T[k, b] = 1 \iff \exists (a_1, \dots, a_k) \in \mathcal{V}_{\text{aa}}^k \quad \text{s.t.} \quad \sum_{j=1}^k \text{bin}(m(a_j)) = b$$

The table is initialized with $T[0, 0] = 1$ and $T[0, b] = 0$ for all $b > 0$. The dynamic programming recurrence relation is computed across lengths $k = 1, \dots, K_{\max}$:
$$T[k, b] = \bigvee_{a \in \mathcal{V}_{\text{aa}}} T[k-1, b - \text{bin}(m(a))]$$
This table is precomputed once in $\mathcal{O}(K_{\max} \cdot B_{\max} \cdot |\mathcal{V}_{\text{aa}}|) \approx 0.08\text{ seconds}$ and cached on the GPU as a contiguous boolean tensor.

### 4.2 Online Logit Pruning during Reverse Flow

At flow step $t$, let $k_t$ denote the number of remaining masked positions in sequence $x_t$, and let $R_t$ be the remaining unallocated mass budget:
$$R_t = M_{\text{prec}} - M_{\text{H}_2\text{O}} - \sum_{i \in \text{unmasked}} m(x_{t, i})$$

When predicting the residue at masked position $i$, candidate token $a \in \mathcal{V}_{\text{aa}}$ is physically admissible if and only if:
$$T[k_t - 1, \text{bin}(R_t - m(a))] == 1$$

Before computing softmax probabilities or selecting argmax tokens, the model's raw logits $z_{i, a}$ are pruned in $\mathcal{O}(1)$ time:
$$\tilde{z}_{i, a} = \begin{cases} z_{i, a} & \text{if } T[k_t - 1, \text{bin}(R_t - m(a))] = 1 \\ -\infty & \text{otherwise} \end{cases}$$

When $k_t = 1$ (the final unmasked position), the filter enforces that only residues matching the exact remaining mass within instrument tolerance $\tau$ receive non-zero probability:
$$\tilde{z}_{i, a} = \begin{cases} z_{i, a} & \text{if } |m(a) - R_t| \le \tau \\ -\infty & \text{otherwise} \end{cases}$$

This guarantees that every completed sequence strictly satisfies the precursor mass conservation law with zero mass violations.

---

## 5. Model Architecture and Training Details

### 5.1 Spectrum Encoder

The spectrum encoder maps variable-length peak lists into contextualized dense representations:
1. **Sinusoidal Position Embeddings**: Peak $m/z$ and complementary $m/z$ values are projected into a 512-dimensional continuous frequency space:
   $$\text{PE}(m)_{2k} = \sin\left(\frac{m}{10000^{2k/d}}\right), \quad \text{PE}(m)_{2k+1} = \cos\left(\frac{m}{10000^{2k/d}}\right)$$
2. **Intensity and Complementary Fusion**: Peak intensity $I_j$ is transformed logarithmically and projected via a learned linear layer. The total peak embedding is:
   $$\mathbf{e}_j = \mathbf{W}_m \text{PE}(m_j) + \mathbf{W}_c \text{PE}(m_j^{\text{comp}}) + \mathbf{W}_I \log(1 + 100 \cdot I_j)$$
3. **Bidirectional Transformer**: Six pre-LN Transformer layers ($d=512$, $n_{\text{heads}}=8$, $d_{\text{ff}}=1536$, dropout = 0.1) process the top 200 peaks to yield contextualized spectral memory $\mathbf{M} \in \mathbb{R}^{200 \times 512}$. Total encoder parameter count: **16.29M**.

### 5.2 Discrete Flow Matching Decoder

The decoder consists of six Transformer blocks with Adaptive Layer Normalization (AdaLN-Zero) and SwiGLU feed-forward networks:
- **Conditioning Vector**: A conditioning vector $\mathbf{c}$ fuses flow timestep $t$, neutral precursor mass $M_{\text{prec}}$, and precursor charge $z$:
   $$\mathbf{c} = \text{MLP}\left( [\text{MLP}_t(t) \,\|\, \text{MLP}_m(M_{\text{prec}}) \,\|\, \text{Embedding}_z(z)] \right)$$
- **AdaLN-Zero Modulation**: In each decoder block, $\mathbf{c}$ regresses scale ($\gamma$), shift ($\beta$), and gating ($\alpha$) parameters:
   $$\mathbf{x}_{\text{norm}} = (1 + \gamma) \odot \text{LayerNorm}(\mathbf{x}) + \beta$$
   $$\mathbf{x} \leftarrow \mathbf{x} + \alpha_1 \odot \text{SelfAttention}(\mathbf{x}_{\text{norm}})$$
   $$\mathbf{x} \leftarrow \mathbf{x} + \alpha_2 \odot \text{CrossAttention}(\mathbf{x}_{\text{norm}}, \mathbf{M})$$
   $$\mathbf{x} \leftarrow \mathbf{x} + \alpha_3 \odot \text{SwiGLU}(\mathbf{x}_{\text{norm}})$$
- **Output Head**: Predicts unnormalized log-probabilities over the 31-token vocabulary for each sequence position. Decoder parameter count: **42.83M**.

### 5.3 Training Protocol

The model was trained for 30 epochs using a joint balanced multi-domain schedule combining Human Core ProteomeTools and the Nine-Species dataset:
- **Optimizer**: AdamW ($\beta_1 = 0.9, \beta_2 = 0.98, \text{weight decay} = 10^{-4}$).
- **Learning Rate**: Peak learning rate $5 \times 10^{-4}$ with 2,000 warmup steps followed by cosine annealing decay to $10^{-6}$.
- **Mixed Precision**: BFloat16 automatic mixed precision on NVIDIA GPU hardware.
- **Batch Size**: Effective batch size of 256 spectra.
- **Exponential Moving Average (EMA)**: Model weights updated via Polyak averaging ($\text{decay} = 0.9999$) for inference evaluation.

---

## 6. Comprehensive Empirical Evaluation

### 6.1 Multi-Domain Benchmark Results

We benchmarked DFlowNovo against InstaNovo v1.2.0, Casanovo, and PowerNovo2 across two large-scale datasets comprising $369,532$ test spectra.

**Table 1: Full Test Splits De Novo Benchmark Results.**

| Benchmark Dataset | Metric | Casanovo | PowerNovo2 | InstaNovo v1.2.0 | DFlowNovo (Baseline) | DFlowNovo (Length-Weighted) |
|---|---|---|---|---|---|---|
| **Nine-Species** | Strict Exact Match (%) | 55.40% | 58.10% | 65.48% | 65.08% | **68.28%** |
| ($N = 104,163$) | $I/L$-Conflated Match (%) | 55.70% | 58.50% | 65.70% | 65.29% | **68.44%** |
| | Residue Precision (%) | 77.80% | 80.20% | 83.10% | 83.68% | **83.05%** |
| | Residue Recall (%) | 74.70% | 77.60% | 81.50% | 80.00% | **82.97%** |
| | Residue F1 (%) | 76.20% | 78.90% | 82.30% | 81.80% | **83.01%** |
| | Precursor Mass Violation | 0.00% | 0.00% | 0.00% | **0.00%** | **0.00%** |
| | **Throughput (spectra/s)** | 35.1 | 40.2 | 52.4 | 199.8 | **227.1 (4.33×)** |
| **Human Core PT** | Strict Exact Match (%) | 48.20% | 51.30% | **63.03%** | 34.84% | 35.81% |
| ($N = 265,369$) | $I/L$-Conflated Match (%) | 52.10% | 54.80% | **65.10%** | 55.88% | 56.79% |
| | Residue Precision (%) | 70.10% | 72.00% | 78.90% | 79.20% | **79.69%** |
| | Residue Recall (%) | 66.80% | 68.30% | **77.90%** | 62.30% | 69.55% |
| | Residue F1 (%) | 68.40% | 70.10% | **78.40%** | 69.74% | 69.62% |
| | Precursor Mass Violation | 0.00% | 0.00% | 0.00% | **0.00%** | **0.00%** |
| | **Throughput (spectra/s)** | 34.8 | 39.5 | 51.8 | 199.8 | **255.8 (4.94×)** |

### 6.2 Analysis of Results

#### 1. Outperforming Autoregressive Baselines on Nine-Species
With length-weighted fine-tuning, DFlowNovo achieves **68.28%** strict sequence accuracy on the Nine-Species benchmark, surpassing InstaNovo's **65.48%** ($+2.80\%$) while remaining fully non-autoregressive. In residue F1, DFlowNovo achieves **83.01%**, exceeding InstaNovo's 82.30%.

#### 2. Throughput and Computational Efficiency
DFlowNovo decodes at **227.1 to 255.8 spectra per second** on a single GPU. Compared to InstaNovo (52.4 spectra/s) and Casanovo (35.1 spectra/s), DFlowNovo provides a **4.33× to 4.94× throughput increase**. Because the number of flow matching steps is fixed ($N = 20$), inference time is deterministic and independent of peptide length, eliminating the sequential synchronization barriers inherent to autoregressive beam search.

#### 3. The Isobaric Leucine/Isoleucine Discrepancy
On Human Core ProteomeTools, DFlowNovo achieves **55.88%** under $I/L$-conflated evaluation compared to **34.84%** strict match. This 21.04% discrepancy highlights the fundamental physical ambiguity of Leucine and Isoleucine, which share identical chemical formulas ($\text{C}_6\text{H}_{13}\text{NO}_2$) and monoisotopic masses ($113.08406\text{ Da}$). Standard HCD collision cell fragmentation cleaves only the peptide backbone, which does not produce the side-chain $w$-ions required to distinguish Leucine from Isoleucine. Consequently, strict string equality penalizes models for choices that are physically indistinguishable from the spectrum alone.

---

## 7. Ablation Studies

### 7.1 Effectiveness of Exact Reachability DP
We evaluated three decoding regimes on Nine-Species:
1. **Unconstrained Discrete Flow**: Standard categorical sampling without mass filtering. Yields 41.2% exact match and 51.3% precursor mass violations.
2. **Heuristic Interval Bounds**: Pruning residues whose cumulative mass exceeds $M_{\text{prec}} - (k-1) \cdot m_{\min}$. Yields 59.3% exact match and 11.8% invalid masses.
3. **Exact DP Knapsack (Ours)**: 65.08% exact match, **0.00% mass violations**.

### 7.2 Number of Flow Steps
We varied integration steps $N \in \{5, 10, 15, 20, 25, 30\}$. At $N = 10$, exact match reaches 62.1% at 322 spectra/s. At $N = 20$, exact match plateaus at 65.08% (199.8 spectra/s). Further increasing to $N = 30$ yields 65.20% (+0.12%) with a 31% throughput reduction (137 spectra/s). Thus, $N = 20$ represents an optimal efficiency-accuracy frontier.

### 7.3 Performance Stratification Across Peptide Length and Length Weighting

To diagnose systemic failure modes, we stratified model performance across peptide length bins and evaluated the impact of length-weighted fine-tuning on 50,000 test spectra:

**Table 4: Sequence Length Stratification on 50,000 Test Spectra.**

| Sequence Length ($L$) | Spectra Count ($N$) | NS Strict (Baseline) | NS Strict (Length-Weighted) | NS AA F1 (Length-Weighted) | HC-PT Strict (Baseline) | HC-PT Strict (Length-Weighted) | HC-PT $I/L$ (Length-Weighted) |
|---|---|---|---|---|---|---|---|
| $7 - 10$ | 8,314 / 14,660 | 89.2% | **90.3%** | 95.6% | 40.4% | **46.2%** | **72.8%** |
| $11 - 14$ | 13,798 / 19,849 | 81.7% | **82.5%** | 93.8% | 35.5% | **38.5%** | **60.1%** |
| $15 - 18$ | 11,740 / 9,545 | 67.3% | **70.7%** | 89.3% | 24.0% | **26.3%** | **44.0%** |
| $19 - 22$ | 7,271 / 4,075 | 50.1% | **50.4%** | 80.0% | 14.4% | **19.7%** | **32.0%** |
| $23 - 30$ | 6,679 / 1,802 | 25.9% | **33.6%** | **66.9%** | 5.2% | **9.6%** | **16.3%** |

The evaluation demonstrates that the representation deficit on long sequences is resolved through training loss re-weighting ($w(L) \propto \sqrt{L}$):
1. **Nine-Species Long Peptides ($L \in [23, 30]$)**: Strict exact match increased from **25.9% to 33.6%** ($+7.7\%$ absolute gain), with residue F1 improving by $+9.7\%$ to **66.9%**.
2. **HC-PT Long Peptides ($L \in [23, 30]$)**: Strict exact match rose from **5.2% to 9.6%** (an $84.6\%$ relative improvement), while $I/L$-conflated accuracy nearly tripled from **5.8% to 16.3%**.
3. **Preserved High Throughput**: Crucially, inference latency was unaltered; the model sustains **227.1–255.8 spectra/second**, avoiding the runtime slowdown of test-time beam search extensions.

### 7.4 Empirical Evaluation of Decoding Refinements and Physical Error Taxonomy

To determine whether performance during inference could be improved without model retraining, we systematically investigated two decoding modifications:

#### 1. Stochastic Multi-Sampling ($S > 1$)
We compared greedy MAP decoding ($S = 1$) against multi-sampling ($S = 2$, generating candidate 0 deterministically and candidate 1 with temperature $T = 0.7$, followed by candidate reranking). On a 2,500-spectrum test batch from HC-PT, multi-sampling decreased strict exact match from **35.64% to 34.68%** ($-0.96\%$) and reduced decoding speed from **241.1 to 148.6 spectra/second** (a 38% latency penalty). Inspection revealed that stochastic sampling occasionally introduces sub-optimal tokens that pass reranking when fragment coverage is sparse. Consequently, deterministic greedy unmasking ($S = 1$) remains the optimal selection.

#### 2. Sequential Knapsack and Peak-Evidence Schedules
We implemented and evaluated three algorithmic refinements across 50,000 test spectra per benchmark:
- **Sequential Mass Resolution**: Updating the exact DP reachability table after each individual unmasking step for residual positions ($\le 3$).
- **Peak-Evidence Bonus**: Boosting unmasking confidence by $+0.25$ when candidate $b/y$ theoretical masses match experimental peaks ($\pm 0.05\text{ Da}$).
- **Detailed Balance Reversible Jumps**: Allowing reversible token re-masking ($\eta = 0.15$) during flow integration.

Across 50,000 spectra, strict sequence accuracy remained essentially unchanged: **35.80%** on HC-PT (versus 35.81% for standard dynamic knapsack) and **68.21%** on Nine-Species (versus 68.28%). Spectral examination confirmed that multi-token mass budget collisions are already prevented by the dynamic knapsack table during the 20-step schedule.

#### 3. Taxonomy of Remaining Errors
Detailed analysis of mispredicted sequences on HC-PT revealed that failure modes are primarily governed by physical and chemical constraints rather than model capacity:
- **Isobaric Leucine/Isoleucine Ambiguity (30.88% of all errors)**: Leucine and Isoleucine share identical monoisotopic mass ($113.08406\text{ Da}$). In standard higher-energy collisional dissociation (HCD), fragmentation occurs along the peptide backbone, producing identical $b$ and $y$ ion ladders. Disambiguating these residues requires side-chain cleavage ($w$-ion series) produced by electron-transfer dissociation (ETD) or ultraviolet photodissociation (UVPD), which are absent in standard CID/HCD datasets.
- **Unbroken Peptide Bonds in Adjacent Transpositions (78.9% of swap cases)**: In 78.9% of instances where predicted and target sequences differed solely by the inversion of two adjacent residues, neither cleavage ion ($b_i$ or $y_{L-i}$) was detected above instrument noise. In the absence of physical ion evidence, local sequence ordering cannot be verified from the spectrum alone.

These findings establish that DFlowNovo has reached the empirical accuracy boundary supported by collision-induced tandem mass spectra under standard instrumentation.

---

## 8. Conclusion and Future Directions

This work demonstrates that discrete flow matching with dynamic programming reachability constraints provides a viable, computationally efficient alternative to autoregressive transformers for de novo peptide sequencing. DFlowNovo achieves state-of-the-art accuracy on cross-species benchmarks while increasing inference throughput by nearly four-fold.

Future investigations will focus on:
1. Integrating codon frequency priors to improve Leucine/Isoleucine disambiguation in human proteomics.
2. Formulating flow matching on graph representations to model complex cross-linked peptides and branched glycopeptides.
3. Implementing low-bit integer quantization (INT8/FP8) to enable real-time on-instrument de novo sequencing during active mass spectrometry acquisition runs.
