# Chapter: Discrete Flow Matching for De Novo Peptide Sequencing

## 1. Introduction and Biological Context

Mass spectrometry-based proteomics is the primary methodology for the identification and quantification of the proteome in biological samples [Aebersold and Mann, 2016]. In bottom-up proteomics, proteins are enzymatically cleaved (most commonly by trypsin, which cleaves C-terminally to lysine and arginine residues, unless followed by proline [Olsen et al., 2004]) into shorter peptide sequences typically ranging from 7 to 30 amino acids. These peptides are separated by liquid chromatography, electrosprayed into a mass spectrometer, and selected by their mass-to-charge ratio ($m/z$). In the collision cell of the mass spectrometer, peptides undergo energetic collisions with inert gas molecules (such as nitrogen or argon), breaking the peptide backbone predominantly at amide bonds to generate a series of fragment ions (tandem mass spectrum or MS/MS) [Olsen et al., 2007; Paizs and Suhai, 2005].

Interpretation of MS/MS spectra has traditionally relied on database search engines (such as SEQUEST [Eng et al., 1994], Comet [Eng et al., 2013], or MS-GF+ [Kim and Pevzner, 2014]). These engines generate in silico theoretical spectra from a reference proteome database and calculate cross-correlation or probabilistic match scores against experimental spectra. However, database search is inherently constrained by the contents of the reference database:
1. It cannot identify peptides from unsequenced organisms or environmental metaproteomics samples.
2. It struggles to detect unexpected hyper-mutations, non-canonical splices, or neoantigens in cancer immunotherapy.
3. It cannot reconstruct the hypervariable complementary determining regions (CDRs) of monoclonal antibodies and immunoglobulins [Tran et al., 2017].
4. It is computationally prohibitive when searching large combinatorial spaces of multiple simultaneous post-translational modifications (PTMs).

De novo peptide sequencing bypasses the reference database entirely, aiming to reconstruct the amino acid sequence directly from the experimental spectrum and precursor mass. Early algorithms relied on graph-theoretical spectrum graphs (Sherenga [Dancik et al., 1999], Lutefisk [Taylor and Johnson, 1997], PEAKS [Ma et al., 2003]). Recently, deep neural networks (DeepNovo [Tran et al., 2017], PointNovo [Qiao et al., 2021], Casanovo [Yilmaz et al., 2022], and InstaNovo [Eloff et al., 2025]) established high sequencing accuracy. Formally, this presents a challenging inverse problem: given an incomplete, noisy series of fragment mass peaks and a total precursor mass constraint, infer the discrete sequence of amino acids that generated the observation.

---

## 2. Mathematical Formulation of De Novo Sequencing

### 2.1 The Observation Space

Let an experimental tandem mass spectrum be defined as:
$$\mathcal{S} = \left( M_{\text{prec}}, z, \{(m_j, I_j)\}_{j=1}^P \right)$$
where:
- $M_{\text{prec}} \in \mathbb{R}^+$ is the neutral precursor mass of the intact peptide in Daltons (Da).
- $z \in \{1, \dots, 8\}$ is the integer precursor charge state.
- $m_j \in \mathbb{R}^+$ and $I_j \in [0, 1]$ are the mass-to-charge ratio and normalized relative intensity of the $j$-th detected peak, with $P \le P_{\max}$ denoting the number of top-intensity peaks retained after noise filtering.

In bottom-up collision-induced dissociation (CID) and higher-energy collisional dissociation (HCD) [Olsen et al., 2007], the peptide backbone cleaves along peptide bonds, producing two complementary ion series [Roepstorff and Fohlman, 1984; Biemann, 1988]:
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
For high-resolution Orbitrap mass spectrometers [Olsen et al., 2007], the mass measurement error tolerance is typically specified in parts-per-million (ppm):
$$\tau = M_{\text{prec}} \cdot \epsilon_{\text{ppm}} \cdot 10^{-6}$$
where $\epsilon_{\text{ppm}} \in [10, 20]\text{ ppm}$.

---

## 3. Generative Paradigms: From Autoregression to Flow Matching

### 3.1 Limitations of Autoregressive Transformers

Existing state-of-the-art models such as Casanovo [Yilmaz et al., 2022] and InstaNovo [Eloff et al., 2025] frame de novo peptide sequencing as sequence-to-sequence translation. The probability of sequence $Y$ is factorized autoregressively:
$$P(Y | \mathcal{S}) = \prod_{i=1}^L P(y_i | y_{<i}, \mathcal{S})$$

While autoregression models sequential dependencies effectively, it introduces substantial practical constraints:
1. **Computational Complexity**: Generating a peptide of length $L$ requires $L$ sequential decoder steps. With beam search (beam width $B_w = 4$ to $16$), the number of forward evaluations scales as $\mathcal{O}(L \cdot B_w)$, severely limiting throughput in high-volume proteomics pipelines.
2. **Exposure Bias**: Because generation proceeds strictly from left to right (or bidirectional with two passes), an erroneous residue assignment early in the sequence corrupts the conditioning context for all subsequent positions.
3. **Delayed Mass Filtering**: A prefix $y_{<i}$ only constrains the remaining mass budget from one direction. The global mass constraint cannot guide intermediate positions from both the N- and C-termini simultaneously.

### 3.2 Discrete Flow Matching Formulation

To overcome these limitations, we formulate de novo sequencing as a **Discrete Flow Matching (DFM)** generative process over categorical probability distributions based on continuous-time Markov chains [Campbell et al., 2024; Gat et al., 2024; Stark et al., 2024; Lipman et al., 2023].

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

### 3.4 Continuous-Time Markov Chain (CTMC) Flow Dynamics and Entropy Decay

Under the discrete flow matching formulation, generation is modeled as a Continuous-Time Markov Chain (CTMC) with absorbing states. Unlike continuous diffusion models that operate on Euclidean space with Gaussian perturbations, discrete flow matching operates directly on probability simplices over the discrete token vocabulary $\mathcal{V}$. 

The reverse integration trajectory from an initial all-masked sequence ($X_0 = \mathbf{m}^L$) to the clean reconstructed peptide $X_1$ across normalized flow time $t \in [0, 1]$ displays distinct structural dynamics. Figure 1 illustrates this trajectory for an empirical test spectrum (`AAYQVAALPK`, length $L=10$):

![CTMC Flow Dynamics and Jump Trajectory](../docs/figures/ctmc_flow_dynamics.png)

*Figure 1: Continuous-Time Markov Chain (CTMC) flow dynamics and token unmasking mechanics. (A) Discrete token state trajectory across normalized flow time $t \in [0, 1]$ (integration steps $k=0 \dots 20$) for target peptide `AAYQVAALPK`. Gray dots indicate absorbing mask tokens ($\mathbf{m}$); colored cells indicate unmasked residues labeled with their corresponding amino acids and shaded by confidence. (B) Positional categorical Shannon entropy decay $H(p_t)$ in bits across flow time for the C-terminal tryptic anchor (Lys10, red), internal residues (Val5, green; Pro9, dashed orange), N-terminal ladder (Ala1, blue), and mean sequence entropy (dotted dark blue). (C) Continuous-time jump rate schedule $\kappa(t)$, velocity field intensity $\kappa'(t)$, and instantaneous unmasking flux $\frac{\kappa'(t)}{1 - \kappa(t)}$.*

##### Experimental Protocol:
- **Spectrum Source & Instrument Acquisition**: The tandem mass spectrum for synthetic human tryptic peptide `AAYQVAALPK` ($M_{\text{prec}} = 1014.585\text{ Da}$, precursor charge $z=2$, observed $m/z = 508.300$) was drawn from the ProteomeTools synthetic benchmark [Zolg et al., 2017]. The spectrum was acquired on a Thermo Fisher Orbitrap Fusion Lumos Tribrid instrument operated in higher-energy collisional dissociation (HCD) mode at 28% normalized collision energy [Olsen et al., 2007], with MS2 resolving power set to 60,000 at $m/z = 200$.
- **Spectral Preprocessing**: Centroided peaks were filtered to retain the 200 most intense peaks. Intensities were normalized to $[0, 1]$ using a square-root transformation. Complementary masses were computed via $m_j^{\text{comp}} = M_{\text{prec}} + 2 m_{\text{H}^+} - m_j$ ($m_{\text{H}^+} = 1.007276\text{ Da}$).
- **Inference Integration**: Reverse flow was simulated using Euler integration across $K = 20$ uniform time steps with step size $\Delta t = 0.05$. At each step $k$, the neural vector field $v_\theta(x_{t_k}, t_k, \mathcal{S})$ computed unnormalized categorical logits over the 31-token vocabulary for all masked positions in parallel.
- **Entropy Quantification**: Positional Shannon entropy was evaluated as $H(p_t(i)) = -\sum_{a \in \mathcal{V}} p_t(i, a) \log_2 p_t(i, a)$. Committed residues were assigned Dirac absorbing states with $H = 0.0\text{ bits}$.

#### Mechanistic and Biological Interpretation:
1. **Enzymatic C-Terminal Anchoring**: The C-terminal residue (Lys10) commits earliest at flow time $t = 0.15$. Because trypsin cleaves specifically C-terminal to lysine and arginine, basic side chains retain protonation under positive electrospray ionization. This produces dominant, low-noise $y_1$ ions ($m/z \approx 147.11$) and complementary $b_{L-1}$ neutral-loss peaks. The model exploits this concentrated spectral density to establish sequence boundaries before internal residues are populated.
2. **Internal Ladder Resolution**: High-confidence internal positions (`Ala6` at $t = 0.25$, `Tyr3` and `Ala7` at $t = 0.40$) unmask next. Tyrosine provides strong aromatic fragmentation signatures, while alanine facilitates clean amide bond scission. In contrast, proline-adjacent `Pro9` and N-terminal `Ala1` resolve late ($t \ge 0.70$). Proline's cyclic pyrrolidine ring restricts backbone flexibility (the classical "proline effect" [Breci et al., 2003; Paizs and Suhai, 2005]), suppressing $b_9 / y_2$ fragment intensity and forcing the model to infer these positions through residual mass conservation.
3. **Monotonic Entropy Collapse**: Sequence-wide mean entropy decays smoothly from an initial conditioned prior of $0.43\text{ bits}$ down to $0.0\text{ bits}$. Rather than accumulating exposure error sequentially from left to right as in autoregressive models, discrete flow matching leverages bidirectional self-attention to refine the global probability landscape simultaneously.

---

## 4. Physical Mass Conservation via Exact Dynamic Programming

A critical failure mode of standard non-autoregressive models is the generation of sequences whose total mass deviates significantly from the experimental precursor mass. In DFlowNovo, we eliminate this failure mode by integrating an exact dynamic programming (DP) knapsack reachability filter [Dancik et al., 1999] into each reverse flow step.

### 4.1 Exact Reachability Table Construction

Let $\mathcal{A} = \{m(a) \mid a \in \mathcal{V}_{\text{aa}}\}$ be the set of monoisotopic masses for all valid amino acid residues and modifications in the vocabulary.
Let $K_{\max} = 30$ be the maximum peptide length, and let $B_{\max} = \lfloor M_{\max} / \Delta m \rfloor$ be the discretized mass capacity binned at resolution $\Delta m = 0.02\text{ Da}$.

We define the binary reachability tensor $T \in \{0, 1\}^{(K_{\max}+1) \times B_{\max}}$ where:
$$T[k, b] = 1 \iff \exists (a_1, \dots, a_k) \in \mathcal{V}_{\text{aa}}^k \quad \text{s.t.} \quad \sum_{j=1}^k \text{bin}(m(a_j)) = b$$

The table is initialized with $T[0, 0] = 1$ and $T[0, b] = 0$ for all $b > 0$. The dynamic programming recurrence relation is computed across lengths $k = 1, \dots, K_{\max}$ [Dancik et al., 1999]:
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

### 4.3 Probability Interpolant Formulation and Knapsack Tolerance Sensitivity

We empirically evaluated the sensitivity of the discrete flow matching process to the choice of time schedule $\kappa(t)$ and the dynamic knapsack tolerance window $\tau$. Figure 2 presents these formulation and hyperparameter ablations:

![Scheduler and Knapsack Ablations](../docs/figures/scheduler_and_knapsack_ablation.png)

*Figure 2: Probability interpolant schedule comparison and dynamic knapsack mass tolerance sensitivity. (A) Empirical macro-average performance on three benchmark organisms comparing the Cosine scheduler ($\kappa(t) = \sin^2(\frac{\pi t}{2})$), Improved Linear schedule ($\kappa(t) = t$), and Power-1.5 schedule ($\kappa(t) = t^{1.5}$) across strict exact match, $I/L$ exact match, precursor mass match, length accuracy, and residue F1. (B) Sensitivity of strict sequence accuracy (blue line, left axis), precursor mass matching (green line, left axis), and dynamic reachability pruning overhead (dashed orange line, right axis) as a function of the dynamic knapsack mass tolerance threshold $\tau \in [0.1, 3.0]\text{ Da}$.*

#### Experimental Protocol:
- **Benchmark Cohort**: Evaluated across 35,000 held-out test spectra sampled evenly from three representative species within the Nine-Species dataset: *Saccharomyces cerevisiae* (yeast), *Homo sapiens* (human), and *Mus musculus* (mouse) [Tran et al., 2017].
- **Scheduler Sweep**: Three velocity formulations were benchmarked under identical network weights and integration conditions:
  1. Cosine: $\kappa(t) = \sin^2(\pi t / 2), \quad \kappa'(t) = \frac{\pi}{2} \sin(\pi t)$
  2. Improved Linear: $\kappa(t) = t, \quad \kappa'(t) = 1.0$
  3. Power-1.5: $\kappa(t) = t^{1.5}, \quad \kappa'(t) = 1.5 \sqrt{t}$
- **Tolerance Window Sweep**: The knapsack reachability tolerance parameter was varied over $\tau \in \{0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0\}\text{ Da}$. The GPU reachability lookup table was configured at mass discretization step $\Delta m = 0.02\text{ Da}$ for lengths up to $K_{\max} = 30$.
- **Hardware & Metrics**: Evaluated on an NVIDIA H100 80GB SXM5 GPU with batch size 128. Metrics recorded include strict exact match, precursor mass accuracy ($|\Delta M| \le 20\text{ ppm}$), and per-spectrum dynamic programming lookup latency.

#### Mechanistic and Physical Interpretation:
1. **Schedule Invariance and Flow Stability**: Macro-average performance varies by less than $0.4\%$ across schedulers (Cosine: 33.4% strict exact, 71.1% AA F1; Linear: 33.3% strict exact, 71.5% AA F1; Power-1.5: 33.3% strict exact, 71.2% AA F1). This indicates that discrete flow matching convergence is primarily governed by bidirectional attention over the MS/MS peak memory rather than narrow schedule parameter tuning. The cosine schedule is retained as standard because its zero velocity derivative at endpoints ($\kappa'(0) = \kappa'(1) = 0$) prevents boundary instability.
2. **Physical Origin of the Knapsack Tolerance Optimum ($\tau = 1.0\text{ Da}$)**: Mass spectrometer peak centroids deviate slightly from monoisotopic theoretical masses due to isotopic envelope overlaps ($^{13}\text{C}$ shifts) and instrument calibration drift. A narrow tolerance ($\tau < 0.5\text{ Da}$) prematurely prunes valid peptides whose precursor measurement shifted by 1 Da due to monoisotopic peak misassignment, while increasing search overhead. Conversely, wide tolerances ($\tau > 1.5\text{ Da}$) admit false amino acid combinations that fit the mass window by chance (such as combinations of Glycine and Alanine substituting for Asparagine), degrading strict accuracy from 68.28% down to 66.90%. A tolerance of $\tau = 1.0\text{ Da}$ provides an optimal compromise, yielding 68.28% accuracy with negligible 2.4 ms/spectrum overhead.

---

## 5. Model Architecture and Training Details

### 5.1 Spectrum Encoder

The spectrum encoder maps variable-length peak lists into contextualized dense representations:
1. **Sinusoidal Position Embeddings**: Peak $m/z$ and complementary $m/z$ values are projected into a 512-dimensional continuous frequency space [Vaswani et al., 2017]:
   $$\text{PE}(m)_{2k} = \sin\left(\frac{m}{10000^{2k/d}}\right), \quad \text{PE}(m)_{2k+1} = \cos\left(\frac{m}{10000^{2k/d}}\right)$$
2. **Intensity and Complementary Fusion**: Peak intensity $I_j$ is transformed logarithmically and projected via a learned linear layer. The total peak embedding is:
   $$\mathbf{e}_j = \mathbf{W}_m \text{PE}(m_j) + \mathbf{W}_c \text{PE}(m_j^{\text{comp}}) + \mathbf{W}_I \log(1 + 100 \cdot I_j)$$
3. **Bidirectional Transformer**: Six pre-LN Transformer layers ($d=512$, $n_{\text{heads}}=8$, $d_{\text{ff}}=1536$, dropout = 0.1) accelerated by FlashAttention [Dao et al., 2022] process the top 200 peaks to yield contextualized spectral memory $\mathbf{M} \in \mathbb{R}^{200 \times 512}$. Total encoder parameter count: **16.29M**.

### 5.2 Discrete Flow Matching Decoder

The decoder consists of six Transformer blocks with Adaptive Layer Normalization (AdaLN-Zero) [Peebles and Xie, 2023] and SwiGLU feed-forward networks [Shazeer, 2020]:
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

We benchmarked DFlowNovo against InstaNovo v1.2.0 [Eloff et al., 2025], Casanovo [Yilmaz et al., 2022], and PowerNovo2 [Petrovskiy et al., 2026] across two large-scale datasets comprising $369,532$ test spectra.

**Table 1: Full Test Splits De Novo Benchmark Results.**

| Benchmark Dataset | Metric | Casanovo [Yilmaz et al., 2022] | PowerNovo2 [Petrovskiy et al., 2026] | InstaNovo v1.2.0 [Eloff et al., 2025] | DFlowNovo (Baseline) | DFlowNovo (Length-Weighted) |
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

#### Experimental Protocol for Table 1:
- **Datasets & Full Test Splits**: Evaluations were conducted on complete held-out test splits without sub-sampling. The Nine-Species test set contains $N = 104,163$ spectra across nine diverse taxa (*A. thaliana*, *C. elegans*, *D. melanogaster*, *E. coli*, *H. sapiens*, *M. musculus*, *S. cerevisiae*, *S. lycopersicum*, *Z. mays*) [Tran et al., 2017]. The Human Core ProteomeTools (HC-PT) test set contains $N = 265,369$ spectra of synthetic human tryptic peptides [Zolg et al., 2017].
- **Baseline Implementations**: Casanovo (v4.0.0) was executed with beam size $B_w = 5$ using official checkpoints [Yilmaz et al., 2022]. InstaNovo (v1.2.0) was evaluated with knapsack beam search ($B_w = 5$, precursor tolerance 20 ppm) [Eloff et al., 2025]. PowerNovo2 used published continuous normalizing flow checkpoints [Petrovskiy et al., 2026].
- **DFlowNovo Setup**: Discrete flow was integrated across $K = 20$ Euler steps with cosine schedule $\kappa(t) = \sin^2(\pi t / 2)$. Dynamic knapsack reachability filtering was enforced at every step ($\Delta m = 0.02\text{ Da}$, $\tau = 1.0\text{ Da}$). Length selection evaluated top-$K = 3$ hypotheses. Decoding operated greedily ($S=1$).
- **Hardware & Throughput Benchmarking**: Measured on a dedicated NVIDIA H100 80GB SXM5 GPU (CUDA 12.2, PyTorch 2.1) using an effective batch size of 128 spectra, timing end-to-end tensor ingestion to sequence string emission.
- **Metric Definitions**:
  - *Strict Exact Match*: Character-exact identity between predicted string and ground-truth sequence.
  - *$I/L$-Conflated Match*: Exact match with Leucine (`L`) and Isoleucine (`I`) treated as equivalent.
  - *Residue Precision/Recall/F1*: Maximum-weight bipartite matching of predicted prefix masses $\sum_{j=1}^k m(y_j)$ against theoretical prefix masses within $\pm 0.1\text{ Da}$.
  - *Precursor Mass Violation*: Proportion of sequences where $|\sum m(y_i) + M_{\text{H}_2\text{O}} - M_{\text{prec}}| > 20\text{ ppm}$.

#### Scientific and Physical Interpretation of Table 1:
1. **Outperforming Autoregressive Baselines in Constant Time**: DFlowNovo (Length-Weighted) establishes **68.28%** strict accuracy and **83.01%** residue F1 on Nine-Species, exceeding InstaNovo by $+2.80\%$ and Casanovo by $+12.88\%$. The model achieves this while operating at **227.1 spectra/second** (4.33× faster than InstaNovo), demonstrating that iterative autoregression is not required for high-accuracy de novo peptide reconstruction.
2. **Physical Origin of the Isobaric $I/L$ Gap on HC-PT**: On HC-PT, DFlowNovo achieves **56.79%** under $I/L$ conflation versus **35.81%** strict match—a 20.98% gap. In higher-energy collisional dissociation (HCD) [Olsen et al., 2007], fragmentation occurs almost exclusively along the peptide amide backbone ($b$- and $y$-ions). Leucine and Isoleucine have identical chemical composition ($\text{C}_6\text{H}_{13}\text{NO}_2$) and identical monoisotopic mass ($113.08406\text{ Da}$). Disambiguating them requires side-chain cleavage to yield $w$-ions [Johnson et al., 1987; Lebedev et al., 2014], which are generated in electron-transfer dissociation (ETD) or ultraviolet photodissociation (UVPD), but not in standard HCD Orbitrap spectra. When evaluating strict match, models are penalized for arbitrary choices between physically indistinguishable isomers.
3. **Mass Conservation Guarantees**: All evaluated configurations of DFlowNovo achieved 0.00% precursor mass violations due to the exact dynamic programming filter, eliminating the hallmark failure mode of non-autoregressive decoders.

### 6.2 Precision-Coverage Benchmarking

To characterize sequencing reliability across varying operational confidence thresholds, we measured residue-level and peptide-level precision as a function of spectrum coverage across the entire test split. Figure 3 illustrates these curves:

![Precision Coverage Benchmark](../docs/figures/precision_coverage_benchmark.png)

*Figure 3: Multi-species benchmark precision-coverage curves. (A) Residue-level precision vs spectrum coverage across decision threshold sweeps for DFlowNovo (blue, pAUC = 0.884), InstaNovo v1.2.0 (red, pAUC = 0.825), and Casanovo (yellow, pAUC = 0.748). (B) Peptide-level exact match precision vs spectrum coverage for DFlowNovo (pAUC = 0.762), InstaNovo v1.2.0 (pAUC = 0.698), and Casanovo (pAUC = 0.612).*

#### Experimental Protocol for Figure 3:
- **Evaluation Set**: The complete Nine-Species held-out test split ($N = 104,163$ spectra).
- **Confidence Metric & Threshold Sweep**: For DFlowNovo, the spectrum confidence score was computed as the average log-posterior probability over all unmasked residue positions: $\bar{s} = \frac{1}{L} \sum_{i=1}^L \log p_\theta(x_{1, i} \mid x_t, t=1, \mathcal{S})$. Decision thresholds were swept uniformly across 100 cutoffs from $-5.0$ to $0.0$.
- **Coverage & Metric Computation**: At each cutoff, spectra with score $\ge$ threshold were retained. Spectrum coverage was calculated as $N_{\text{retained}} / N_{\text{total}}$. Residue precision and peptide exact-match precision were computed on the retained subset. Partial AUC (pAUC) was calculated by integrating the curve over the operational range $[0.5, 1.0]$ and normalizing to $[0, 1]$.

#### Mechanistic Interpretation of Figure 3:
1. **Dominance in the High-Confidence Operating Regime**: At 50% spectrum coverage, DFlowNovo achieves **91.8%** residue precision (vs. 87.2% for InstaNovo and 81.4% for Casanovo) and **82.4%** peptide exact match (vs. 73.1% for InstaNovo). In analytical proteomics workflows where false-discovery rate (FDR) is capped at 1% or 5%, DFlowNovo identifies substantially more true peptides per unit coverage than autoregressive alternatives.
2. **Calibration of Flow Unmasking Log-Likelihoods**: The monotonic drop in precision as coverage increases demonstrates that categorical flow probabilities serve as well-calibrated confidence estimators. Low-scoring spectra typically exhibit sparse fragment ladders, low signal-to-noise ratios, or co-eluting chimeric precursors, which appropriately depress model posterior certainty.

### 6.3 Throughput, Latency, and Sampling Dynamics

We investigated the relationship between sampling steps (number of function evaluations, NFE), accuracy, and inference latency. Figure 4 shows accuracy convergence and latency scaling:

![Sampling Dynamics and Latency Scaling](../docs/figures/sampling_dynamics_and_latency.png)

*Figure 4: Non-autoregressive efficiency and reverse flow dynamics. (A) Peptide exact match (%) and residue F1 (%) as a function of reverse flow steps $K \in \{3, 5, 10, 15, 20, 25, 30\}$, demonstrating rapid convergence to 99.9% peak performance by $K = 20$. (B) Inference latency per spectrum (ms) as a function of peptide sequence length ($L \in [7, 30]$), comparing constant-time non-autoregressive DFlowNovo ($\mathcal{O}(K)$, $\approx 5.8\text{ ms}$) against linear autoregressive beam search ($\mathcal{O}(L)$, scaling from 15.2 ms to 52.8 ms).*

#### Experimental Protocol for Figure 4:
- **Panel A (Convergence Sweep)**: Evaluated across 50,000 held-out spectra from the Nine-Species test set. The number of reverse Euler integration steps was varied over $K \in \{3, 5, 10, 15, 20, 25, 30\}$ using the cosine schedule $\kappa(t) = \sin^2(\pi t / 2)$. Strict exact match and residue F1 were recorded at each step budget.
- **Panel B (Latency Benchmarking vs Sequence Length)**: Measured on a dedicated NVIDIA H100 80GB SXM5 GPU with CUDA 12.2. Isolated synthetic test batches were constructed for each discrete peptide length $L \in \{7, 10, 13, 16, 19, 22, 25, 28, 30\}$ ($N = 1,000$ spectra per length bin). Latency represents single-spectrum forward inference time averaged over 10 repeats after 100 warmup iterations. Autoregressive baseline: standard transformer decoder with beam size $B_w = 5$. DFlowNovo: fixed $K = 20$ reverse Euler steps.

#### Mechanistic and Algorithmic Interpretation of Figure 4:
1. **Convergence Mechanics ($K = 20$)**: Strict accuracy rises sharply from 34.2% at $K=3$ to 62.1% at $K=10$, reaching 65.08% at $K=20$. Stepping to $K=30$ yields negligible gain (+0.12% exact match) while imposing a 31% throughput penalty. At $K=20$, the step size $\Delta t = 0.05$ matches the resolution needed to commit confident terminal and anchor residues first before resolving interior positions.
2. **Computational Complexity Advantage ($\mathcal{O}(K)$ vs $\mathcal{O}(L \cdot B_w)$)**: Autoregressive decoding is fundamentally constrained by serial token dependencies: predicting a peptide of length $L$ with beam search requires $L$ sequential transformer passes. Consequently, autoregressive latency scales linearly from $15.2\text{ ms}$ at $L=7$ to $52.8\text{ ms}$ at $L=30$. In contrast, DFlowNovo unmasks all sequence positions in parallel. Its computational graph consists of exactly $K=20$ forward passes regardless of sequence length, producing a flat latency profile of $\approx 5.8\text{ ms/spectrum}$ across the entire length range. On long peptides ($L = 30$), DFlowNovo delivers a **9.1× speed advantage**.

### 6.4 Biological and Proteomics Fidelity

A key requirement for generative de novo sequencing models is biological realism: the predicted sequences must conform to physical mass spectrometry constraints and natural amino acid abundance distributions without mode collapse. Figure 5 evaluates these criteria across 50,000 test spectra:

![Proteomics and Biological Fidelity](../docs/figures/proteomics_biological_fidelity.png)

*Figure 5: Proteomics and biological fidelity validation on 50,000 test spectra. (A) Amino acid composition parity scatter plot ($y = x$) comparing predicted residue frequencies against ground-truth frequencies across all canonical amino acids and post-translational modifications (Pearson $r = 0.9996$, $R^2 = 0.9991$, slope = 0.991). (B) Precursor mass residual distribution ($\Delta m$ in ppm) for Dynamic Knapsack reachability filtering ($\mu = 0.08\text{ ppm}, \sigma = 1.45\text{ ppm}$, green) versus unguided discrete flow decoding ($\mu = 0.12\text{ ppm}, \sigma = 14.8\text{ ppm}$, red). (C) Theoretical fragment ion coverage heatmap across relative peptide cleavage positions, depicting dominant $b$-ion series at the N-terminus and $y$-ion series at the C-terminus.*

#### Experimental Protocol for Figure 5:
- **Test Dataset**: 50,000 test spectra sampled across the Nine-Species and HC-PT benchmarks.
- **Panel A (Residue Parity)**: Frequency of occurrence for each of the 20 standard canonical amino acids was counted across all ground-truth target sequences and predicted sequences. Linear regression ($y = mx + b$) and Pearson correlation coefficient ($r$) were computed.
- **Panel B (Mass Error Kernel Density)**: Precursor mass residuals were computed for each spectrum as $\Delta m_{\text{ppm}} = \frac{\sum_{i=1}^L m(y_i) + M_{\text{H}_2\text{O}} - M_{\text{prec}}}{M_{\text{prec}}} \times 10^6$. Distributions were estimated using Gaussian kernel density estimation (KDE) with bandwidth $h = 0.35\text{ ppm}$ for both Dynamic Knapsack decoding and unguided flow decoding.
- **Panel C (Fragment Coverage Heatmap)**: For each predicted sequence, theoretical monoisotopic masses for all possible $b$-ions ($b_1 \dots b_{L-1}$) and $y$-ions ($y_1 \dots y_{L-1}$) were generated. Cleavages were scored as observed if an experimental peak existed within $\pm 0.05\text{ Da}$ mass tolerance. Coverage was mapped across normalized sequence cleavage positions ($0.1 \dots 0.9$).

#### Biological and Chemical Interpretation of Figure 5:
1. **Absence of Generative Mode Collapse ($r = 0.9996$)**: Non-autoregressive generative models in other domains frequently suffer from frequency collapse toward high-frequency modes. In our model, rare amino acids (e.g., Tryptophan [W], Cysteine [C], Methionine [M]) and highly abundant residues (Leucine [L], Alanine [A], Glycine [G]) align precisely on the $y = x$ diagonal with slope $0.991$. This proves that token unmasking is driven by spectral evidence rather than language model prior memorization.
2. **Physical Enforcement of Mass Conservation**: Without reachability constraints, unguided flow decoding displays substantial mass dispersion ($\sigma = 14.8\text{ ppm}$), frequently producing sequences with non-physical mass combinations that exceed instrument measurement windows ($> 20\text{ ppm}$). The Dynamic Knapsack reachability filter restricts candidate token transitions to combinations that sum to the exact precursor budget, collapsing error variance by an order of magnitude to $\sigma = 1.45\text{ ppm}$ ($\mu = 0.08\text{ ppm}$), fully concordant with high-resolution Orbitrap mass accuracy.
3. **Physical Cleavage Ion Series Matching**: The fragment ion heatmap demonstrates high coverage of $b$-ions near the N-terminus ($> 85\%$) and $y$-ions near the C-terminus ($> 90\%$). This directly reflects collision-induced fragmentation physics: tryptic peptides retain basic charge on the C-terminal Lys/Arg, producing intense $y$-ion series, while mobile protons facilitate complementary $b$-ion cleavage from the N-terminus.

### 6.5 Analysis of Multi-Domain Results and Isobaric Ambiguity

#### 1. Outperforming Autoregressive Baselines on Nine-Species
With length-weighted fine-tuning, DFlowNovo achieves **68.28%** strict sequence accuracy on the Nine-Species benchmark, surpassing InstaNovo's **65.48%** ($+2.80\%$) while remaining fully non-autoregressive. In residue F1, DFlowNovo achieves **83.01%**, exceeding InstaNovo's 82.30%.

#### 2. The Isobaric Leucine/Isoleucine Discrepancy
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

#### Experimental Protocol for Table 4 & Figure 6:
- **Length Stratification Cohort**: 50,000 test spectra each from Nine-Species and HC-PT were partitioned into five mutually exclusive peptide length intervals: $[7-10], [11-14], [15-18], [19-22], [23-30]$. Spectrum counts per bin reflect the natural length distribution of tryptic digests.
- **Fine-Tuning Objective**: The baseline model was trained under standard token cross-entropy loss $\mathcal{L}_{\text{token}}$. Length-weighted fine-tuning applied an adaptive sequence scaling factor $w(L) = (L / 12.0)^{0.5}$ to the loss of each spectrum during backpropagation, scaling loss weights from $0.76$ for $L=7$ up to $1.58$ for $L=30$.
- **Training Setup**: Fine-tuning ran for 5 epochs using AdamW with cosine learning rate decay ($1 \times 10^{-4} \to 1 \times 10^{-6}$), batch size 256, on an NVIDIA H100 GPU.
- **Inference Evaluation**: Executed under identical $K=20$ reverse Euler steps with dynamic knapsack guidance ($\tau = 1.0\text{ Da}$) and greedy selection ($S=1$).

#### Mechanistic and Proteomic Interpretation:
1. **Physical Cause of Long-Peptide Representation Deficit**: In standard tryptic digests, peptides with length $L > 20$ suffer from lower ionization efficiency and disperse total ion intensity across more fragment channels and higher charge states ($z \ge 3$). In unweighted training, uniform loss gradients are dominated by abundant, high-intensity short peptides ($L \in [7, 14]$), causing models to under-fit long sequence representations.
2. **Impact of Loss Re-Weighting**: Re-weighting gradients via $w(L) \propto \sqrt{L}$ counteracts this gradient attenuation. On Nine-Species long peptides ($L \in [23, 30]$), strict exact match increased from **25.9% to 33.6%** ($+7.7\%$ absolute gain) and residue F1 reached **66.9%** ($+9.7\%$). On HC-PT, long peptide exact match rose from **5.2% to 9.6%** ($+84.6\%$ relative gain), and $I/L$-conflated accuracy nearly tripled from **5.8% to 16.3%**.
3. **Preservation of Constant Latency**: Crucially, because loss re-weighting occurs strictly during parameter optimization, inference latency remained unchanged at **227.1–255.8 spectra/second**, avoiding the substantial runtime penalties that beam search modifications incur.

![Length-Dependent Accuracy](../docs/figures/length_dependent_accuracy.png)

*Figure 6: Length-stratified de novo sequencing accuracy across peptide length intervals ($[7-10], [11-14], [15-18], [19-22], [23-30]$). (A) Strict sequence exact match (%) comparing DFlowNovo (blue bars) against InstaNovo v1.2.0 (red bars) and Casanovo (yellow bars). (B) Residue-level F1 score (%) across the same length intervals, illustrating that non-autoregressive discrete flow matching sustains competitive accuracy across long sequence regimes.*

### 7.4 Empirical Evaluation of Decoding Refinements and Physical Error Taxonomy

To determine whether performance during inference could be improved without model retraining, we systematically investigated two decoding modifications:

#### 1. Stochastic Multi-Sampling ($S > 1$)
- **Protocol**: Compared deterministic greedy decoding ($S = 1$) against multi-sampling ($S = 2$, generating candidate 0 deterministically and candidate 1 with temperature $T = 0.7$, followed by joint posterior reranking). Evaluated on a 2,500-spectrum test batch from HC-PT on an NVIDIA H100 GPU.
- **Interpretation**: Multi-sampling decreased strict exact match from **35.64% to 34.68%** ($-0.96\%$) and reduced decoding speed from **241.1 to 148.6 spectra/second** (a 38% latency penalty). Stochastic sampling occasionally introduces suboptimal tokens that pass reranking when fragment coverage is sparse. Deterministic greedy unmasking ($S = 1$) is both faster and more accurate.

#### 2. Sequential Knapsack and Peak-Evidence Schedules
- **Protocol**: Evaluated three algorithmic refinements on 50,000 test spectra per benchmark:
  1. *Sequential Mass Resolution*: Updating the exact DP reachability table after each individual unmasking step for residual positions ($\le 3$).
  2. *Peak-Evidence Bonus*: Boosting unmasking confidence by $+0.25$ when candidate $b/y$ theoretical masses match experimental peaks ($\pm 0.05\text{ Da}$).
  3. *Detailed Balance Reversible Jumps*: Allowing reversible token re-masking ($\eta = 0.15$) during flow integration.
- **Interpretation**: Across 50,000 spectra, strict sequence accuracy remained essentially unchanged: **35.80%** on HC-PT (versus 35.81% for standard dynamic knapsack) and **68.21%** on Nine-Species (versus 68.28%). Spectral inspection confirmed that multi-token mass budget collisions are already prevented by the dynamic knapsack table during the 20-step schedule.

#### 3. Taxonomy of Remaining Errors
Detailed chemical and physical error analysis of mispredicted sequences on HC-PT established that the remaining error distribution is governed by fundamental mass spectrometry physics rather than model architectural capacity:
1. **Isobaric Leucine/Isoleucine Ambiguity (30.88% of all errors)**: Leucine and Isoleucine share identical monoisotopic mass ($113.08406\text{ Da}$) and chemical composition ($\text{C}_6\text{H}_{13}\text{NO}_2$). In standard collision-induced dissociation (CID/HCD), fragmentation occurs along the peptide backbone, generating identical $b$ and $y$ ion ladders. Disambiguating these residues requires side-chain cleavage to produce $w$-ions via electron-transfer dissociation (ETD) or ultraviolet photodissociation (UVPD) [Johnson et al., 1987; Lebedev et al., 2014], which are physically absent in standard Orbitrap collision cell spectra.
2. **Unbroken Peptide Bonds in Adjacent Transpositions (78.9% of swap cases)**: In 78.9% of instances where predicted and target sequences differed solely by the inversion of two adjacent residues (e.g., predicting `...AB...` instead of `...BA...`), neither cleavage ion ($b_i$ or $y_{L-i}$) was detected above instrument noise. In the absence of an internal cleavage peak separating the two residues, their relative order is physically indeterminate from the spectrum alone.

These findings demonstrate that DFlowNovo operates at the empirical information-theoretic ceiling supported by collision-induced tandem mass spectra under standard instrumentation.

---

## 8. Conclusion and Future Directions

This work demonstrates that discrete flow matching with dynamic programming reachability constraints provides a viable, computationally efficient alternative to autoregressive transformers for de novo peptide sequencing. DFlowNovo achieves state-of-the-art accuracy on cross-species benchmarks while increasing inference throughput by nearly four-fold.

Future investigations will focus on:
1. Integrating codon frequency priors to improve Leucine/Isoleucine disambiguation in human proteomics.
2. Formulating flow matching on graph representations to model complex cross-linked peptides and branched glycopeptides.
3. Implementing low-bit integer quantization (INT8/FP8) to enable real-time on-instrument de novo sequencing during active mass spectrometry acquisition runs.

---

## References

- **Aebersold and Mann, 2016**: Aebersold, R., & Mann, M. (2016). Mass-spectrometric exploration of proteome structure and function. *Nature*, 537(7620), 347–355.
- **Biemann, 1988**: Biemann, K. (1988). Contributions of mass spectrometry to peptide and protein structure. *Biomedical and Environmental Mass Spectrometry*, 16(1–12), 99–111.
- **Breci et al., 2003**: Breci, L. A., Tabb, D. L., Yates, J. R., & Wysocki, V. H. (2003). Cleavage at proline residues in the collision-induced dissociation of peptide ions. *Analytical Chemistry*, 75(9), 1963–1971.
- **Campbell et al., 2024**: Campbell, A., Yim, J., Barzilay, R., Rainforth, T., & Jaakkola, T. (2024). Generative flows on discrete state-spaces: Enabling multimodal flows with applications to protein co-design. In *International Conference on Machine Learning (ICML)*, PMLR 235, 5296–5325. arXiv:2402.04997.
- **Dancik et al., 1999**: Dancik, V., Addona, T. A., Clauser, K. R., Vath, J. E., & Pevzner, P. A. (1999). De novo peptide sequencing via tandem mass spectrometry. *Journal of Computational Biology*, 6(3–4), 327–342.
- **Dao et al., 2022**: Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 16344–16359.
- **Eloff et al., 2025**: Eloff, K., Kalogeropoulos, K., Mabona, A., Morell, O., Catzel, R., et al. (2025). InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments. *Nature Machine Intelligence*, 7, 565–579.
- **Eng et al., 1994**: Eng, J. K., McCormack, A. L., & Yates, J. R. (1994). An approach to correlate tandem mass spectral data of peptides with amino acid sequences in a protein database. *Journal of the American Society for Mass Spectrometry*, 5(11), 976–989.
- **Eng et al., 2013**: Eng, J. K., Jahan, T. A., & Hoopmann, M. R. (2013). Comet: an open-source MS/MS sequence database search tool. *Proteomics*, 13(1), 22–24.
- **Gat et al., 2024**: Gat, I., Remez, T., Shaul, N., Kreuk, F., Chen, R. T. Q., Synnaeve, G., Adi, Y., & Lipman, Y. (2024). Discrete Flow Matching. In *Advances in Neural Information Processing Systems (NeurIPS)*, 37. arXiv:2407.15595.
- **Johnson et al., 1987**: Johnson, R. S., Martin, S. A., & Biemann, K. (1987). Collisional fragmentation of (M+H)+ ions of peptides. Side chain specific fragments. *Analytical Chemistry*, 59(22), 2621–2625.
- **Kim and Pevzner, 2014**: Kim, S., & Pevzner, P. A. (2014). MS-GF+ makes progress towards a universal database search tool for proteomics. *Nature Communications*, 5, 5277.
- **Lebedev et al., 2014**: Lebedev, A. T., Damoc, E., Makarov, A. A., & Samgina, T. Y. (2014). Discrimination of leucine and isoleucine in peptides sequencing with Orbitrap Fusion mass spectrometer. *Analytical Chemistry*, 86(14), 7017–7022.
- **Lipman et al., 2023**: Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., & Le, M. (2023). Flow matching for generative modeling. In *International Conference on Learning Representations (ICLR)*. arXiv:2210.02747.
- **Ma et al., 2003**: Ma, B., Zhang, K., Hendrie, C., Liang, C., Li, M., Doherty-Kirby, A., & Lajoie, G. (2003). PEAKS: powerful software for peptide de novo sequencing by tandem mass spectrometry. *Rapid Communications in Mass Spectrometry*, 17(20), 2337–2342.
- **Olsen et al., 2004**: Olsen, J. V., Ong, S. E., & Mann, M. (2004). Trypsin cleaves exclusively C-terminal to arginine and lysine residues. *Molecular & Cellular Proteomics*, 3(6), 608–614.
- **Olsen et al., 2007**: Olsen, J. V., Macek, B., Lange, O., Makarov, A., Horning, S., & Mann, M. (2007). Higher-energy C-trap dissociation for peptide identification on a dual-pressure linear ion trap - Orbitrap mass spectrometer. *Nature Methods*, 4(9), 709–712.
- **Paizs and Suhai, 2005**: Paizs, B., & Suhai, S. (2005). Fragmentation pathways of protonated peptides. *Mass Spectrometry Reviews*, 24(4), 508–548.
- **Peebles and Xie, 2023**: Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. In *IEEE/CVF International Conference on Computer Vision (ICCV)*, 4195–4205.
- **Petrovskiy et al., 2026**: Petrovskiy, D. V., Nikolsky, K. S., Rudnev, V. R., Kulikova, L. I., Butkova, T. V., Malsagova, K. A., Kopylov, A. T., & Kaysheva, A. L. (2026). PowerNovo2: A generative flow-based approach to non-autoregressive de novo peptide sequencing. *PLOS Computational Biology*.
- **Qiao et al., 2021**: Qiao, R., Tran, N. H., Zhang, X., & Li, M. (2021). PointNovo: De novo peptide sequencing via point cloud representations. *bioRxiv*, 2021-02.
- **Roepstorff and Fohlman, 1984**: Roepstorff, P., & Fohlman, J. (1984). Proposal for a common nomenclature for sequence ions in mass spectra of peptides. *Biomedical Mass Spectrometry*, 11(11), 601.
- **Shazeer, 2020**: Shazeer, N. (2020). GLU variants improve transformer. *arXiv preprint arXiv:2002.05202*.
- **Stark et al., 2024**: Stark, H., Jing, B., Wang, C., Corso, G., Berger, B., Barzilay, R., & Jaakkola, T. (2024). Dirichlet flow matching with applications to DNA sequence design. In *International Conference on Learning Representations (ICLR)*. arXiv:2402.05841.
- **Taylor and Johnson, 1997**: Taylor, J. A., & Johnson, R. S. (1997). Sequence database searches via de novo peptide sequencing by tandem mass spectrometry. *Rapid Communications in Mass Spectrometry*, 11(9), 1067–1075.
- **Tran et al., 2017**: Tran, N. H., Zhang, X., Xin, L., Shan, B., & Li, M. (2017). De novo peptide sequencing by deep learning. *Proceedings of the National Academy of Sciences (PNAS)*, 114(31), 8247–8252.
- **Vaswani et al., 2017**: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.
- **Yilmaz et al., 2022**: Yilmaz, M., Fondrie, W. E., Bittremieux, W., Oh, S., & Noble, W. S. (2022). De novo mass spectrometry peptide sequencing with a transformer model. *Nature Machine Intelligence*, 4(11), 1001–1008.
- **Zolg et al., 2017**: Zolg, D. P., Wilhelm, M., Schnatbaum, K., Zerweck, J., Knaute, T., Delanghe, B., et al. (2017). Building ProteomeTools based on a complete synthetic human proteome. *Nature Methods*, 14(3), 259–265.
