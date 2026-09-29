# DFlowNovo: Master's Defense Oral Presentation Script & Committee Q&A Guide

**Candidate:** Joël Gédéon  
**Degree:** Master of Science in Artificial Intelligence, African Institute for Mathematical Sciences (AIMS South Africa)  
**Supervisors:** *********  
**Date of Defense:** September 30, 2026  
**Total Defense Duration:** 30 Minutes (20–22 Minutes Oral Presentation + 8–10 Minutes Committee Q&A)

---

## 1. Defense Timing and Pacing Strategy

| Segment | Slide Range | Content Focus | Target Timestamp | Allotted Time |
| :--- | :--- | :--- | :--- | :--- |
| **I. Biological & Physical Motivation** | Slides 1–4 | Proteomics context, MS/MS inverse problem, failure of prior art | 00:00 – 04:00 | 4.0 min |
| **II. Core Theory & KnapsackDP** | Slides 5–8 | CTMC Discrete Flow Matching & exact GPU Knapsack reachability | 04:00 – 08:30 | 4.5 min |
| **III. Architecture & Training** | Slides 9–10 | Neural model vs sampling pipeline, joint balanced curriculum | 08:30 – 11:30 | 3.0 min |
| **IV. Primary Benchmarks & Biophysics** | Slides 11–16 | Nine-Species SOTA, I/L biophysical proof, 256 spec/s latency, ablations | 11:30 – 17:00 | 5.5 min |
| **V. Generative Dynamics & Production** | Slides 17–19 | Entropy collapse trajectories, qualitative regimes, Hugging Face / Gradio | 17:00 – 19:30 | 2.5 min |
| **VI. Limitations, Roadmap & Close** | Slides 20–22 | Methodological boundaries, five core contributions, conclusion | 19:30 – 21:30 | 2.0 min |
| **VII. Committee Examination (Q&A)** | Backup A1–A4 | Deep-dive defenses, mathematical proofs, biophysical validations | 21:30 – 30:00 | 8.5 min |

---

## 2. Slide-by-Slide Verbatim Speaking Script & Visual Cues

---

### Slide 1: Title & Candidate Information
* **Target Time:** `00:00 – 00:45` (Duration: 45s)
* **Visual Direction:** Maintain open posture; glance briefly at the title slide, then make direct eye contact with the examination committee.
* **Spoken Script:**
  > "Good morning, esteemed committee members, supervisors, and colleagues. I am Joël Gédéon, and today I have the privilege of presenting my Master's thesis entitled: *'DFlowNovo: Continuous-Time Discrete Flow Matching and Dynamic Knapsack Guidance for High-Throughput De Novo Peptide Sequencing'*.
  >
  > This research was conducted at the African Institute for Mathematical Sciences in South Africa, under the supervision of *********.
  >
  > Today, I will demonstrate how framing de novo peptide sequencing as a continuous-time Markov jump process directly on the discrete categorical simplex—coupled with an exact dynamic programming mass reachability algorithm—overcomes fundamental bottlenecks in proteomics, establishing new state-of-the-art accuracy while delivering an order-of-magnitude inference speedup."
* **Transition:** *"To understand why this is necessary, let us examine the fundamental biological problem."*

---

### Slide 2: The Biological Problem: Bottom-Up Proteomics & Database Search Limitations
* **Target Time:** `00:45 – 01:45` (Duration: 60s)
* **Visual Direction:** Point to the left column contrasting genomics and proteomics, then gesture to the alert block on database limitations.
* **Spoken Script:**
  > "While the genome represents the static genetic blueprint of an organism, proteins constitute the functional machinery of life. In bottom-up proteomics, complex protein mixtures are digested by enzymes such as trypsin into short peptides, which are ionized, fragmented, and analyzed via tandem mass spectrometry.
  >
  > For over two decades, the standard paradigm has been database search tools like MaxQuant and Comet. These tools work by comparing experimental spectra against theoretical spectra generated from pre-existing genomic databases.
  >
  > However, this paradigm suffers from a critical vulnerability: genomic blindness. In non-model organisms, environmental metagenomics, hypermutated cancer neoantigens, and antibody repertoires—where sequence space exceeds $10^{14}$ unique variants—no reference database exists.
  >
  > *De novo* peptide sequencing solves this by reconstructing the amino acid sequence directly from physical fragmentation spectra without any genomic reference. The central challenge has always been achieving the accuracy and throughput necessary for clinical-scale discovery."
* **Transition:** *"However, reconstructing peptides directly from spectra is a notoriously ill-posed physical inverse problem."*

---

### Slide 3: The Physical Challenge: Tandem Mass Spectrometry Inverse Problem
* **Target Time:** `01:45 – 02:45` (Duration: 60s)
* **Visual Direction:** Direct attention to the mathematical duality equation $m(b_k) + m(y_{L-k}) = M_{\text{prec}} + 2m(\text{H}^+)$ and the four listed physical bottlenecks.
* **Spoken Script:**
  > "When an isolated peptide precursor undergoes Higher-energy Collisional Dissociation, or HCD, the peptide backbone cleaves along amide bonds. This produces two complementary ion series: $b$-ions extending from the N-terminus, and $y$-ions extending from the C-terminus.
  >
  > Notice the fundamental physical symmetry: for any cleavage site $k$, the mass of the prefix $b$-ion plus the mass of the suffix $y$-ion must sum exactly to the precursor mass plus protons.
  >
  > Why is this inverse mapping so challenging? Four physical reasons:
  > First, missing cleavages: peptides often fail to cleave uniformly, leaving unobserved mass gaps exceeding 200 Daltons.
  > Second, peak directionality ambiguity: a raw mass-to-charge peak carries no label stating whether it is a $b$-ion, a $y$-ion, or an internal fragment.
  > Third, chemical noise from neutral losses such as water and ammonia.
  > And fourth, combinatorial explosion: for an average 15-mer peptide, the unconstrained search space is $20^{15}$, or over $3 \times 10^{19}$ potential sequences."
* **Transition:** *"Let us examine how contemporary deep learning methods have attempted to address this challenge."*

---

### Slide 4: Why Prior Deep Learning Approaches Fall Short
* **Target Time:** `02:45 – 04:00` (Duration: 75s)
* **Visual Direction:** Contrast Paradigm 1 (Autoregressive) on the left with Paradigm 2 (Continuous Flows) on the right. Emphasize the bolded failure metrics.
* **Spoken Script:**
  > "Prior machine learning approaches have bifurcated into two distinct paradigms, both of which suffer from severe structural limitations.
  >
  > In Paradigm 1, autoregressive transformers such as Casanovo and InstaNovo decode peptides token-by-token from left to right. This imposes a linear $\mathcal{O}(L)$ latency bottleneck, requiring 30 sequential forward passes per peptide and limiting throughput to 28 to 52 spectra per second. Worse, autoregressive decoding suffers from cascading error propagation: an error at position 2 or 3 permanently misaligns all subsequent ion ladders. Furthermore, InstaNovo v1.2 exhibits severe species token collapse, dropping from 63% on human to only 15.45% on divergent species.
  >
  > In Paradigm 2, continuous flow models like PowerNovo2 attempt non-autoregressive generation by flowing continuous latent vectors in $\mathbb{R}^D$ and applying post-hoc Integer Linear Programming. As shown by our benchmarks, this leads to a catastrophic discretization failure: PowerNovo2 achieves only 3.16% strict match on the Nine-Species benchmark, because continuous Euclidean trajectories do not respect the discrete topology of peptide space.
  >
  > This led us to our fundamental research question: Can we design a non-autoregressive generative model directly on the discrete categorical simplex while strictly enforcing physical precursor mass conservation?"
* **Transition:** *"This brings us to the theoretical foundation of DFlowNovo: Continuous-Time Markov Chain Discrete Flow Matching."*

---

### Slide 5: Key Concept: CTMC Discrete Flow Matching
* **Target Time:** `04:00 – 05:15` (Duration: 75s)
* **Visual Direction:** Point to the probability path formulation $p_{t|1}(x_t^d = j | x_1^d)$ and the transition rate matrix $R_t(x, y)$.
* **Spoken Script:**
  > "Rather than mapping discrete tokens to continuous Gaussians, DFlowNovo operates directly on the sequence space $\mathcal{V}^L$ via a Continuous-Time Markov Chain jump process, building upon recent breakthroughs by Campbell et al. and Lipman et al.
  >
  > At time $t=0$, the state $x_0$ is a sequence composed entirely of absorbing mask tokens $[M]$. Over continuous time $t \in [0, 1]$, probability mass flows along a predefined path towards the target peptide $x_1$ sampled from the true conditional data distribution.
  >
  > The evolution of this jump process is governed by a transition rate matrix $R_t(x, y)$. Each sequence position jumps independently from the masked state to the ground-truth amino acid according to a monotonic schedule $\kappa(t)$.
  >
  > Because the state space remains strictly categorical at every intermediate time $t$, there is zero continuous rounding error. In addition, all $L$ sequence positions evolve concurrently, enabling true non-autoregressive parallel generation."
* **Transition:** *"Let us examine the directional vector field and an important mathematical invariance that governs this flow."*

---

### Slide 6: Continuous-Time Probability Paths & Directional Flow Vector Field
* **Target Time:** `05:15 – 06:15` (Duration: 60s)
* **Visual Direction:** Point to the Kolmogorov Master Equation on the left and the Detailed Balance Invariance Theorem on the right.
* **Spoken Script:**
  > "The time evolution of the marginal probability distribution $p_t(x)$ satisfies the Kolmogorov Forward Equation, also known as the Chemical Master Equation in physical chemistry.
  >
  > The canonical conditional rate matrix $R_t^*(x, j \mid x_1)$ is obtained via the positive part of the time derivative of the probability path divided by the current marginal. Taking the conditional expectation under our neural network's posterior yields the marginal rate field $R_t(x, y)$.
  >
  > Now consider the right column: we prove a Detailed Balance Invariance Theorem. If we add any rate matrix $R_t^{\text{DB}}$ that satisfies detailed balance with respect to $p_t$, the resulting flow generates the exact same marginal probability distribution.
  >
  > This invariance guarantees that we can introduce stochastic exploration during sampling—controlled by parameter $\eta$—to escape shallow local modes without introducing any bias into the target peptide distribution."
* **Transition:** *"However, non-autoregressive independence introduces a critical physical dilemma."*

---

### Slide 7: The Precursor Mass Dilemma
* **Target Time:** `06:15 – 07:15` (Duration: 60s)
* **Visual Direction:** Emphasize the conflict between independent factorized probabilities and the global summation constraint $\sum m(a_i) = M_{\text{target}}$.
* **Spoken Script:**
  > "In non-autoregressive generation, token positions unmask conditionally independent given the spectrum and intermediate state $x_t$.
  >
  > Yet, mass spectrometry imposes an unyielding physical constraint: the sum of the mono-isotopic masses of all amino acids must equal the measured precursor neutral mass within instrument tolerance $\tau$.
  >
  > If positions unmask independently without global coordination, the probability that 15 independently sampled amino acids exactly sum to $M_{\text{target}}$ within 1 Dalton decays exponentially to less than one-tenth of one percent!
  >
  > As shown on the slide, unconstrained flow causes precursor mass error variance to explode to $\sigma = 14.8\text{ ppm}$, hallucinating chemically impossible combinations and resulting in more than 35% of generated sequences being rejected post-hoc.
  >
  > The core algorithmic hurdle is therefore: how do we constrain parallel flow generation to strictly valid knapsack solutions without sacrificing GPU throughput?"
* **Transition:** *"Our solution is KnapsackDP: an exact dynamic programming reachability algorithm optimized for modern GPUs."*

---

### Slide 8: The Core Innovation: Exact KnapsackDP Guidance
* **Target Time:** `07:15 – 08:30` (Duration: 75s)
* **Visual Direction:** Walk through the recurrence equation, the 1D GPU pooling kernel, and the dynamic logit pruning mechanism.
* **Spoken Script:**
  > "We formulate mass reachability as a bounded dynamic programming knapsack problem. We discretize the mass axis with a resolution of $\Delta m = 0.02\text{ Da}$, yielding 250,000 bins up to 5,000 Daltons.
  >
  > We precompute a boolean reachability tensor $T[k, b]$, where entry $(k, b)$ is 1 if and only if exactly $k$ amino acids can sum to mass bin $b$. This recurrence is computed in just 28 milliseconds once at initialization and requires only 6.65 Megabytes of memory, fitting entirely within the GPU's L2 cache.
  >
  > To account for instrument calibration tolerance $\tau = 1.0\text{ Da}$, we apply a 1D GPU max-pooling kernel with a window of 101 bins across the mass dimension.
  >
  > At inference time, during each reverse Euler CTMC step, when $k$ positions remain masked and the remaining mass deficit is $R'$, our CUDA kernel evaluates the reachability of every candidate amino acid $a$ in $\mathcal{O}(1)$ time. If candidate $a$ cannot lead to a valid precursor sum within $k-1$ remaining steps, its logit is set to negative infinity.
  >
  > The empirical effect is dramatic: precursor mass error variance contracts tenfold—from $\sigma = 14.8\text{ ppm}$ down to $1.45\text{ ppm}$—with absolutely zero latency overhead."
* **Transition:** *"Now let us look at how the neural network and the sampling pipeline integrate into a cohesive system."*

---

### Slide 9: System Architecture: Clean Separation of Model and Decoding Pipeline
* **Target Time:** `08:30 – 10:00` (Duration: 90s)
* **Visual Direction:** Direct the committee's eyes to Part A (left half of the central figure) and Part B (right half).
* **Spoken Script:**
  > "Slide 9 depicts the complete architectural layout of DFlowNovo, highlighting our explicit architectural separation between the neural model in Part A and the inference sampling engine in Part B.
  >
  > In Part A, the neural model totals 59.48 million parameters.
  > It features a 16.3-million-parameter Bidirectional Spectrum Encoder that processes sinusoidal mass-to-charge embeddings and intensity projections using FlashAttention-2.
  > The global $[CLS]$ representation conditions an auxiliary 2-layer MLP Length Predictor, which achieves 68.4% top-1 accuracy and 94.2% top-3 accuracy.
  > The core generative module is our 42.8-million-parameter Flow Decoder, composed of 6 bidirectional Transformer blocks with adaptive layer normalization (AdaLN-Zero) and SwiGLU activations, which predicts categorical transition logits across all positions concurrently.
  >
  > In Part B, during inference, we extract the top-$B$ length candidates—typically $B=3$—and initialize parallel masked chains. We execute a 20-step reverse Euler CTMC loop with our dynamic GPU KnapsackDP logit pruning. Finally, candidates are evaluated by a Bayesian re-ranking function that incorporates theoretical fragment ladder matches."
* **Transition:** *"Next, let us address a critical training challenge: how to train across divergent species without catastrophic forgetting."*

---

### Slide 10: Multi-Domain Training: Joint Balanced Curriculum
* **Target Time:** `10:00 – 11:30` (Duration: 90s)
* **Visual Direction:** Point to the comparison chart on the left, contrasting synthetic-only, sequential, and joint balanced training.
* **Spoken Script:**
  > "Training deep models on mass spectrometry data involves navigating severe domain shifts between synthetic peptide libraries and natural biological digests. In our study, we sourced both benchmark datasets directly from the standardized Hugging Face repositories published by InstaDeep: `ms_proteometools` (comprising 265,369 synthetic test spectra) and `ms_ninespecies_benchmark` (comprising 104,163 biological test spectra).
  >
  > If we train exclusively on synthetic data, HC-PT accuracy reaches 36.4%, but performance on Nine-Species collapses to 12.01%.
  > Conversely, if we attempt sequential fine-tuning—first pre-training on synthetic data and then fine-tuning on Nine-Species—the model suffers severe catastrophic forgetting: Nine-Species reaches 65.6%, but HC-PT drops to 12.85%.
  >
  > To resolve this, we designed a Joint Balanced Curriculum. We interleave biological and synthetic mini-batches in an exact 50/50 ratio within each training iteration.
  > We maintain Exponential Moving Average (EMA) shadow weights with decay $\beta = 0.999$ to stabilize flow trajectories, combined with a cosine learning rate schedule warmed up over 2,000 steps.
  >
  > As demonstrated by the plot, this preserves high accuracy on both domains simultaneously—64.92% on Nine-Species and 34.93% on HC-PT—while training at an exceptional 2.0 minutes per epoch on an NVIDIA H100 GPU."
* **Transition:** *"Let us examine the primary benchmark results against contemporary state-of-the-art models."*

---

### Slide 11: Primary Benchmark Results: SOTA on Nine-Species Test Set
* **Target Time:** `11:30 – 13:00` (Duration: 90s)
* **Visual Direction:** Highlight the top row of the table (DFlowNovo 68.28%) and the McNemar statistical test result.
* **Spoken Script:**
  > "Slide 11 presents our primary benchmark evaluation on the full Nine-Species test set from InstaDeep's `ms_ninespecies_benchmark` repository, encompassing 104,163 tandem mass spectra across nine phylogenetically divergent organisms, ranging from yeast to human.
  >
  > DFlowNovo establishes a new state of the art, achieving **68.28% strict exact match** and **83.01% amino acid F1 score**, while processing **227.1 spectra per second**.
  >
  > Notice the critical comparisons:
  > First, DFlowNovo achieves a +15.08% absolute strict match improvement over the baseline InstaNovo v1.0, which achieved 53.20%.
  > Second, we completely resolve the cross-species collapse observed in InstaNovo v1.2, whose strict match fell to 15.45% on non-human species due to sub-optimal tokenization shifts.
  > Third, continuous flow matching via PowerNovo2 completely fails, managing only 3.16% strict match.
  >
  > To verify statistical significance, we performed McNemar's paired test across all 104,163 test spectra. The test yielded a chi-squared value of 7,542.8 ($p < 10^{-15}$), confirming that DFlowNovo's performance advantage is statistically significant."
* **Transition:** *"Now let us investigate an apparent performance gap on human synthetic peptides, which reveals a clear biophysical insight."*

---

### Slide 12: Cross-Domain Generalization & The Isobaric Leucine/Isoleucine Phenomenon
* **Target Time:** `13:00 – 14:15` (Duration: 75s)
* **Visual Direction:** Point to the red-vs-blue bar comparison on the left and the biophysical isomer structure on the right.
* **Spoken Script:**
  > "When evaluating DFlowNovo on the human synthetic HC-PT benchmark, an interesting phenomenon emerges: our strict exact match is 35.81%, but our Leucine/Isoleucine-equivalent match jumps to 56.79%—a gap of 20.98%!
  >
  > Rather than an algorithmic failure, this gap is governed by fundamental biophysical chemistry.
  > Leucine and Isoleucine are constitutional isomers sharing the identical chemical formula $\text{C}_6\text{H}_{13}\text{NO}_2$ and an identical monoisotopic mass of 113.084064 Daltons.
  >
  > In HCD mass spectrometry, collision energy cleaves backbone peptide bonds. Because the aliphatic side chains remain intact, the resulting $b$- and $y$-ions are mathematically identical in mass.
  >
  > Our detailed error analysis proves that **30.88% of all discordant predictions on HC-PT are pure Leucine/Isoleucine swaps**, accounting for 68.4% of all single-residue errors.
  >
  > To physically resolve this ambiguity requires radical Electron-Activated Dissociation (EAD or ETD), which cleaves side chains to produce $w$-ions with a 14.015 Dalton diagnostic shift. In standard HCD, evaluating without I/L equivalence artificially penalizes models for guessing an unobservable physical state."
* **Transition:** *"Next, let us examine the inference speed and throughput dynamics of our non-autoregressive architecture."*

---

### Slide 13: High-Throughput Inference Dynamics: 256 spectra/sec
* **Target Time:** `14:15 – 15:15` (Duration: 60s)
* **Visual Direction:** Point to the flat green line in plot B on the left, contrasting with the steeply rising dashed red line of autoregressive models.
* **Spoken Script:**
  > "In high-throughput proteomics, instruments acquire hundreds of spectra per second, making inference latency a primary operational consideration.
  >
  > In plot B on the left, we contrast the inference latency scaling against peptide length. Autoregressive models scale linearly $\mathcal{O}(L)$, taking up to 42.5 milliseconds per spectrum for longer peptides.
  >
  > In stark contrast, DFlowNovo exhibits flat $\mathcal{O}(K)$ scaling, maintaining an average latency of approximately 5.8 milliseconds regardless of peptide length, because all positions unmask in parallel over 20 integration steps.
  >
  > As detailed in the table, DFlowNovo achieves an inference throughput of 227 to 256 spectra per second on an H100 GPU. This is **4.3 to 4.9 times faster than InstaNovo**, and **8 to 9 times faster than Casanovo**.
  >
  > In clinical terms: sequencing 1,000,000 patient spectra requires just **1.09 hours** with DFlowNovo, compared to nearly **10 hours** with Casanovo."
* **Transition:** *"Speed must not come at the expense of confidence calibration. Let us examine precision-coverage."*

---

### Slide 14: Precision-Coverage & Calibrated Confidence Scoring
* **Target Time:** `15:15 – 16:00` (Duration: 45s)
* **Visual Direction:** Direct attention to the precision-coverage curves and the +31,412 accepted PSMs metric.
* **Spoken Script:**
  > "In experimental proteomics, researchers rarely analyze raw unthresholded predictions. Instead, candidate peptide-spectrum matches are filtered at strict confidence thresholds, such as 80% precision or a 1% False Discovery Rate.
  >
  > At an 80% precision operating point on the Nine-Species benchmark, DFlowNovo yields **86,412 high-confidence accepted PSMs**, corresponding to 80.15% coverage.
  >
  > Compared to InstaNovo v1.0, which yielded 55,000 accepted PSMs, DFlowNovo provides **31,412 additional verified peptides—a 57.1% increase in biological discovery yield**.
  >
  > Furthermore, our Bayesian scoring function achieves an Average Precision of 88.4%, demonstrating exceptional probability calibration for downstream biomarker and neoantigen screening."
* **Transition:** *"To validate each component of our methodology, we conducted detailed ablation studies."*

---

### Slide 15: Comprehensive Ablation Studies: Knapsack, Steps, and Schedulers
* **Target Time:** `16:00 – 17:00` (Duration: 60s)
* **Visual Direction:** Walk through the three subplots on the left corresponding to tolerance $\tau$, steps $K$, and schedule $\kappa(t)$.
* **Spoken Script:**
  > "Slide 15 isolates the empirical impact of each hyperparameter.
  >
  > First, the knapsack tolerance window $\tau$: Setting $\tau$ too tight, at 0.1 Daltons, degrades accuracy to 58.2% because it rejects valid peptides subject to minor instrument calibration drift. Setting $\tau$ too loose, at 2.0 Daltons, permits excessive isobaric branching. The sweet spot is $\tau = 1.0\text{ Da}$, which achieves peak accuracy.
  >
  > Second, the integration step budget $K$: A 5-step Euler integration yields 51.2% accuracy. Performance converges at $K=20$ steps to 65.1%. Increasing to 50 steps yields a negligible 0.3% gain while more than doubling compute, confirming that $K=20$ is Pareto-optimal.
  >
  > Third, the time schedule $\kappa(t)$: Our Cosine schedule consistently outperforms a Linear schedule because it delays unmasking flux during early steps, allowing the bidirectional encoder representations to mature before committing to residue assignments."
* **Transition:** *"We also addressed a well-known vulnerability in deep learning for proteomics: degradation on long peptides."*

---

### Slide 16: Resolving the Long Peptide Bottleneck: Length-Weighted Loss
* **Target Time:** `17:00 – 17:45` (Duration: 45s)
* **Visual Direction:** Highlight the red curve in the left plot maintaining elevated accuracy past length 23.
* **Spoken Script:**
  > "Tryptic digest distributions are naturally skewed, with peptides of length 23 or greater accounting for less than 5% of all training examples.
  >
  > Under standard cross-entropy loss, neural networks suffer severe performance collapse on long peptides, dropping to less than 1% exact match for $L \ge 23$.
  >
  > To solve this, we introduced a square-root length-weighted loss, $w(L) = \sqrt{L / \bar{L}}$, which appropriately penalizes errors on long peptides without destabilizing short peptide gradients.
  >
  > As shown by the red curve, this recovers exact match accuracy on long peptides to 33.6%, while short peptides ($L \le 10$) maintain an exceptional 88.19% exact match and 94.34% amino acid F1."
* **Transition:** *"Let us now inspect the internal mechanics of the generative process as time evolves."*

---

### Slide 17: Empirical Flow Dynamics: Entropy Collapse and Jump Flux Trajectories
* **Target Time:** `17:45 – 18:30` (Duration: 45s)
* **Visual Direction:** Trace the trajectory phases from top to bottom on the left plot, pointing out Phase I, II, and III.
* **Spoken Script:**
  > "Slide 17 visualizes the empirical generative flow trajectory across normalized time $t \in [0, 1]$. We observe three distinct physical phases:
  >
  > In Phase I, from $t=0$ to $0.25$, entropy is high across all positions. The model establishes global precursor mass and length constraints. Prominent terminal anchor peaks unmask first—specifically the C-terminal tryptic basic residues, Lysine and Arginine.
  >
  > In Phase II, from $t=0.25$ to $0.70$, an unmasking avalanche occurs. The jump rate $\lambda_t$ reaches its peak, and intense $b$- and $y$-ion pairs unmask prefix and suffix residues concurrently in parallel.
  >
  > In Phase III, from $t=0.70$ to $1.0$, our dynamic GPU KnapsackDP locks in. With only two or three positions remaining, unviable mass trajectories are pruned, the residual deficit forces a unique valid completion, and categorical entropy collapses to zero."
* **Transition:** *"Let us examine qualitative case studies illustrating how this behaves in concrete physical regimes."*

---

### Slide 18: Qualitative Case Studies: Four Canonical Physical Regimes
* **Target Time:** `18:30 – 19:15` (Duration: 45s)
* **Visual Direction:** Walk through the four color-coded quadrants on the right, connecting each to its underlying mass spectrometry mechanic.
* **Spoken Script:**
  > "Slide 18 presents four canonical prediction regimes from the Nine-Species benchmark:
  >
  > In Case 1, a 14-mer yeast peptide `IVSWYDNEYGYSTR`: DFlowNovo achieves a 100% exact consensus match with 13 of 13 fragment cleavages verified and 0.00 Dalton mass error, whereas InstaNovo collapsed to an incorrect Leucine.
  >
  > In Case 2, an isobaric ambiguity on a 12-mer: DFlowNovo predicts `DSMKEL-SESPER` where ground truth is `DSMKEI-SESPER`. This is an unavoidable I/L swap with identical HCD spectra.
  >
  > In Case 3, the well-known Proline effect: Tertiary amine nitrogen in Proline suppresses fragmentation of the preceding amide bond, creating a 2-residue mass gap. DFlowNovo correctly localizes the flanking sequence while inverting the adjacent dipeptide `[IE] \to [EI]`.
  >
  > In Case 4, near-isobaric dipeptide substitution: DFlowNovo substitutes `VQ` for `NI`. The mass difference between Valine-Glutamine and Asparagine-Isoleucine is just 4.8 milli-Daltons, or 4.0 ppm—well below the resolution limit of standard collision cells."
* **Transition:** *"To ensure this research has immediate community impact, we engineered a production-grade deployment."*

---

### Slide 19: Production Deployment: Hugging Face Model Hub & Live Interactive Studio
* **Target Time:** `19:15 – 20:00` (Duration: 45s)
* **Visual Direction:** Point to the one-line Python loading code on the left and the Gradio Studio interface description on the right.
* **Spoken Script:**
  > "Reproducibility and clinical translation were primary design goals.
  >
  > We stripped optimizer states and quantized weights into a lean 454-Megabyte BF16 checkpoint, available directly on the Hugging Face Model Hub. A user can load and run DFlowNovo with just three lines of Python code.
  >
  > It features native I/O support for MGF, mzML, and Thermo raw files, and our complete codebase is open-sourced under the Apache 2.0 license.
  >
  > Furthermore, we developed a full Gradio Web Studio. Lab technicians can upload raw mzML files, interactively zoom into fragment spectra, observe real-time animated flow unmasking, and inspect color-coded $b$- and $y$-ion coverage alongside precursor PPM error gauges."
* **Transition:** *"In the spirit of scientific rigor, let us openly discuss the methodological limitations."*

---

### Slide 20: Limitations & Scientific Boundary Conditions
* **Target Time:** `20:00 – 20:45` (Duration: 45s)
* **Visual Direction:** Acknowledge all four boxes with transparency and scientific maturity.
* **Spoken Script:**
  > "Every scientific methodology possesses operational boundary conditions, and we explicitly delineate four:
  >
  > First, complex post-translational modifications: DFlowNovo natively handles Methionine oxidation and Cysteine carbamidomethylation. However, open searches with variable phosphorylation or glycosylation require multi-dimensional knapsack state tensors, which we formulate in Backup Slide A3.
  >
  > Second, low-resolution ion-trap instruments: Our model is calibrated for high-resolution Orbitrap analyzers with accuracy under 20 ppm; low-resolution instruments requiring $\pm 0.5\text{ Da}$ tolerances increase isobaric candidate branching.
  >
  > Third, non-tryptic digests: Tryptic peptides benefit from strong C-terminal Lysine/Arginine priors; HLA immunopeptidomes lacking this bias require wider beam search budgets.
  >
  > And fourth, severe fragmentation gaps: In cases where internal fragmentation is physically absent, no machine learning model can hallucinate missing physical information."
* **Transition:** *"This brings us to our summary of contributions and future outlook."*

---

### Slide 21: Summary of Contributions & Future Roadmap
* **Target Time:** `20:45 – 21:30` (Duration: 45s)
* **Visual Direction:** Summarize the five bolded contributions on the left, then outline the three future directions on the right.
* **Spoken Script:**
  > "To summarize our primary contributions:
  > First, we developed the first Continuous-Time Markov Chain Discrete Flow Matching framework for peptide sequencing, eliminating continuous rounding loss.
  > Second, we introduced exact GPU KnapsackDP reachability guidance, contracting precursor mass variance to 1.45 ppm.
  > Third, we established a new state of the art on Nine-Species with 68.28% strict match and 86,412 accepted PSMs.
  > Fourth, we achieved flat $\mathcal{O}(K)$ latency, running at 227 to 256 spectra per second—over 4 times faster than InstaNovo.
  > And fifth, we provided rigorous biophysical grounding for the 20.98% Leucine/Isoleucine HCD ambiguity.
  >
  > Looking forward, our roadmap includes multi-modal flow matching combining paired HCD and EAD spectra, incorporating orthogonal retention time and ion mobility dimensions, and assembling full-length monoclonal antibodies directly from de novo contigs."
* **Transition:** *"Allow me to conclude by expressing my gratitude."*

---

### Slide 22: Acknowledgments & Conclusion
* **Target Time:** `21:30 – 22:00` (Duration: 30s)
* **Visual Direction:** Acknowledge supervisors and institutions warmly; transition smoothly into the Q&A phase.
* **Spoken Script:**
  > "I would like to express my deepest gratitude to my supervisors, *********, for their exceptional guidance, intellectual encouragement, and high scientific standards throughout this journey.
  >
  > I also extend sincere thanks to the African Institute for Mathematical Sciences and InstaDeep for providing the computational resources, fellowship, and vibrant environment that made this work possible.
  >
  > In conclusion, DFlowNovo proves that discrete flow matching with dynamic mass guidance bridges generative machine learning and physical mass spectrometry, unlocking accurate, high-throughput de novo sequencing for next-generation proteomics.
  >
  > Thank you for your time and attention. I now welcome your questions and comments."

---

## 3. Committee Defense Q&A: Top 10 Hardest Questions & Grounded Defenses

---

### Question 1: Continuous vs. Discrete Flow Formulations
* **Examiner Archetype:** The Theoretical Machine Learning Specialist
* **Question:** *"Why did you formulate the generative model as a Continuous-Time Markov Chain over discrete tokens rather than using standard continuous flow matching in $\mathbb{R}^D$ followed by rounding, as done in PowerNovo2 or audio flow matching?"*
* **Core Trap:** The examiner wants to see if you understand why continuous diffusion/flows work in vision/audio but fail in molecular sequence space.
* **Authoritative Defense:**
  > "Thank you for that fundamental question. The core reason lies in the geometric topology of peptide sequence space.
  >
  > In continuous flow matching, the latent trajectory $z_t$ evolves in Euclidean space $\mathbb{R}^{L \times D}$. While this works well for smooth, continuous signals like pixel intensities or acoustic waveforms, peptide sequences are inherently discrete elements of a finite categorical set $\mathcal{V}^L$.
  >
  > When a continuous flow generates a vector in $\mathbb{R}^D$, one must perform a post-hoc projection or Integer Linear Program to snap continuous coordinates to discrete amino acids. As our empirical evaluation of PowerNovo2 demonstrated on Slide 11, this continuous-to-discrete rounding destroys the learned generative trajectory, collapsing strict match accuracy to just 3.16%.
  >
  > In contrast, our CTMC formulation operates directly on probability distributions over the categorical simplex $\Delta^{|\mathcal{V}|-1}$ at each sequence position. Transitions are modeled as continuous-time jump processes governed by the Kolmogorov Forward Equation. There is no intermediate continuous embedding to discretize, zero quantization error, and categorical probabilities directly interface with our discrete dynamic programming knapsack filter."
* **Supporting Slide:** Slide 5 & Backup Slide A1.

---

### Question 2: KnapsackDP Computational Overhead & Scaling
* **Examiner Archetype:** The Algorithms & Systems Reviewer
* **Question:** *"Dynamic programming for knapsack problems is pseudo-polynomial in time. How can you claim that KnapsackDP runs in $\mathcal{O}(1)$ time without introducing a severe GPU latency bottleneck during real-time decoding?"*
* **Core Trap:** Checking whether you confuse offline precomputation with online inference complexity.
* **Authoritative Defense:**
  > "That is an important distinction to clarify. The pseudo-polynomial complexity $\mathcal{O}(K_{\max} \cdot |\mathcal{V}| \cdot B_{\max})$ applies **exclusively to the offline precomputation phase**, which is executed exactly once upon server startup.
  >
  > Specifically, with $K_{\max} = 30$, $|\mathcal{V}| = 22$, and mass discretization $\Delta m = 0.02\text{ Da}$ up to 5,000 Daltons ($B_{\max} = 250,000$), the precomputation requires only $1.65 \times 10^8$ boolean operations. As detailed in Backup Slide A2, this executes in just 28 milliseconds on a standard CPU.
  >
  > The resulting reachability table $T_{\text{pooled}}$ occupies only 7.39 Megabytes of memory, which resides entirely within the GPU's L2 cache.
  >
  > During online inference at step $t$, for a peptide with $k$ masked positions and remaining mass deficit $R'$, determining the reachability of candidate amino acid $a$ is a single array index lookup: evaluating $T_{\text{pooled}}[k-1, \lfloor (R'-m(a))/\Delta m \rceil]$. This is a true $\mathcal{O}(1)$ memory lookup executed concurrently across all GPU threads. That is precisely why our throughput reaches 256 spectra per second with zero latency penalty."
* **Supporting Slide:** Slide 8 & Backup Slide A2.

---

### Question 3: The Leucine/Isoleucine Discrepancy on HC-PT
* **Examiner Archetype:** The Rigorous Mass Spectrometrist
* **Question:** *"On the HC-PT benchmark, your strict exact match is only 35.81%, while your I/L-equivalent match is 56.79%—a gap of over 20%. Doesn't this large discrepancy indicate that your model is failing to learn residue context accurately on human data?"*
* **Core Trap:** The examiner is testing whether you will blame your model or whether you understand the fundamental biophysical limits of HCD fragmentation.
* **Authoritative Defense:**
  > "I am glad you raised this, because this 20.98% gap actually provides the strongest confirmation that our model is learning the true physical limits of the data rather than overfitting to synthetic artifacts.
  >
  > Leucine and Isoleucine are structural isomers. They have the exact same chemical formula, $\text{C}_6\text{H}_{13}\text{NO}_2$, and the exact same mono-isotopic mass of 113.084064 Daltons. In collision-induced dissociation (HCD), energy is deposited into the peptide backbone, cleaving amide bonds to generate $b$- and $y$-ions. The aliphatic hydrocarbon side chains do not fragment.
  >
  > Consequently, replacing an Isoleucine with a Leucine at any position produces a set of $b$- and $y$-ions whose masses are identical down to the sub-atomic level. A mass spectrometer measuring HCD spectra physically cannot distinguish them.
  >
  > Our detailed error audit on Slide 12 revealed that 30.88% of all errors on HC-PT were pure Leucine/Isoleucine swaps, accounting for 68.4% of all single-residue errors. To distinguish Leucine from Isoleucine requires radical dissociation techniques such as Electron-Activated Dissociation (EAD), which cleave side chains to yield $w$-ions with a diagnostic 14 Dalton shift. Penalizing an HCD model for I/L swaps is penalizing it for an unobservable physical state."
* **Supporting Slide:** Slide 12.

---

### Question 4: Comparison to InstaNovo v1.2 Collapse
* **Examiner Archetype:** The Computational Proteomics Benchmarker
* **Question:** *"InstaNovo v1.2 is widely cited in literature. Why does its strict match drop so drastically to 15.45% on Nine-Species, whereas your model achieves 68.28%? Is this a fair comparison, or was InstaNovo misconfigured?"*
* **Core Trap:** Checking whether you ran baselines properly or manipulated hyperparameter settings.
* **Authoritative Defense:**
  > "We performed our evaluations using the official, author-released checkpoint and default inference pipelines provided by the InstaNovo authors.
  >
  > The collapse of InstaNovo v1.2 to 15.45% on Nine-Species is an established empirical phenomenon caused by cross-species domain shift and vocabulary tokenization changes introduced between version 1.0 and version 1.2.
  >
  > In InstaNovo v1.0, the model achieved 53.20% strict match on Nine-Species. In version 1.2, the authors retrained the model predominantly on human synthetic libraries (HC-PT). This induced severe species token collapse when evaluated on divergent non-human organisms (such as Bacillus, Yeast, and Tomato) due to differing amino acid background frequencies and digest characteristics.
  >
  > In our thesis, we report both: InstaNovo v1.0 at 53.20% and InstaNovo v1.2 at 15.45%. DFlowNovo outperforms both, achieving 68.28% strict match—a +15.08% improvement over InstaNovo's best historical checkpoint—while completely resisting cross-species collapse through our Joint Balanced Curriculum."
* **Supporting Slide:** Slide 10 & Slide 11.

---

### Question 5: Non-Autoregressive Error Coupling vs. Autoregressive Conditioning
* **Examiner Archetype:** The NLP & Sequence Modeling Expert
* **Question:** *"In autoregressive models, predicting position $i$ conditions on all previous tokens $a_{<i}$. In your non-autoregressive flow, positions jump conditionally independent given $x_t$. Doesn't this independent factorization prevent the model from capturing strong local biochemical motifs, like proline cleavage patterns?"*
* **Core Trap:** Pointing out that factorized distributions cannot model dependencies without iterative refinement.
* **Authoritative Defense:**
  > "In a single-step non-autoregressive model, that critique would be entirely valid. However, DFlowNovo is an iterative continuous-time flow model evaluated across $K=20$ integration steps.
  >
  > At step $t=0$, positions are conditionally independent given the spectrum. But as soon as prominent anchor residues unmask at early times $t \in [0.1, 0.3]$—such as the C-terminal tryptic Lysine or Arginine—those unmasked tokens are fed directly back into our 6-layer bidirectional Transformer decoder at subsequent time steps.
  >
  > Through full bidirectional self-attention across all $L$ positions, every remaining masked token is explicitly conditioned on all previously unmasked tokens. As shown in our empirical flow analysis on Slide 17, this creates a cooperative unmasking avalanche where prefix and suffix context actively guides internal motif resolution.
  >
  > Furthermore, our KnapsackDP introduces a global joint coupling across all positions by eliminating token combinations that violate precursor mass conservation. Thus, DFlowNovo achieves the dependency modeling of autoregressive transformers with the speed of parallel decoding."
* **Supporting Slide:** Slide 9, Slide 13 & Slide 17.

---

### Question 6: Handling Complex Post-Translational Modifications (PTMs)
* **Examiner Archetype:** The Biological Proteomics Specialist
* **Question:** *"Your experimental benchmark includes Methionine oxidation and Cysteine carbamidomethylation. How does DFlowNovo scale when moving to open modification searches with phosphorylation, acetylation, or ubiquitination?"*
* **Core Trap:** Exposing that combinatorial PTM explosions might break the knapsack table or vocabulary.
* **Authoritative Defense:**
  > "That is a critical scalability consideration. In our current implementation, fixed carbamidomethylation (+57.021 Da) is handled by statically shifting Cysteine's residue mass, while variable Methionine oxidation (+15.995 Da) is accommodated by adding an explicit token $\text{M}_{\text{ox}}$ to vocabulary $\mathcal{V}$, increasing $|\mathcal{V}|$ from 21 to 22.
  >
  > To expand to variable phosphorylation ($\text{S, T, Y}$) or acetylation ($\text{K}$), we have formulated a Multi-Dimensional KnapsackDP in Backup Slide A3.
  >
  > Instead of a 2D boolean table, we construct a 3D reachability tensor $T[k, c, b]$, where $c \in \{0, 1, 2, 3\}$ indexes the count of variable modifications. Because biological peptides rarely carry more than 2 or 3 variable modifications simultaneously, capping $c \le 3$ restricts the state space.
  >
  > For a 3-modification search, the reachability table expands from 7.4 MB to 29.6 MB, and precomputation takes 120 milliseconds. This easily fits inside GPU VRAM, allowing DFlowNovo to perform guided open PTM searches with minimal computational overhead."
* **Supporting Slide:** Slide 20 & Backup Slide A3.

---

### Question 7: Precursor Mass Tolerance Sensitivity & Instrument Calibration
* **Examiner Archetype:** The Analytical Chemist
* **Question:** *"Your ablation study shows that setting tolerance $\tau = 0.1\text{ Da}$ dropped accuracy from 65.08% to 58.2%. Why would a tighter, more precise mass tolerance hurt performance on high-resolution Orbitrap instruments that have sub-ppm mass accuracy?"*
* **Core Trap:** Confusing MS1 precursor measurement accuracy with theoretical binning discretization and isotope peak selection.
* **Authoritative Defense:**
  > "While modern Orbitrap analyzers achieve sub-ppm mass precision in MS1 scans, two physical factors make an overly rigid $\tau = 0.1\text{ Da}$ filter detrimental in de novo sequencing:
  >
  > First, precursor isolation windows in quadrupole mass filters frequently select the second or third isotopic peak ($^{13}\text{C}$ or $^{15}\text{N}$) rather than the monoisotopic precursor, a phenomenon known as an isotopic offset error of $\approx 1.0033\text{ Da}$. If the search algorithm rigidly enforces $\tau = 0.1\text{ Da}$ around the recorded precursor mass without accounting for isotope envelopes, it erroneously prunes the true ground-truth peptide.
  >
  > Second, experimental charge-state misassignment can induce small apparent mass offsets.
  >
  > By setting $\tau = 1.0\text{ Da}$ during KnapsackDP logit pruning, our filter acts as a robust physical guardrail that accommodates isotope shifts and instrument calibration drift, while our Bayesian candidate re-ranking step subsequently applies a fine-grained quadratic PPM mass penalty ($\alpha = 0.05$) to reward high-precision matches."
* **Supporting Slide:** Slide 8, Slide 15 & Backup Slide A4.

---

### Question 8: Peptide Length Prediction Reliability
* **Examiner Archetype:** The Machine Learning Systems Reviewer
* **Question:** *"Your architecture relies on an auxiliary length predictor. If the length predictor guesses the wrong length, does the entire generative flow fail?"*
* **Core Trap:** Pointing out a potential single point of failure in non-autoregressive models.
* **Authoritative Defense:**
  > "We specifically anticipated this potential vulnerability, which is why DFlowNovo implements a parallel Length Beam Search rather than committing to a single top-1 prediction.
  >
  > As shown on Slide 9, our 2-layer length predictor achieves 68.4% top-1 accuracy, but reaches **94.2% top-3 accuracy**.
  >
  > At inference time, we extract the top-$B$ length candidates—where $B=3$—and initialize $B$ independent masked sequences in parallel on the GPU. Because DFlowNovo executes in parallel across the batch dimension, evaluating $B=3$ lengths concurrently incurs virtually no latency penalty on an NVIDIA H100.
  >
  > After completing the 20-step reverse flow for all $B$ chains, our Bayesian joint scoring function evaluates candidates across lengths, incorporating the length prior $\log p(L \mid \mathcal{S})$, token confidences, precursor mass agreement, and theoretical fragment ladder coverage. Consequently, an incorrect top-1 length prediction does not cause a failure because the true length is retained and selected in the beam."
* **Supporting Slide:** Slide 9 & Backup Slide A4.

---

### Question 9: Detailed Balance and Stochastic Sampling
* **Examiner Archetype:** The Stochastic Processes & Physics Specialist
* **Question:** *"In Slide 6, you present a Detailed Balance Invariance Theorem for the rate matrix. In practice, did you set $\eta > 0$ for stochastic sampling, or did you use deterministic argmax integration? What is the empirical trade-off?"*
* **Core Trap:** Testing whether theoretical concepts presented on slides were actually tested and understood in code.
* **Authoritative Defense:**
  > "In our benchmark experiments, we evaluated both deterministic flow ($\eta = 0$) and stochastic jump sampling ($\eta \in [0.1, 0.5]$).
  >
  > Deterministic Euler integration ($\eta = 0$) evaluates the mode of the conditional vector field at each step. This maximizes top-1 exact match on clean spectra where prominent fragment peaks clearly dictate the residue sequence.
  >
  > However, when spectra exhibit low signal-to-noise ratios or missing cleavages exceeding 200 Daltons, deterministic flow can become trapped in greedy local minima. Setting $\eta = 0.2$ injects stochastic jumps that satisfy detailed balance, allowing the sampler to explore alternative isobaric sequences without altering the asymptotic marginal distribution.
  >
  > When generating an ensemble of $N$ diverse candidate peptides for database-independent metaproteomics, $\eta > 0$ yields a 4.2% higher top-5 candidate recall. For our primary reported benchmarks on Slide 11, we report deterministic mode ($\eta = 0$) for fair comparison against the deterministic greedy/beam search baselines."
* **Supporting Slide:** Slide 6 & Backup Slide A1.

---

### Question 10: Clinical Translation & Production Viability
* **Examiner Archetype:** The Translational & Clinical Proteomics Director
* **Question:** *"How does DFlowNovo fit into actual clinical workflows, such as identifying personalized cancer neoantigens for mRNA vaccines? What are the practical barriers to adoption?"*
* **Core Trap:** Assessing whether the candidate understands real-world clinical implementation versus academic benchmarks.
* **Authoritative Defense:**
  > "In personalized cancer neoantigen discovery, time and precision are both matter-of-life factors. Clinical protocols sequence tumor biopsies via LC-MS/MS to identify human leukocyte antigen (HLA) presentation peptides that harbor patient-specific somatic mutations.
  >
  > Standard database searches are blind to these novel mutations unless an entire exome sequencing database is constructed and searched per patient, which introduces massive search spaces and multiple-testing FDR penalties.
  >
  > DFlowNovo provides three direct clinical advantages:
  > First, it requires zero reference database, discovering mutated neoantigen sequences directly from fragment spectra.
  > Second, as shown on Slide 14, at an 80% precision threshold, DFlowNovo yields **86,412 verified PSMs**—a 57.1% increase in peptide yield over InstaNovo, ensuring that low-abundance tumor neoantigens are not missed.
  > Third, its high throughput of 256 spectra per second allows a patient's entire MS/MS run of 1,000,000 spectra to be processed in **1.09 hours** rather than 10 hours, fitting comfortably within the turnaround windows required for mRNA vaccine manufacturing.
  >
  > Our open-source Gradio studio and Hugging Face checkpoint provide a turn-key pipeline directly usable by clinical technicians."
* **Supporting Slide:** Slide 2, Slide 13, Slide 14 & Slide 19.

---

## 4. Emergency Fallback & Recovery Strategies During Defense

1. **If an examiner points out an unexpected number or discrepancy:**
   * *Acknowledge and frame:* "Thank you for that sharp observation. That metric reflects the full test split evaluation ($N=104,163$). Let us examine Slide 11 or Backup Slide A4 where the exact breakdown and statistical confidence bounds are detailed."
2. **If an examiner asks a question outside the scope of your thesis (e.g., cross-linking MS or top-down proteomics):**
   * *Acknowledge boundary and pivot:* "That is an excellent question that touches on the frontiers of structural mass spectrometry. While DFlowNovo was specifically formulated and calibrated for bottom-up tryptic LC-MS/MS, its underlying CTMC flow matching formulation is fundamentally agnostic to fragmentation chemistry. Extending this to top-down intact protein spectra would require expanding the knapsack state tensor to larger mass windows and multi-charge deconvolution, which represents an exciting avenue for our future research roadmap."
3. **If you lose your train of thought:**
   * *Pause, breathe, and anchor to the slide figure:* "Allow me to ground this point directly in the experimental data shown here on the left..."
