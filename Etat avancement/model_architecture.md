# Model Architecture: DFM (Discrete Flow Matching) for De Novo Peptide Sequencing

## 1. Overview
The proposed model, **DFLowNovo**, aims to solve the de novo peptide sequencing problem by predicting an amino acid sequence $y = (y_1, y_2, \dots, y_L)$ from a tandem mass spectrum $\mathcal{S}$. We frame this as a conditional generative modeling task using Discrete Flow Matching (DFM). The model comprises three main modules:
1. **Spectrum Encoder:** Extracts a continuous representation of the mass spectrum $\mathcal{S}$.
2. **Peptide Length Predictor:** Estimates the length $L$ of the peptide from the spectrum embedding.
3. **Peptide Decoder:** An iterative discrete flow matching decoder that generates the peptide sequence conditioned on the spectrum, precursor mass, charge, predicted length, and diffusion time $t$.

## 2. Spectrum Encoder and Feature Engineering
The mass spectrum $\mathcal{S}$ is represented as a set of $k$ peaks, where each peak $i$ has a mass-to-charge ratio $(m/z)_i$ and an intensity $I_i$. 

### The $m/z$-Complementary Peak Idea
Inspired by the **$\pi$-Helix paper** [1], we augment the physical representation of each peak by computing its theoretical complementary fragment. In tandem mass spectrometry (e.g., HCD fragmentation), peptide bonds break to form $b$-ions and $y$-ions. If a peak is a $b$-ion, its corresponding $y$-ion is usually present in the spectrum. The relationship is governed by the precursor mass $M_{prec}$:
$$ (m/z)_{comp, i} = M_{prec} - (m/z)_i $$
By explicitly providing $(m/z)_{comp, i}$ to the encoder, we force the attention mechanism to recognize structural complementary pairs (b/y pairs) without having to implicitly learn the subtraction arithmetic. 

### Encoder Architecture
For each of the top $k=200$ peaks, we project the tuple $((m/z)_i, (m/z)_{comp, i}, I_i)$ into a high-dimensional continuous space using Sinusoidal embeddings (for $m/z$ values) and MLPs. These features are fused into a single token per peak $h_i \in \mathbb{R}^d$. 
The encoder applies $N_e = 6$ layers of standard Multi-Head Self-Attention (MHSA) over these peak tokens to model intra-spectrum peak-to-peak interactions (e.g., isotopic envelopes and mass differences corresponding to amino acids).

## 3. Peptide Length Predictor
Unlike autoregressive models (like InstaNovo) that naturally terminate by predicting an End-Of-Sequence (EOS) token, non-autoregressive flow matching models generate all tokens simultaneously. This requires knowing the sequence length $L$ a priori.
We use a small MLP head on the global spectrum embedding (class token $h_{cls}$) combined with the precursor mass and charge:
$$ p(L | \mathcal{S}) = \text{Softmax}(\text{MLP}(h_{cls} \oplus \text{Emb}(M_{prec}) \oplus \text{Emb}(z))) $$
This predicts the length $L \in [1, 30]$. During inference, we sample the most likely length $\hat{L}$ and initialize a noise vector $x_1$ of size $\hat{L}$.

## 4. Peptide Decoder
The decoder is a Transformer designed to denoise the noisy peptide sequence $x_t \in \mathcal{V}^L$ at time $t$. 
The architecture avoids standard Post-LayerNorm in favor of **Pre-LayerNorm**, ensuring training stability without vanishing gradients in deep layers.

### Adaptive Layer Normalization (AdaLN)
Conditioning variables (diffusion time $t$, precursor mass $M_{prec}$, charge $z$, and length $L$) are fused into a global conditioner $c$:
$$ c = \text{Linear}( \text{Emb}(t) \oplus \text{Emb}(M_{prec}) \oplus \text{Emb}(z) \oplus \text{Emb}(L) ) $$
Instead of using cross-attention for these global scalars, we use Adaptive Layer Normalization (AdaLN), which modulates the features $h$ in the decoder blocks:
$$ \text{AdaLN}(h, c) = \gamma(c) \odot \text{LayerNorm}(h) + \beta(c) $$
where $\gamma$ and $\beta$ are computed via linear projections from $c$.

### SwiGLU Feed-Forward Networks
Instead of standard GeLU-based MLPs, we utilize SwiGLU activation, which has been shown (e.g., in LLaMA) to converge faster and yield better performance for language modeling:
$$ \text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV) $$
This block operates inside the decoder with an expansion factor of $\frac{8}{3}$, maintaining the parameter budget while increasing representational capacity.

### Classifier-Free Guidance (CFG)
To improve the alignment between the generated peptide and the input spectrum, we train the model with Classifier-Free Guidance. With probability $p_{uncond} = 0.1$ during training, the spectrum cross-attention conditioner is dropped (replaced by a learned unconditional embedding). During inference, the logits are extrapolated:
$$ \text{Logits}_{final} = \text{Logits}_{uncond} + s \cdot (\text{Logits}_{cond} - \text{Logits}_{uncond}) $$
where $s > 1.0$ is the guidance scale.

## References
[1] *$\pi$-Helix: Enhancing De Novo Peptide Sequencing with Mass-Complementary Attention*.
[2] *InstaNovo: De novo peptide sequencing via deep learning*.
