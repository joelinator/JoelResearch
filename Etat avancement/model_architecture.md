# Model Architecture: DFM (Discrete Flow Matching) for De Novo Peptide Sequencing

## 1. Overview
The proposed model, **DFLowNovo**, solves the de novo peptide sequencing problem by predicting an amino acid sequence $y = (y_1, y_2, \dots, y_L)$ from a tandem mass spectrum $\mathcal{S}$. We frame this as a conditional generative modeling task using Discrete Flow Matching (DFM). The model comprises three main modules:
1. **Spectrum Encoder:** Extracts a continuous representation of the mass spectrum $\mathcal{S}$.
2. **Peptide Length Predictor:** Estimates the length $L$ of the peptide from the spectrum embedding.
3. **Peptide Decoder:** An iterative discrete flow matching decoder that generates the peptide sequence conditioned on the spectrum, precursor mass, charge, predicted length, and diffusion time $t$.

## 2. Spectrum Encoder and Feature Engineering
The mass spectrum $\mathcal{S}$ is represented as a set of $k$ peaks, where each peak $i$ has a mass-to-charge ratio $(m/z)_i$ and an intensity $I_i$. 

### The $m/z$-Complementary Peak Idea
In tandem mass spectrometry (e.g., HCD fragmentation), peptide bonds break to form $b$-ions and $y$-ions. If a peak is a $b$-ion, its corresponding $y$-ion is usually present in the spectrum. The relationship is governed by the precursor mass $M_{prec}$:
$$ (m/z)_{comp, i} = M_{prec} - (m/z)_i $$
By explicitly providing $(m/z)_{comp, i}$ to the encoder, we assist the attention mechanism in recognizing structural complementary pairs ($b/y$ pairs).

### Encoder Architecture
For each of the top $k=200$ peaks, we project the tuple $((m/z)_i, (m/z)_{comp, i}, I_i)$ into a continuous space using Sinusoidal embeddings (for $m/z$ values) and MLPs. These features are fused into a single token per peak $h_i \in \mathbb{R}^d$. 
The encoder applies $N_e = 6$ layers of Multi-Head Self-Attention (MHSA) over these peak tokens to model intra-spectrum peak-to-peak interactions.

## 3. Peptide Length Predictor
Non-autoregressive flow matching models generate all tokens simultaneously, requiring the sequence length $L$ beforehand.
We use a classification head on the global spectrum embedding (class token $h_{cls}$) combined with the precursor mass and charge:
$$ p(L \mid \mathcal{S}) = \text{Softmax}(\text{MLP}(h_{cls} \oplus \text{Emb}(M_{prec}) \oplus \text{Emb}(z))) $$
This predicts the length $L \in [1, 30]$.

## 4. Peptide Decoder
The decoder is a Transformer designed to denoise the noisy peptide sequence $x_t \in \mathcal{V}^L$ at time $t$. 
The architecture uses **Pre-LayerNorm** for training stability without vanishing gradients in deep layers.

### Discrete Length Embeddings
Length conditioning is handled via a dedicated `nn.Embedding(max_length + 1, emb_dim)` lookup table, ensuring exact 1-based index representation across all valid lengths.

### Adaptive Layer Normalization (AdaLN)
Conditioning variables (diffusion time $t$, precursor mass $M_{prec}$, charge $z$, and length $L$) are fused into a global conditioner $c$:
$$ c = \text{Linear}( \text{Emb}(t) \oplus \text{Emb}(M_{prec}) \oplus \text{Emb}(z) \oplus \text{Emb}(L) ) $$
Adaptive Layer Normalization (AdaLN) modulates the features $h$ in the decoder blocks:
$$ \text{AdaLN}(h, c) = \gamma(c) \odot \text{LayerNorm}(h) + \beta(c) $$
where $\gamma$ and $\beta$ are computed via linear projections from $c$.

### SwiGLU Feed-Forward Networks
We utilize SwiGLU activation with an expansion factor of $8/3$:
$$ \text{SwiGLU}(x, W, V) = \text{Swish}(xW) \odot (xV) $$

### Classifier-Free Guidance (CFG)
During training with probability $p_{uncond} = 0.1$, the spectrum cross-attention conditioner is dropped. During inference, logits are extrapolated:
$$ \text{Logits}_{final} = \text{Logits}_{uncond} + s \cdot (\text{Logits}_{cond} - \text{Logits}_{uncond}) $$
where $s \ge 1.0$ is the guidance scale.

## 5. Top-k Length Beam Decoding at Inference
Rather than decoding only the single argmax length, the model evaluates the top-$K$ candidate lengths ($K=3$) in parallel over an expanded batch of size $B \times K$. The generated sequences are scored according to:
$$ \text{Score}(\hat{Y}^{(k)}) = -\mathcal{H}(\hat{Y}^{(k)} \mid \mathcal{S}) - \alpha \cdot \left\vert{} \sum_{i=1}^{L_k} m(\hat{Y}_i^{(k)}) - (M_{prec} - M_{H_2O}) \right\vert{} $$
The candidate maximizing spectral confidence and minimizing precursor mass error is returned.

## References
[1] Pi-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing.
[2] InstaNovo: De novo peptide sequencing via deep learning.
