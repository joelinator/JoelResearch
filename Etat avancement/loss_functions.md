# Multi-Objective Loss Functions

The DFLowNovo model is trained using a multi-task objective that balances the primary generative modeling of the sequence, the prediction of the sequence length, and a physics-based regularization term.

The total loss is defined as:
$$ \mathcal{L}_{total} = \mathcal{L}_{decoder} + \lambda(e) \mathcal{L}_{length} + \gamma(e) \mathcal{L}_{mass} $$
where $e$ is the current training epoch (or step), and $\lambda, \gamma$ are dynamic scheduling weights.

## 1. Decoder Loss ($\mathcal{L}_{decoder}$)
This is the primary flow-matching objective. Given a noisy sequence $x_t$ at time $t$, the decoder attempts to predict the clean target sequence $x_0$. 
Since the state space is discrete (amino acids), the loss is the standard Cross-Entropy over the vocabulary for all unpadded tokens:
$$ \mathcal{L}_{decoder} = - \frac{1}{L} \sum_{i=1}^{L} \log p_\theta(x_0^{(i)} | x_t, \mathcal{S}, M_{prec}, z, L, t) $$
Padding tokens (`<pad>`) are ignored in this calculation via `ignore_index`.

## 2. Length Loss ($\mathcal{L}_{length}$)
Because non-autoregressive decoding requires the sequence length $L$ beforehand, we train a classifier on the spectrum embeddings to predict $L$.
We frame this as a 30-class classification problem (for lengths $L \in [1, 30]$):
$$ \mathcal{L}_{length} = \text{CrossEntropy}(\text{Logits}_{length}, L_{true}) $$
**Schedule $\lambda(e)$:** The length loss has a linear warmup from 0 to $\lambda_{max} = 0.15$ over the first 15% of training. This ensures the model focuses entirely on denoising early on, before the length predictor gradients interfere with the spectrum encoder representations.

## 3. Physics-Based Mass Regularization ($\mathcal{L}_{mass}$)
Unlike NLP, where words have no physical constraint, peptides must physically sum to the precursor mass observed in the mass spectrometer. 
Let $m(a)$ be the mass of amino acid $a$. The total sequence mass should satisfy:
$$ \sum_{i=1}^{L} m(y_i) + M_{H_2O} \approx M_{prec} $$
where $M_{H_2O} = 18.010565$ Da is the mass of the terminal water molecule.

To enforce this, we compute the expected mass of the predicted sequence using the softmax probabilities $p_{i,a} = p_\theta(x_0^{(i)} = a)$:
$$ \mathbb{E}[M_{pred}] = \sum_{i=1}^{L} \sum_{a \in \mathcal{V}} p_{i,a} \cdot m(a) $$
The loss is then a **Huber penalty** (smooth L1) between the expected mass and the target residue mass $(M_{prec} - M_{H_2O})$:
$$ \text{error} = \frac{|\mathbb{E}[M_{pred}] - (M_{prec} - M_{H_2O})|}{M_{prec} - M_{H_2O}} $$
$$ \mathcal{L}_{mass} = \begin{cases} 
0.5 \times \text{error}^2 & \text{if error} < \delta \\ 
\delta (\text{error} - 0.5 \delta) & \text{otherwise} 
\end{cases} $$
**Schedule $\gamma(e)$:** Applying this loss at the beginning of training (when outputs are random noise) creates chaotic gradients. Therefore, $\gamma(e)$ remains $0$ for the first 20% of training, then linearly ramps to $0.08$ by 60% of training, holding steady thereafter. This forces the model to refine its valid chemical outputs to precisely match the precursor mass.
