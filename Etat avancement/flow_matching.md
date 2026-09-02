# Discrete Flow Matching

Standard continuous Flow Matching and Diffusion models (like DDPMs used in vision) operate on continuous variables $x \in \mathbb{R}^d$. However, peptide sequences consist of discrete, categorical tokens (amino acids). DFLowNovo utilizes **Discrete Flow Matching (DFM)** [1], a framework that defines probability paths and vector fields over discrete state spaces.

## 1. The State Space
Let $\mathcal{V}$ be the vocabulary of size $K$ (amino acids, pad, mask). A peptide sequence is a vector $x_0 \in \mathcal{V}^L$.
Instead of adding Gaussian noise, the corruption process transitions discrete tokens into other discrete tokens over continuous time $t \in [0, 1]$.

## 2. The Forward Process (Noising)
We construct a time-dependent transition probability matrix $Q_{t|0} \in \mathbb{R}^{K \times K}$. The element $[Q_{t|0}]_{ij}$ gives the probability $P(x_t = j | x_0 = i)$.

We define a **Masking Scheme** as our probability path. As $t$ increases from $0$ to $1$, the original data tokens are progressively replaced by a special absorbing `<mask_token>`.
Let $\sigma(t)$ be a monotonic noise schedule where $\sigma(0) = 0$ and $\sigma(1) = 1$. (e.g., using a cosine schedule).
The transition probabilities are defined as:
$$ P(x_t = \text{mask} | x_0) = \sigma(t) $$
$$ P(x_t = x_0 | x_0) = 1 - \sigma(t) $$

At $t=0$, $x_0$ is completely clean (the ground truth peptide). At $t=1$, the state $x_1$ is entirely composed of `<mask_token>` (pure noise).

## 3. The Reverse Process (Denoising)
In the reverse process, we want to sample from $P(x_{t-dt} | x_t)$ starting from $x_1 = [\text{mask}, \dots, \text{mask}]$.
Using Bayes' theorem and the Markov property of the forward process, the exact reverse transition probability relies on the marginal probability of the true data:
$$ P(x_{t-dt} | x_t) = \sum_{x_0} P(x_{t-dt} | x_t, x_0) P(x_0 | x_t) $$

Since the true data distribution $P(x_0 | x_t)$ is unknown, we approximate it with our neural network (the Peptide Decoder):
$$ p_\theta(x_0 | x_t, \mathcal{S}) $$

### Reverse Sampling Step
During inference (decoding), we discretize the time interval $[0, 1]$ into $N$ steps (e.g., $N=20$).
At step $t$:
1.  The model receives the noisy sequence $x_t$ (containing a mix of amino acids and masks) and the spectrum $\mathcal{S}$.
2.  It outputs categorical logits $p_\theta(x_0 | x_t)$.
3.  We sample a predicted clean sequence $\hat{x}_0 \sim p_\theta$.
4.  We compute the next state $x_{t-dt}$ by sampling from the known forward transition matrix: $x_{t-dt} \sim P(x_{t-dt} | \hat{x}_0, x_t)$.

Because our forward process is a simple masking scheme, step 4 reduces to a stochastic unmasking operation. Specifically, a fraction of the currently masked tokens are permanently replaced by the model's highest-confidence predictions, and the process repeats until $t=0$ and no masks remain.

## 4. Rate Matrices and Objective
In rigorous continuous-time discrete diffusion, the transitions are defined by a time-dependent rate matrix $R(t) = \frac{d}{dt} Q_{t|0} \cdot Q_{t|0}^{-1}$.
Flow matching optimizes the cross-entropy of the vector field. For the masking scheme, this mathematically simplifies directly to standard Cross-Entropy regression on the masked tokens, scaled by the derivative of the noise schedule:
$$ \mathcal{L}_{flow}(t) = - \mathbb{E}_{t, x_0, x_t} \left[ \sum_{i=1}^L \mathbb{I}(x_t^{(i)} = \text{mask}) \log p_\theta(x_0^{(i)} | x_t, \mathcal{S}) \right] $$
This proves that optimizing standard CE loss on masked sequences perfectly aligns with the discrete flow matching ODE, allowing us to leverage massive optimizations from Masked Language Modeling literature while retaining the rigorous theoretical foundation of continuous normalizing flows.

## References
[1] *Discrete Flow Matching* (arXiv:2402.04997)
