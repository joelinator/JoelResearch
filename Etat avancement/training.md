# Training and Optimization

Training a generative Transformer on 2.7 million mass spectra poses significant engineering and mathematical optimization challenges. The DFLowNovo pipeline has been optimized for multi-GPU setups on Google Cloud using PyTorch Lightning.

## 1. Optimizer and Learning Rate Scheduling
We use the **AdamW** optimizer with decoupled weight decay ($\beta_1=0.9, \beta_2=0.999$, weight decay=0.01). 

### Warmup and Cosine Decay
To stabilize initial gradient dynamics in architectures with Pre-LayerNorm, SwiGLU, and AdaLN:
1. **Linear Warmup:** The learning rate ramps linearly from $0$ to $\eta_{max} = 3 \times 10^{-4}$ over the first 5% of total training steps.
2. **Cosine Decay:** For the remaining 95% of steps, the learning rate decays following a half-cosine wave down to $0$:
   $$ \eta(s) = \eta_{max} \times 0.5 \left(1 + \cos\left( \pi \frac{s - s_{warmup}}{S_{total} - s_{warmup}} \right) \right) $$

## 2. Length Noising Regularization
To train the decoder to be robust against length prediction errors at inference time:
* Peptide length conditioning passed to the decoder is randomly perturbed by $\pm 1$ with probability $p = 0.10$.
* Length classification loss continues to be supervised with the true ground truth sequence length.
* Mass regularization loss is automatically masked out for length-perturbed batches to avoid introducing corrupted gradients.

## 3. Multi-Objective Loss Scheduling
The total loss is:
$$ \mathcal{L}_{total} = \mathcal{L}_{decoder} + \lambda(e)\mathcal{L}_{length} + \gamma(e)\mathcal{L}_{mass} $$
* $\lambda(e)$ ramps linearly from $0$ to $0.15$ over the first 15% of training.
* $\gamma(e)$ remains $0$ for the first 20% of training, then linearly ramps to $0.08$ by 60% of training.

## 4. Gradient Accumulation and Batch Sizing
To stabilize flow matching expectations across random noise timesteps $t \sim \mathcal{U}(0, 1)$:
* Physical Batch Size: 128
* Accumulate Grad Batches: 4
* Effective Batch Size: 512

## 5. Mixed Precision and Compilation (AMP & torch.compile)
1. **Automatic Mixed Precision (AMP):** Model weights and activations use `bfloat16` for broad dynamic range.
2. **torch.compile:** Uses PyTorch 2.0 JIT compilation (`mode="reduce-overhead"`) to eliminate Python dispatch overhead in diffusion loops.

## 6. Real-Time Telemetry and Monitoring
The training pipeline logs metrics simultaneously to:
* **CSV Logs:** Stored in `artifacts/<run_name>/version_0/metrics.csv` for parsing and plotting.
* **TensorBoard:** Event logs written to `artifacts/<run_name>/` allowing live dashboard inspection of train/valid losses, learning rate schedules, and generative precision/recall.
