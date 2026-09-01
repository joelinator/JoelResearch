# Training and Optimization

Training a generative Transformer on 2.7 million mass spectra poses significant engineering and mathematical optimization challenges. The DFLowNovo pipeline has been aggressively optimized for multi-GPU setups on Google Cloud using PyTorch Lightning.

## 1. Optimizer and Learning Rate Scheduling
We use the **AdamW** optimizer with decoupled weight decay ($\beta_1=0.9, \beta_2=0.999$, weight decay=0.01). 

### The Necessity of Warmup
Architectures utilizing Pre-LayerNorm, SwiGLU, and Adaptive Layer Normalization (AdaLN) are susceptible to extreme initial gradient spikes. Without a warmup period, the Adam optimizer's moment estimates become permanently skewed, resulting in loss divergence (NaNs) or entrapment in suboptimal local minima.

To solve this, we implemented a custom scheduling lambda applied at every *step*:
1.  **Linear Warmup:** The learning rate ramps linearly from $0$ to $\eta_{max} = 3 \times 10^{-4}$ over the first 5% of total training steps.
2.  **Cosine Decay:** For the remaining 95% of steps, the learning rate decays following a half-cosine wave down to $0$:
    $$ \eta(s) = \eta_{max} \times 0.5 \left(1 + \cos\left( \pi \frac{s - s_{warmup}}{S_{total} - s_{warmup}} \right) \right) $$

## 2. Gradient Accumulation and Batch Sizing
Flow matching and discrete diffusion models suffer from high gradient variance because the loss is computed over random noise timesteps $t \sim \mathcal{U}(0, 1)$ for every batch. 

To stabilize the noise expectation, large batch sizes are required. Given the VRAM constraints of 40GB A100 GPUs, a physical batch size of 512 results in an Out-Of-Memory (OOM) error. We solve this via **Gradient Accumulation**:
*   Physical Batch Size: 128
*   Accumulate Grad Batches: 4
*   Effective Batch Size: 512

This guarantees mathematically identical gradients to a batch of 512 while keeping peak memory usage perfectly constrained.

## 3. Mixed Precision and Compilation (AMP & torch.compile)
To achieve training times of ~4 days on a single A100 GPU for 95 million samples, computational throughput was strictly optimized:
1.  **Automatic Mixed Precision (AMP):** We cast model weights and forward passes to `bfloat16`. Unlike `float16`, `bfloat16` shares the 8-bit exponent of `float32`, providing a massive dynamic range that prevents gradient scaling underflows common in Transformers.
2.  **torch.compile:** We utilize PyTorch 2.0's JIT compiler. The decoder relies on fixed-shape inner loops over static sequence lengths (e.g., max length 30). By compiling the model with `mode="reduce-overhead"`, CUDA graphs are captured and Python dispatch overhead is eliminated, increasing step throughput by ~20%.
