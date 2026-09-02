# DFLowNovo: De Novo Peptide Sequencing via Discrete Flow Matching

Research Project by Joel Gedeon at AIMS South Africa.

## 1. Overview

DFLowNovo is a non-autoregressive deep learning framework designed to predict peptide amino acid sequences directly from tandem mass spectrometry (MS/MS) data using Discrete Flow Matching (DFM). 

Unlike traditional autoregressive models that generate sequences residue-by-residue in an iterative left-to-right fashion, DFLowNovo generates all positions simultaneously across a continuous diffusion time horizon $t \in [0, 1]$. This formulation enables fast inference, explicit global mass constraints, and controllable multi-candidate beam search.

---

## 2. Model Architecture

The architecture consists of three integrated components:

### 2.1. Spectrum Encoder and Feature Engineering
* **Mass Spectrum Representation:** Tandem mass spectra are represented by the top $k = 200$ peaks, characterized by their mass-to-charge ratio $(m/z)_i$ and intensity $I_i$.
* **$m/z$-Complementary Peak Projection:** To facilitate detection of complementary $b$-ion and $y$-ion pairs, theoretical complementary masses are explicitly computed and injected:
  $$(m/z)_{comp, i} = M_{prec} - (m/z)_i$$
* **Peak Token Fusion:** Sinusoidal embeddings of $(m/z)_i$ and $(m/z)_{comp, i}$ are concatenated with projected square-root normalized intensities and passed through a 6-layer Multi-Head Self-Attention (MHSA) Transformer encoder.

### 2.2. Peptide Length Predictor
* Non-autoregressive sequence generation requires knowing the sequence length $L$ beforehand.
* A classification head over the global spectrum class token ($h_{cls}$), precursor mass, and precursor charge estimates the probability distribution $p(L \mid \mathcal{S})$ for $L \in [1, 30]$.

### 2.3. Peptide Decoder
* **Discrete Flow Matching Decoder:** Generates the sequence $x_t \in \mathcal{V}^L$ across discrete probability vectors.
* **Discrete Length Embeddings:** Length conditioning uses dedicated `nn.Embedding` lookup tables for robust 1-based index modulation.
* **Pre-LayerNorm & Adaptive Layer Normalization (AdaLN):** Global conditions (diffusion time $t$, precursor mass $M_{prec}$, charge $z$, and length $L$) modulate intermediate activations via AdaLN scale and shift parameters.
* **SwiGLU Feed-Forward Networks:** Employs Swish-Gated Linear Units with an expansion factor of $8/3$ for improved convergence and representational capacity.
* **Classifier-Free Guidance (CFG):** Drops the spectral cross-attention conditioner during training with probability $p_{uncond} = 0.1$, enabling logit extrapolation during inference:
  $$\text{Logits}_{final} = \text{Logits}_{uncond} + s \cdot (\text{Logits}_{cond} - \text{Logits}_{uncond})$$

---

## 3. Training and Loss Formulation

### 3.1. Multi-Objective Scheduled Loss
The training objective balances sequence generation, length prediction, and physical mass adherence:
$$\mathcal{L}_{total} = \mathcal{L}_{decoder} + \lambda(e)\mathcal{L}_{length} + \gamma(e)\mathcal{L}_{mass}$$

* **Decoder Cross-Entropy ($\mathcal{L}_{decoder}$):** Standard cross-entropy evaluated over unmasked active residue positions.
* **Length Classification Loss ($\mathcal{L}_{length}$):** Cross-entropy between predicted length logits and ground truth peptide length, warmed up linearly via $\lambda(e)$ over the first 15% of training.
* **Physics-Based Mass Regularization ($\mathcal{L}_{mass}$):** Huber penalty on the expected sequence mass $\mathbb{E}[M_{pred}]$ relative to target precursor neutral mass $(M_{prec} - M_{H_2O})$, activated via $\gamma(e)$ after epoch 20% to prevent gradient instability during early random states.

### 3.2. Length Noising Regularization
* During training, input length conditioning to the decoder is randomly perturbed by $\pm 1$ with probability $p = 0.10$.
* This trains the decoder to be resilient to length predictor errors during generative decoding.
* Mass loss is automatically masked on perturbed batches to avoid penalizing mismatched length targets.

### 3.3. Optimization & Throughput
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = 0.01) with linear warmup (first 5% steps) and cosine learning rate decay.
* **Multi-GPU / Gradient Accumulation:** Effective batch size of 512 (physical batch 128 accumulated over 4 batches) to stabilize flow matching gradient expectations.
* **Precision & Compilation:** Automatic Mixed Precision with `bfloat16` and JIT compilation via `torch.compile(mode="reduce-overhead")`.

---

## 4. Inference: Top-k Length Beam Decoding

During inference, DFLowNovo employs parallelized Top-$k$ Length Beam Decoding:
1. The length predictor extracts the top-$K$ most probable candidate lengths $[L_1, \dots, L_K]$ (default: $K=3$).
2. The decoder runs diffusion sampling on all $K$ candidate lengths in a single batched GPU pass.
3. Candidate sequences $\hat{Y}^{(k)}$ are evaluated using the joint scoring function:
   $$\text{Score}(\hat{Y}^{(k)}) = -\mathcal{H}(\hat{Y}^{(k)} \mid \mathcal{S}) - \alpha \cdot \left\vert{} \sum_{i=1}^{L_k} m(\hat{Y}_i^{(k)}) - (M_{prec} - M_{H_2O}) \right\vert{}$$
4. The candidate maximizing spectral log-likelihood while minimizing precursor mass error is selected.

---

## 5. Progress: What Has Been Done

* **Dataset & Physical Preprocessing:** Integrated the ProteomeTools HC dataset (~2.7M spectra), implemented intensity root-normalization, precursor peak removal, and theoretical complementary peak generation.
* **Model Implementation:** Built the full DFLowNovo architecture (Spectrum Encoder, Length Predictor, Discrete Flow Matching Decoder with AdaLN, SwiGLU, and discrete length embeddings).
* **Multi-Task Loss Pipeline:** Formulated scheduled multi-objective loss with dynamic warmup for length and Huber mass regularizers.
* **Length Noising:** Integrated batch-level length noising during training with automatic mass regularization masking.
* **Top-k Length Beam Search:** Vectorized parallel length beam search with joint spectral and mass scoring, active by default.
* **Multi-GPU Lightning Pipeline:** Integrated PyTorch Lightning training with dual CSV and TensorBoard logging.
* **Cloud Infrastructure Automation:** Built deployment scripts and environment templates for Google Cloud Platform VM provisioning, tmux background runs, and live telemetry.

---

## 6. To-Do: Plans for Next Days

* [ ] **Full-Scale Cloud VM Training:** Launch and complete 30-epoch training on the complete 2.7M ProteomeTools dataset on an A100 VM.
* [ ] **Benchmark Against Baselines:** Measure exact Amino Acid Precision/Recall and Peptide Precision/Recall against InstaNovo and Pi-PrimeNovo.
* [ ] **Guidance Scale & Beam Hyperparameter Tuning:** Evaluate combinations of CFG scale $s \in [1.0, 2.0]$ and length beam penalty $\alpha \in [0.05, 0.2]$.
* [ ] **Attention Interpretability Analysis:** Inspect learned self-attention maps in the Spectrum Encoder to verify attention concentration on complementary $b/y$ ion pairs.

---

## 7. Google Cloud Platform (GCP) Setup & Monitoring

### 7.1. Configuration File (.env)
Copy `.env.example` to `.env` and fill in your secrets:
```bash
cp .env.example .env
```
Key variables to specify:
* `HF_TOKEN`: Hugging Face access token for dataset downloading.
* `GCP_PROJECT`: Your GCP Project ID.
* `GCP_ZONE`: Target compute zone (e.g., `europe-west4-a` or `us-central1-a`).
* `GCP_VM_NAME`: Name for the training instance (e.g., `dfm-train-a100`).

### 7.2. Creating and Provisioning the VM
Run the automated creation script:
```bash
bash scripts/gcp_create_vm.sh
```

### 7.3. Syncing Code and Launching Training
Sync repository files and launch the background training run:
```bash
bash scripts/gcp_sync_project.sh
bash scripts/gcp_run_remote.sh
```

### 7.4. Real-Time Monitoring
You can monitor training evolution at any time:

* **Live Output Logs:**
  ```bash
  bash scripts/gcp_run_remote.sh --logs
  ```
* **Interactive Tmux Session:**
  ```bash
  bash scripts/gcp_run_remote.sh --attach
  ```
* **TensorBoard Dashboard:**
  Port-forward the TensorBoard server from the VM:
  ```bash
  gcloud compute ssh dfm-train-a100 --zone=europe-west4-a -- -L 6006:localhost:6006
  ```
  On the VM, run:
  ```bash
  tensorboard --logdir ~/dfm-joelresearch/artifacts --port 6006
  ```
  Open `http://localhost:6006` in your local browser to view loss curves, learning rates, and generative validation metrics.

---

## 8. Literature Review

* [InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments | Nature Machine Intelligence](https://scholar.google.com/scholar?q=InstaNovo+enables+diffusion-powered+de+novo+peptide+sequencing)
* [[2406.04843] Variational Flow Matching for Graph Generation](https://arxiv.org/abs/2406.04843)
* [[2402.04997] Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997)
* [Regressor-guided Diffusion Model for De Novo Peptide Sequencing with Explicit Mass Control](https://scholar.google.com/scholar?q=Regressor-guided+Diffusion+Model+for+De+Novo+Peptide+Sequencing+with+Explicit+Mass+Control)
* [π-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing | Nature Communications](https://scholar.google.com/scholar?q=π-PrimeNovo:+an+accurate+and+efficient+non-autoregressive+deep+learning+model)
