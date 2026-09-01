# DFLowNovo: De Novo Peptide Sequencing via Discrete Flow Matching
Research Project by Joel Gedeon at AIMS South Africa.

## 🎯 Overview
This project aims to use **Discrete Flow Matching (DFM)** to predict peptide sequences directly from mass spectrum data. By leveraging non-autoregressive flow matching models, we intend to deliver a fast and accurate alternative to traditional autoregressive methods.

## 🚀 Progress: What Has Been Done
* **Dataset & Preprocessing Pipeline** 
  * Integrated the ProteomeTools HC dataset (~2.7M spectra).
  * Implemented physical preprocessing, including peak filtering, intensity normalization, and the calculation of $m/z$-complementary peaks to explicitly capture structural b/y-ion pairs.
* **DFLowNovo Architecture Design**
  * **Spectrum Encoder**: Built an encoder with Multi-Head Self-Attention using sinusoidal embeddings for mass/charge values.
  * **Peptide Length Predictor**: Implemented a length prediction head for non-autoregressive generation.
  * **Peptide Decoder**: Designed a Transformer-based discrete flow matching decoder utilizing Pre-LayerNorm, Adaptive Layer Normalization (AdaLN) for global conditioning, and SwiGLU activation for stable, efficient training.
  * **Classifier-Free Guidance (CFG)**: Implemented CFG dropout during training to improve alignment between the spectrum and generated peptide.
* **Loss Functions Formulation**
  * Established a multi-objective loss balancing Decoder Cross-Entropy, Length Prediction Cross-Entropy, and a Physics-Based Mass Regularization (Huber penalty).
  * Designed dynamic scheduling to ramp up the length and mass losses correctly without destabilizing early training.
* **Training Optimizations**
  * Configured an optimized PyTorch Lightning pipeline for multi-GPU setups.
  * Implemented AdamW with linear warmup and cosine decay.
  * Applied Gradient Accumulation, Automatic Mixed Precision (bfloat16), and `torch.compile` to maximize step throughput and prevent OOM errors on A100 GPUs.
* **Metrics Tracking**
  * Set up highly reliable proxies for evaluating Amino Acid Accuracy and exact Peptide Recall during training.

## 📝 To-Do: Plans for the Next Days
* [ ] **Run Full-Scale Training**: Execute the optimized training pipeline on the entire 2.7 million mass spectra dataset.
* [ ] **Benchmark against Baselines**: Evaluate the model's exact Peptide Recall and run comparisons against autoregressive baselines (e.g., InstaNovo) and other non-autoregressive models.
* [ ] **Hyperparameter Tuning**: Optimize the Classifier-Free Guidance (CFG) scale $s$ during inference to find the optimal trade-off between sample quality and sequence diversity.
* [ ] **Interpretability Analysis**: Analyze the learned attention weights within the Spectrum Encoder to verify whether it correctly identifies b/y ion pairs via the complementary peak mechanism.

## 📚 Literature Review (Papers Read)
* [InstaNovo enables diffusion-powered de novo peptide sequencing in large-scale proteomics experiments | Nature Machine Intelligence](https://scholar.google.com/scholar?q=InstaNovo+enables+diffusion-powered+de+novo+peptide+sequencing)
* [[2406.04843] Variational Flow Matching for Graph Generation](https://arxiv.org/abs/2406.04843)
* [[2402.04997] Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design](https://arxiv.org/abs/2402.04997)
* [Regressor-guided Diffusion Model for De Novo Peptide Sequencing with Explicit Mass Control](https://scholar.google.com/scholar?q=Regressor-guided+Diffusion+Model+for+De+Novo+Peptide+Sequencing+with+Explicit+Mass+Control)
* [π-PrimeNovo: an accurate and efficient non-autoregressive deep learning model for de novo peptide sequencing | Nature Communications](https://scholar.google.com/scholar?q=π-PrimeNovo:+an+accurate+and+efficient+non-autoregressive+deep+learning+model)
