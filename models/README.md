# Pretrained Model Checkpoints

This directory houses pretrained DFlowNovo model weights for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS).

### Available Checkpoints

1. **Production Model Checkpoint**:
   - **File Name**: `frozen_production_model.ckpt` (454 MB lean checkpoint, stripped of AdamW optimizer states)
   - **Training**: Pretrained on synthetic peptides from High-Complexity ProteomeTools (HC-PT; [InstaDeepAI/ms_proteometools](https://huggingface.co/datasets/InstaDeepAI/ms_proteometools)) and fine-tuned on the Nine-Species benchmark ([InstaDeepAI/ms_ninespecies_benchmark](https://huggingface.co/datasets/InstaDeepAI/ms_ninespecies_benchmark), trained on 8 species, evaluated on held-out yeast).
   - **Metrics**: Evaluated by Peptide Accuracy (strict and I/L-tolerant exact match) and Amino Acid F1 Score.

2. **MassIVE-KB Pretrained Checkpoint**:
   - **File Name**: `dfm_massive_kb.ckpt` (227 MB)
   - **Training**: Pretrained exclusively on ~28M human MS/MS spectra from the MassIVE-KB dataset ([MSV000084075](https://massive.ucsd.edu/ProteoSAFe/dataset.jsp?task=86ee46be220f4f9f9ce6d0ef06a7be7c)) without fine-tuning (supports peptide sequences up to length 50).
   - **Metrics**: Evaluated by Peptide Accuracy and Amino Acid F1 Score.

### Download Checkpoints
You can download checkpoints directly via:
- **Hugging Face Hub**: [`joelinator/dflow-novo-model`](https://huggingface.co/joelinator/dflow-novo-model)
- **GitHub Releases**:
  - Production Model: [v0.2.0](https://github.com/joelinator/JoelResearch/releases/tag/v0.2.0)
  - MassIVE-KB Model: [v0.3.0](https://github.com/joelinator/JoelResearch/releases/tag/v0.3.0)

```bash
# Using huggingface-cli
huggingface-cli download joelinator/dflow-novo-model frozen_production_model.ckpt --local-dir models/
huggingface-cli download joelinator/dflow-novo-model dfm_massive_kb.ckpt --local-dir models/

# Using gh cli
gh release download v0.2.0 -p frozen_production_model.ckpt -D models/
gh release download v0.3.0 -p dfm_massive_kb.ckpt -D models/
```

When running inference via the CLI, Python API, or Gradio web app, DFlowNovo automatically detects these models.
