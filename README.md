# DFlowNovo

Discrete Flow Matching for *de novo* peptide sequencing from tandem mass spectrometry (MS/MS) data.

## Overview
DFlowNovo formulates de novo sequencing as continuous-time probability flow over categorical simplices. By combining non-autoregressive velocity fields with dynamic programming mass constraints and classifier-free guidance, DFlowNovo generates peptide sequences from tandem mass spectra in fixed sampling steps.

## Installation
```bash
git clone https://github.com/joelinator/JoelResearch.git
cd JoelResearch
pip install -r requirements.txt
pip install -e .
```

## Structure
- `src/`: Core library (data loaders, model architecture, flow matching, inference)
- `scripts/`: Training, evaluation, and inference CLI utilities
- `tests/`: Unit and integration test suite
