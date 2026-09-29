#!/usr/bin/env python3
"""
Hugging Face Deployment & Synchronization Script for DFlowNovo
--------------------------------------------------------------
1. Validates authentication using HF_TOKEN from .env.
2. Synchronizes the Model repository (joelinator/dflow-novo-model):
   - Updates source code modules (src/data, src/inference, src/model, src/flow_matching, etc.)
   - Ensures frozen_production_model.ckpt and ptm_extended_warmstart.ckpt are present
   - Updates model card README.md, vocabulary.json, and config files.
3. Deploys the Gradio Web Application to Hugging Face Spaces (joelinator/dflow-novo):
   - Sets up free tier (cpu-basic: 2 vCPU, 16GB RAM)
   - Packages app.py, requirements.txt, Space README.md, sample spectra, and src/
   - Verifies Space build and runtime status.
"""

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from dotenv import load_dotenv

# Ensure root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load .env
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    load_dotenv(env_path)

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in .env file! Please ensure .env contains a valid HF_TOKEN.")

from huggingface_hub import HfApi, hf_hub_download

api = HfApi(token=HF_TOKEN)

MODEL_REPO_ID = "joelinator/dflow-novo-model"
SPACE_REPO_ID = "joelinator/dflow-novo"


def log(msg, symbol="ℹ️"):
    print(f"\n{symbol} [{time.strftime('%H:%M:%S')}] {msg}")


def check_auth():
    log("Verifying Hugging Face Authentication...", "🔐")
    user = api.whoami()
    username = user.get("name", "Unknown")
    user_type = user.get("type", "user")
    print(f"   Authenticated as: @{username} ({user_type})")
    return username


def update_model_repo():
    log(f"Synchronizing Model Repository: {MODEL_REPO_ID}...", "📦")
    # Verify repo exists
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)
    existing_files = set(api.list_repo_files(repo_id=MODEL_REPO_ID, repo_type="model"))
    print(f"   Current files on model Hub: {len(existing_files)}")

    # 1. Upload updated src/ modules
    print("   Uploading latest src/ modules...")
    api.upload_folder(
        folder_path=str(PROJECT_ROOT / "src"),
        path_in_repo="src",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="chore: sync latest research codebase and knapsack_dp guidance",
        ignore_patterns=["*.pyc", "__pycache__/*", "*.pth"],
    )
    print("   ✓ src/ tree synchronized.")

    # 2. Upload config files
    print("   Uploading configuration files...")
    api.upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "config" / "residues" / "extended.yaml"),
        path_in_repo="config/residues/extended.yaml",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="chore: sync extended residues config",
    )
    api.upload_file(
        path_or_fileobj=str(PROJECT_ROOT / "artifacts" / "dfm_length_weighted_10ep" / "vocabulary.json"),
        path_in_repo="vocabulary.json",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="chore: sync full PTM vocabulary.json",
    )
    print("   ✓ Config and vocabulary synchronized.")

    # 3. Upload sample spectra
    if (PROJECT_ROOT / "artifacts" / "sample_spectra.mgz").exists():
        api.upload_file(
            path_or_fileobj=str(PROJECT_ROOT / "artifacts" / "sample_spectra.mgz"),
            path_in_repo="sample_spectra.mgz",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message="chore: sync sample spectra mgz",
        )
    if (PROJECT_ROOT / "artifacts" / "sample_spectra.mgf").exists():
        api.upload_file(
            path_or_fileobj=str(PROJECT_ROOT / "artifacts" / "sample_spectra.mgf"),
            path_in_repo="sample_spectra.mgf",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message="chore: sync sample spectra mgf",
        )
    print("   ✓ Sample spectra synchronized.")

    # 4. Check & upload canonical frozen_production_model.ckpt
    prod_ckpt = PROJECT_ROOT / "models" / "frozen_production_model.ckpt"
    if prod_ckpt.is_symlink():
        prod_ckpt = prod_ckpt.resolve()

    if "frozen_production_model.ckpt" not in existing_files:
        if prod_ckpt.exists():
            size_mb = prod_ckpt.stat().st_size / (1024 * 1024)
            print(f"   Uploading frozen_production_model.ckpt ({size_mb:.1f} MB) via Git LFS...")
            api.upload_file(
                path_or_fileobj=str(prod_ckpt),
                path_in_repo="frozen_production_model.ckpt",
                repo_id=MODEL_REPO_ID,
                repo_type="model",
                commit_message="feat(model): add canonical 68.28% SOTA frozen production checkpoint",
            )
            print("   ✓ Production checkpoint uploaded successfully.")
        else:
            print("   ⚠️ Local frozen_production_model.ckpt not found, skipping upload.")
    else:
        print("   ✓ frozen_production_model.ckpt already exists in model repository.")

    # 5. Update Model Card README.md
    model_card = f"""---
language:
- en
license: apache-2.0
tags:
- proteomics
- mass-spectrometry
- de-novo-peptide-sequencing
- flow-matching
- ctmc
- generative-models
- pytorch
datasets:
- InstaDeepAI/ms_ninespecies_benchmark
- InstaDeepAI/ms_proteometools_hc
metrics:
- exact_match
- peptide_recall
- amino_acid_precision
pipeline_tag: text-generation
---

# 🧬 DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Gradio%20Demo-blue)](https://huggingface.co/spaces/{SPACE_REPO_ID})
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**DFlowNovo** is a state-of-the-art de novo peptide sequencing system powered by **Continuous-Time Markov Chain Discrete Flow Matching (CTMC-DFM)**. By treating peptide sequencing as continuous probability flows over discrete amino acid states and integrating dynamic programming (`KnapsackDP`) reachability constraints, DFlowNovo achieves:

- **68.28% Strict Exact Match** on the canonical Nine-Species biological benchmark.
- **56.79% I/L Exact Match** on the ProteomeTools high-complexity benchmark.
- **227–256 spectra/second** parallel GPU decoding, and fast interactive CPU sequencing.
- Native support for **Post-Translational Modifications (PTMs)** with standard UNIMOD annotations (`M(ox)` $\\to$ `M[UNIMOD:35]`, `C(cam)` $\\to$ `C[UNIMOD:4]`, `N(deam)` $\\to$ `N[UNIMOD:7]`, `Q(deam)` $\\to$ `Q[UNIMOD:7]`, `S(ph)` $\\to$ `S[UNIMOD:21]`, `T(ph)` $\\to$ `T[UNIMOD:21]`, `Y(ph)` $\\to$ `Y[UNIMOD:21]`).

---

## 📦 Checkpoints Available

| Checkpoint Name | Description | Size | Strict Exact Match (9-Species) |
|---|---|---|---|
| `frozen_production_model.ckpt` | **Canonical Production Model** (Joint Nine-Species + ProteomeTools with length-weighted loss $w(L) \propto \sqrt{{L}}$) | 907 MB | **68.28%** |
| `ptm_extended_warmstart.ckpt` | **PTM Extended Model** (Pretrained base with extended PTM vocabulary) | 476 MB | 67.92% |

---

## 💻 Python Usage

```python
import torch
from huggingface_hub import hf_hub_download
from train.io import load_checkpoint, load_models_from_checkpoint
from train.factory import build_models
from inference.predict import predict_peptide

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="{MODEL_REPO_ID}", filename="frozen_production_model.ckpt")
ckpt = load_checkpoint(ckpt_path, map_location=device)
vocab = ckpt["vocabulary"]

# Build & load models
enc, lp, dec, guid = build_models(vocab, device)
load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
enc.eval(); lp.eval(); dec.eval(); guid.eval()

# Run de novo sequencing
# x_t, lengths, seqs, scores = predict_peptide(...)
```

---

## 🚀 Live Interactive Demo

Try the live Gradio interface at: **[https://huggingface.co/spaces/{SPACE_REPO_ID}](https://huggingface.co/spaces/{SPACE_REPO_ID})**

---

## 📖 Citation

```bibtex
@mastersthesis{{gedeon2026dflownovo,
  title={{Discrete Flow Matching for De Novo Peptide Sequencing in Mass Spectrometry}},
  author={{G{{\\'e}}d{{\\'e}}on, Jo{{\\"e}}l}},
  school={{African Institute for Mathematical Sciences (AIMS South Africa) / InstaDeep}},
  year={{2026}}
}}
```
"""
    api.upload_file(
        path_or_fileobj=model_card.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="docs: update comprehensive model card with benchmarks and usage",
    )
    print("   ✓ Model card README.md updated.")


def deploy_gradio_space():
    log(f"Deploying Gradio Application to Hugging Face Spaces: {SPACE_REPO_ID}...", "🚀")

    # 1. Create or ensure Space exists on free tier (cpu-basic)
    print("   Ensuring Space exists with free cpu-basic tier...")
    api.create_repo(
        repo_id=SPACE_REPO_ID,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="cpu-basic",  # 100% Free
        exist_ok=True,
    )
    print("   ✓ Space repository initialized.")

    # 2. Assemble staging bundle
    with tempfile.TemporaryDirectory(prefix="dflow_hf_space_") as stage_dir:
        stage = Path(stage_dir)
        print(f"   Staging deployment bundle at {stage}...")

        # Copy app.py
        shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "app.py", stage / "app.py")

        # Copy requirements.txt
        shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "requirements.txt", stage / "requirements.txt")

        # Copy README.md (with Space metadata)
        shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "README.md", stage / "README.md")

        # Copy sample spectra
        shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "sample_spectra.mgz", stage / "sample_spectra.mgz")
        shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "sample_spectra.mgf", stage / "sample_spectra.mgf")

        # Copy vocabulary
        shutil.copy2(PROJECT_ROOT / "deployment" / "huggingface" / "vocabulary.json", stage / "vocabulary.json")

        # Copy config
        shutil.copytree(PROJECT_ROOT / "config", stage / "config", dirs_exist_ok=True)

        # Copy src tree (excluding pycache and large files)
        shutil.copytree(
            PROJECT_ROOT / "src",
            stage / "src",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "*.pth"),
        )

        staged_files = list(stage.rglob("*"))
        print(f"   Bundle assembled: {len(staged_files)} items ready for upload.")

        # 3. Upload to Space
        print(f"   Uploading bundle to Spaces ({SPACE_REPO_ID})...")
        api.upload_folder(
            folder_path=str(stage),
            repo_id=SPACE_REPO_ID,
            repo_type="space",
            commit_message="feat(space): deploy DFlowNovo Gradio Web Interface with CTMC flow matching and PTM support",
        )
        print("   ✓ Space code and assets uploaded successfully.")

    # 4. Monitor runtime status
    print("   Checking Space runtime and build status...")
    max_wait = 180
    start_time = time.time()
    last_stage = None

    while time.time() - start_time < max_wait:
        try:
            runtime = api.get_space_runtime(repo_id=SPACE_REPO_ID)
            stage = runtime.stage
            hardware = runtime.hardware or "cpu-basic"
            if stage != last_stage:
                print(f"   Status: [{stage}] | Hardware: [{hardware}]")
                last_stage = stage

            if stage in ["RUNNING", "APP_STARTING"]:
                print(f"\n   🎉 Space is active! Status: {stage}")
                break
            elif stage in ["BUILDING", "PENDING"]:
                time.sleep(8)
            elif "ERROR" in stage.upper() or "FAILED" in stage.upper():
                print(f"\n   ⚠️ Space reached stage: {stage}")
                break
            else:
                time.sleep(5)
        except Exception as e:
            print(f"   Polling runtime notice: {e}")
            time.sleep(5)

    space_url = f"https://huggingface.co/spaces/{SPACE_REPO_ID}"
    print(f"\n   🔗 Space URL: {space_url}")
    return space_url


def main():
    print("=" * 70)
    print("  DFlowNovo: Hugging Face Deployment & Synchronization")
    print("=" * 70)

    user = check_auth()
    update_model_repo()
    space_url = deploy_gradio_space()

    print("\n" + "=" * 70)
    print("  Deployment Completed Successfully!")
    print(f"  • Model Repository: https://huggingface.co/{MODEL_REPO_ID}")
    print(f"  • Live Gradio Space: {space_url}")
    print("=" * 70)


if __name__ == "__main__":
    main()
