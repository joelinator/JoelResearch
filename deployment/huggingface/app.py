import gzip
import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import gradio as gr

# Ensure root and src/ are in sys.path
ROOT_DIR = Path(__file__).resolve().parent
for p in [str(ROOT_DIR), str(ROOT_DIR / "src")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from huggingface_hub import hf_hub_download

from data.constants import AA_MASSES_DICT, M_H, M_H2O
from data.data import build_vocabulary, parse_peptide, to_unimod_sequence
from flow_matching.scheduler import cosine_scheduler
from inference.predict import predict_peptide
from train.factory import build_models
from train.io import load_checkpoint, load_models_from_checkpoint

# Global device and cached model instances
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHED_MODELS = None
HF_MODEL_REPO = "joelinator/dflow-novo-model"
CKPT_FILENAME = "ptm_extended_warmstart.ckpt"
VOCAB_FILENAME = "vocabulary.json"


def get_model():
    """Load model weights and vocabulary, with local cache and Hub fallback."""
    global CACHED_MODELS
    if CACHED_MODELS is not None:
        return CACHED_MODELS

    # 1. Resolve checkpoint path
    local_candidates = [
        ROOT_DIR / "checkpoints" / CKPT_FILENAME,
        ROOT_DIR / CKPT_FILENAME,
        ROOT_DIR.parent.parent / "artifacts" / "ptm_extended_warmstart" / CKPT_FILENAME,
    ]
    ckpt_path = None
    for cand in local_candidates:
        if cand.exists():
            ckpt_path = str(cand)
            print(f"Loading checkpoint from local path: {ckpt_path}")
            break

    if ckpt_path is None:
        print(f"Downloading checkpoint from Hugging Face Hub: {HF_MODEL_REPO}/{CKPT_FILENAME}...")
        ckpt_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=CKPT_FILENAME)
        print(f"Downloaded checkpoint to: {ckpt_path}")

    # 2. Build vocabulary
    vocab = build_vocabulary(include_ptms=True)

    # 3. Instantiate model architecture and load weights
    ckpt = load_checkpoint(ckpt_path, map_location=DEVICE)
    enc, lp, dec, guid = build_models(vocab, DEVICE)
    load_models_from_checkpoint(ckpt, enc, lp, dec, guid, use_ema=True)
    enc.eval()
    lp.eval()
    dec.eval()
    guid.eval()

    CACHED_MODELS = (enc, lp, dec, guid, vocab)
    return CACHED_MODELS


def parse_mgf_stream(stream):
    """Parse spectra from an open text stream in Mascot Generic Format (MGF)."""
    spectra = []
    current_title = ""
    current_pepmass = None
    current_charge = 2
    mz_list = []
    intensity_list = []
    in_ions = False
    scan_idx = 0

    for line in stream:
        line = line.strip()
        if not line:
            continue
        if line == "BEGIN IONS":
            in_ions = True
            current_title = f"scan_{scan_idx + 1}"
            current_pepmass = None
            current_charge = 2
            mz_list = []
            intensity_list = []
            continue
        elif line == "END IONS":
            if in_ions and current_pepmass is not None and len(mz_list) > 0:
                precursor_mass = current_pepmass * current_charge - current_charge * M_H
                spectra.append(
                    {
                        "title": current_title,
                        "precursor_mz": float(current_pepmass),
                        "precursor_mass": float(precursor_mass),
                        "precursor_charge": int(current_charge),
                        "mz_array": np.array(mz_list, dtype=np.float32),
                        "intensity_array": np.array(intensity_list, dtype=np.float32),
                    }
                )
                scan_idx += 1
            in_ions = False
            continue

        if not in_ions:
            continue

        if "=" in line:
            key, val = line.split("=", 1)
            key = key.strip().upper()
            val = val.strip()
            if key == "TITLE":
                current_title = val
            elif key == "PEPMASS":
                current_pepmass = float(val.split()[0])
            elif key == "CHARGE":
                val = val.rstrip("+").rstrip("-")
                try:
                    current_charge = int(val)
                except ValueError:
                    current_charge = 2
        else:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    mz_list.append(float(parts[0]))
                    intensity_list.append(float(parts[1]))
                except ValueError:
                    pass
    return spectra


def load_spectra_file(file_path):
    """Load spectra from a file, automatically detecting gzip compression (.mgz / .mgf.gz)."""
    with open(file_path, "rb") as f:
        magic = f.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            return parse_mgf_stream(f)
    else:
        with open(file_path, "rt", encoding="utf-8", errors="replace") as f:
            return parse_mgf_stream(f)


def run_sequencing(
    spectrum_file,
    max_spectra: int = 50,
    num_steps: int = 20,
    top_k_lengths: int = 5,
    guidance_scale: float = 1.5,
    beta_frag: float = 0.5,
    trypsin_prior: bool = True,
    progress=gr.Progress(track_tqdm=True),
):
    """Gradio handler: parses uploaded .mgz/.mgf, runs de novo sequencing, and returns results table."""
    if spectrum_file is None:
        raise gr.Error("Please upload a .mgz or .mgf file first.")

    progress(0.05, desc="Loading DFlowNovo model weights...")
    enc, lp, dec, guid, vocab = get_model()

    progress(0.15, desc="Parsing mass spectrometry file...")
    file_path = spectrum_file.name if hasattr(spectrum_file, "name") else str(spectrum_file)
    spectra = load_spectra_file(file_path)

    if not spectra:
        raise gr.Error(
            "No valid MS/MS spectra found in the uploaded file. Please ensure the file adheres to standard MGF / MGZ format."
        )

    total_in_file = len(spectra)
    if total_in_file > max_spectra:
        spectra = spectra[:max_spectra]

    n_samples = len(spectra)
    batch_size = 16 if DEVICE.type == "cpu" else 64
    results = []

    progress(0.25, desc=f"Sequencing {n_samples} spectra on {DEVICE.type.upper()}...")

    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_spectra = spectra[start_idx:end_idx]
        cur_b = len(batch_spectra)

        top_k = 200
        mz_tensors = []
        intens_tensors = []
        prec_masses = []
        prec_charges = []
        batch_titles = []
        batch_orig_mzs = []

        for s in batch_spectra:
            mz_arr = torch.tensor(s["mz_array"], dtype=torch.float32)
            intens_arr = torch.tensor(s["intensity_array"], dtype=torch.float32)
            pm = s["precursor_mass"]
            pc = s["precursor_charge"]

            # Remove precursor peak
            keep = (mz_arr - s["precursor_mz"]).abs() > 1.5
            mz_arr = mz_arr[keep]
            intens_arr = intens_arr[keep]

            # Top-k peaks
            k = min(top_k, intens_arr.shape[0])
            if k > 0:
                intens_arr, indices = torch.topk(intens_arr, k)
                mz_arr = mz_arr[indices]

            batch_titles.append(s["title"])
            batch_orig_mzs.append(s["precursor_mz"])
            prec_masses.append(pm)
            prec_charges.append(pc)
            mz_tensors.append(mz_arr)
            intens_tensors.append(intens_arr)

        # Pad to max peaks
        max_p = max(len(m) for m in mz_tensors)
        padded_mz = torch.zeros((cur_b, max_p), dtype=torch.float32, device=DEVICE)
        padded_intens = torch.zeros((cur_b, max_p), dtype=torch.float32, device=DEVICE)
        padded_mask = torch.ones((cur_b, max_p), dtype=torch.bool, device=DEVICE)

        for i in range(cur_b):
            n = len(mz_tensors[i])
            padded_mz[i, :n] = mz_tensors[i].to(DEVICE)
            padded_intens[i, :n] = intens_tensors[i].to(DEVICE)
            padded_mask[i, :n] = False

        t_prec_mass = torch.tensor(prec_masses, dtype=torch.float32, device=DEVICE)
        t_prec_charge = torch.tensor(prec_charges, dtype=torch.long, device=DEVICE)
        t_mz_comp = t_prec_mass.unsqueeze(-1) + 2 * M_H - padded_mz

        with torch.no_grad():
            x_t, lengths, seqs, scores = predict_peptide(
                mz_array=padded_mz,
                intensity_array=padded_intens,
                precursor_mass=t_prec_mass,
                precursor_charge=t_prec_charge,
                mz_complementary=t_mz_comp,
                spectrum_mask=padded_mask,
                vocabulary=vocab,
                spectrum_encoder=enc,
                length_predictor=lp,
                decoder=dec,
                guidance=guid,
                scheduler=cosine_scheduler,
                num_steps=int(num_steps),
                top_k_lengths=int(top_k_lengths),
                guidance_scale=float(guidance_scale),
                beta=float(beta_frag),
                trypsin_prior=bool(trypsin_prior),
                return_scores=True,
            )

        for i in range(cur_b):
            seq = seqs[i]
            tokens = parse_peptide(seq)
            calc_mass = sum(AA_MASSES_DICT.get(t, 0.0) for t in tokens) + M_H2O
            delta_da = calc_mass - prec_masses[i]
            delta_ppm = (delta_da / max(prec_masses[i], 1.0)) * 1e6
            unimod_seq = to_unimod_sequence(seq)

            results.append(
                {
                    "Scan / Title": batch_titles[i],
                    "Predicted Sequence": seq,
                    "UNIMOD Sequence": unimod_seq,
                    "Confidence Score": round(float(scores[i].item()), 3),
                    "Precursor m/z": round(float(batch_orig_mzs[i]), 4),
                    "Precursor Mass (Da)": round(float(prec_masses[i]), 4),
                    "Calculated Mass (Da)": round(float(calc_mass), 4),
                    "Delta Mass (Da)": round(float(delta_da), 4),
                    "Delta Mass (ppm)": round(float(delta_ppm), 1),
                    "Charge": int(prec_charges[i]),
                    "Length": int(lengths[i].item()),
                }
            )

        frac = 0.25 + 0.70 * (end_idx / n_samples)
        progress(frac, desc=f"Sequenced {end_idx}/{n_samples} spectra...")

    df = pd.DataFrame(results)

    # Save CSV and TSV for export
    csv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tsv_file = tempfile.NamedTemporaryFile(delete=False, suffix=".tsv")
    df.to_csv(csv_file.name, index=False)
    df.to_csv(tsv_file.name, sep="\t", index=False)

    # Build KPI metrics summary card
    avg_conf = df["Confidence Score"].mean()
    med_ppm = df["Delta Mass (ppm)"].abs().median()
    tryptic_pct = (
        df["Predicted Sequence"].apply(lambda s: s.endswith("K") or s.endswith("R")).mean() * 100.0
    )
    ptm_count = df["Predicted Sequence"].apply(lambda s: "(" in s or "[" in s).sum()

    summary_md = f"""
### 📊 Sequencing Run Summary
- **Spectra Processed**: **{len(df)}** (out of {total_in_file} in file)
- **Mean Confidence Score**: **{avg_conf:.2f}**
- **Median Absolute Mass Error**: **{med_ppm:.1f} ppm**
- **Peptides with Identified PTMs**: **{ptm_count}** ({ptm_count / len(df):.1%})
- **Tryptic Cleavage Specificity (C-term K/R)**: **{tryptic_pct:.1f}%**
- **Inference Device**: `{DEVICE.type.upper()}`
"""
    return summary_md, df, csv_file.name, tsv_file.name


# Gradio Web UI Layout
custom_css = """
.gradio-container { max-width: 1280px !important; margin: auto; }
.metric-card { background: #f8fafc; border-radius: 8px; padding: 12px; border: 1px solid #e2e8f0; }
"""

with gr.Blocks(title="DFlowNovo: De Novo Peptide Sequencing") as demo:
    gr.Markdown(
        """
        # 🧬 DFlowNovo: Discrete Flow Matching for De Novo Peptide Sequencing
        ### End-to-End De Novo Sequencing with Post-Translational Modification (PTM) & UNIMOD Support
        Upload an **`.mgz`** or **`.mgf`** mass spectrometry file to reconstruct peptide sequences directly from MS/MS spectra.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📂 Input Spectrum File")
            input_file = gr.File(
                label="Upload .mgz, .mgf, or .mgf.gz file",
                file_types=[".mgz", ".mgf", ".gz", ".txt"],
                type="filepath",
            )

            max_spec_slider = gr.Slider(
                label="Maximum Spectra to Sequence",
                minimum=1,
                maximum=250,
                value=25,
                step=1,
                info="Limit the number of spectra processed per submission.",
            )

            with gr.Accordion("⚙️ Advanced Inference Parameters", open=False):
                num_steps_slider = gr.Slider(
                    label="Flow Matching Steps",
                    minimum=5,
                    maximum=50,
                    value=20,
                    step=1,
                    info="Number of discrete unmasking steps (default: 20).",
                )
                top_k_slider = gr.Slider(
                    label="Top-k Length Candidates",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    info="Bayesian length candidates evaluated in parallel (default: 5).",
                )
                guidance_slider = gr.Slider(
                    label="Classifier-Free Guidance Scale",
                    minimum=1.0,
                    maximum=3.0,
                    value=1.5,
                    step=0.1,
                    info="Spectral conditioning guidance multiplier (default: 1.5).",
                )
                beta_slider = gr.Slider(
                    label="Fragment Matching Weight (β)",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.05,
                    info="Weight of theoretical b/y ion matching in sequence scoring.",
                )
                trypsin_checkbox = gr.Checkbox(
                    label="Apply Trypsin Cleavage Prior (C-terminal K/R)",
                    value=True,
                )

            run_btn = gr.Button("🚀 Run De Novo Sequencing", variant="primary", size="lg")

            # Built-in sample file for 1-click testing
            sample_path = ROOT_DIR / "sample_spectra.mgz"
            if sample_path.exists():
                gr.Examples(
                    examples=[[str(sample_path), 5, 20, 5, 1.5, 0.5, True]],
                    inputs=[
                        input_file,
                        max_spec_slider,
                        num_steps_slider,
                        top_k_slider,
                        guidance_slider,
                        beta_slider,
                        trypsin_checkbox,
                    ],
                    label="Clickable Example (.mgz)",
                )

        with gr.Column(scale=2):
            summary_output = gr.Markdown("### 📊 Sequencing Run Summary\n*Submit a file to view run statistics.*")
            table_output = gr.DataFrame(
                label="📋 Predicted Peptide Sequences & Mass Spectrometry Metrics",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                csv_download = gr.File(label="📥 Download Results as CSV", interactive=False)
                tsv_download = gr.File(label="📥 Download Results as TSV", interactive=False)

    run_btn.click(
        fn=run_sequencing,
        inputs=[
            input_file,
            max_spec_slider,
            num_steps_slider,
            top_k_slider,
            guidance_slider,
            beta_slider,
            trypsin_checkbox,
        ],
        outputs=[summary_output, table_output, csv_download, tsv_download],
    )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DFlowNovo Gradio Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--share", action="store_true", help="Create public Gradio share link")
    cli_args = parser.parse_args()

    demo.launch(server_name=cli_args.host, server_port=cli_args.port, share=cli_args.share)
