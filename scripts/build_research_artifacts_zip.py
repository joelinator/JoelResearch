#!/usr/bin/env python3
"""
Create a self-contained, publication-grade zip archive of all research artifacts
(prediction CSVs, evaluation metric JSONs, ground truths, training logs, vocabularies)
optimized for future analysis and plotting without requiring re-inference or re-training.
"""

import os
import glob
import zipfile
import pandas as pd
from pathlib import Path

OUTPUT_ZIP = Path("dfm_research_analysis_artifacts.zip")
SCRATCH_DIR = Path("scratch/bundle_staging")
SCRATCH_DIR.mkdir(parents=True, exist_ok=True)

print("1. Optimizing InstaNovo Full Test predictions for compact storage...")
# Process Nine-Species InstaNovo
in_ns_path = Path("artifacts/instanovo_eval/ninespecies_full_test_preds.csv")
if in_ns_path.exists():
    df_ns = pd.read_csv(in_ns_path)
    drop_cols = [c for c in df_ns.columns if "token_log_prob" in c or "token_probabilities" in c or c in ["predictions_tokenised", "group", "prediction_id"]]
    df_ns = df_ns.drop(columns=[c for c in drop_cols if c in df_ns.columns])
    for c in [c for c in df_ns.columns if "prob" in c or "mz" in c or "ppm" in c]:
        df_ns[c] = df_ns[c].round(4)
    out_ns = SCRATCH_DIR / "instanovo_ninespecies_full_test_preds.csv"
    df_ns.to_csv(out_ns, index=False)
    print(f"   Nine-Species InstaNovo: {in_ns_path.stat().st_size / (1024*1024):.1f} MB -> {out_ns.stat().st_size / (1024*1024):.1f} MB")

# Process HC-PT InstaNovo
in_hc_path = Path("artifacts/instanovo_eval/hcpt_full_test_preds.csv")
if in_hc_path.exists():
    df_hc = pd.read_csv(in_hc_path)
    drop_cols = [c for c in df_hc.columns if "token_log_prob" in c or "token_probabilities" in c or c in ["predictions_tokenised", "group", "prediction_id"]]
    df_hc = df_hc.drop(columns=[c for c in drop_cols if c in df_hc.columns])
    for c in [c for c in df_hc.columns if "prob" in c or "mz" in c or "ppm" in c]:
        df_hc[c] = df_hc[c].round(4)
    out_hc = SCRATCH_DIR / "instanovo_hcpt_full_test_preds.csv"
    df_hc.to_csv(out_hc, index=False)
    print(f"   HC-PT InstaNovo: {in_hc_path.stat().st_size / (1024*1024):.1f} MB -> {out_hc.stat().st_size / (1024*1024):.1f} MB")

print("\n2. Assembling zip archive...")
files_added = []
with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
    # A. All JSON files in artifacts/
    for json_file in sorted(glob.glob("artifacts/**/*.json", recursive=True)):
        arcname = json_file
        zf.write(json_file, arcname=arcname)
        files_added.append((arcname, os.path.getsize(json_file)))

    # B. All DFM prediction CSVs (Joint Balanced, Strategy A Base & Finetuned)
    dfm_csvs = (
        sorted(glob.glob("artifacts/eval_joint_balanced/*.csv")) +
        sorted(glob.glob("artifacts/eval_strategy_a/*.csv"))
    )
    for csv_file in dfm_csvs:
        arcname = csv_file
        zf.write(csv_file, arcname=arcname)
        files_added.append((arcname, os.path.getsize(csv_file)))

    # C. Training metric logs
    training_csvs = sorted(glob.glob("artifacts/**/version_*/metrics.csv", recursive=True))
    for csv_file in training_csvs:
        arcname = csv_file
        zf.write(csv_file, arcname=arcname)
        files_added.append((arcname, os.path.getsize(csv_file)))

    # D. InstaNovo test predictions
    if (SCRATCH_DIR / "instanovo_ninespecies_full_test_preds.csv").exists():
        arcname = "artifacts/instanovo_eval/ninespecies_full_test_preds.csv"
        src = SCRATCH_DIR / "instanovo_ninespecies_full_test_preds.csv"
        zf.write(src, arcname=arcname)
        files_added.append((arcname, src.stat().st_size))

    if (SCRATCH_DIR / "instanovo_hcpt_full_test_preds.csv").exists():
        arcname = "artifacts/instanovo_eval/hcpt_full_test_preds.csv"
        src = SCRATCH_DIR / "instanovo_hcpt_full_test_preds.csv"
        zf.write(src, arcname=arcname)
        files_added.append((arcname, src.stat().st_size))

total_uncompressed = sum(size for _, size in files_added)
zip_size = OUTPUT_ZIP.stat().st_size

print(f"\n3. Zip creation complete!")
print(f"   Output file: {OUTPUT_ZIP.resolve()}")
print(f"   Total files archived: {len(files_added)}")
print(f"   Total uncompressed data: {total_uncompressed / (1024*1024):.2f} MB")
print(f"   Final compressed zip size: {zip_size / (1024*1024):.2f} MB")

# Verify integrity
print("\n4. Verifying archive integrity...")
with zipfile.ZipFile(OUTPUT_ZIP, "r") as zf:
    bad_file = zf.testzip()
    if bad_file:
        print(f"   ERROR: Corrupted file in archive: {bad_file}")
    else:
        print(f"   SUCCESS: All {len(zf.namelist())} files verified with zero checksum errors.")
