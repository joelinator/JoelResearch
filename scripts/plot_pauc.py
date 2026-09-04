#!/usr/bin/env python3
"""Script to plot Precision-Coverage curves (PAUC) and score distributions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

scripts_dir = str(Path(__file__).resolve().parent)
while scripts_dir in sys.path:
    sys.path.remove(scripts_dir)

src_dir = str(Path(__file__).resolve().parent.parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import numpy as np
from eval.plots import plot_pauc_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Plot PAUC and precision-coverage curves.")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to evaluation details JSON containing 'scores', 'exact_matches', and 'mass_matches'",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="artifacts/pauc_curve.png",
        help="Destination path for output PNG plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="De Novo Peptide Sequencing Precision-Coverage (PAUC)",
        help="Plot title",
    )
    parser.add_argument("--calibrated-threshold", type=float, default=None)
    parser.add_argument("--calibrated-coverage", type=float, default=None)
    parser.add_argument("--calibrated-precision", type=float, default=None)
    parser.add_argument("--calibrated-recall", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    in_path = Path(args.input_json)
    with in_path.open() as f:
        data = json.load(f)

    scores = np.asarray(data["scores"], dtype=np.float64)
    exact_matches = np.asarray(data["exact_matches"], dtype=bool)
    mass_matches = np.asarray(data["mass_matches"], dtype=bool)

    plot_pauc_curve(
        scores=scores,
        exact_matches=exact_matches,
        mass_matches=mass_matches,
        output_path=args.output_plot,
        title=args.title,
        calibrated_threshold=args.calibrated_threshold,
        calibrated_coverage=args.calibrated_coverage,
        calibrated_precision=args.calibrated_precision,
        calibrated_recall=args.calibrated_recall,
    )


if __name__ == "__main__":
    main()
