from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lensing_pipeline.morphology_models import morphology_features


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract morphology features for one reconstructed source and show final classifier summary.")
    parser.add_argument("--source", type=Path, default=ROOT / "examples" / "lens_00932_reconstructed_source64.png")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "classification_demo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    features = morphology_features(load_gray(args.source))
    with (args.out_dir / "morphology_features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(features.keys()))
        writer.writeheader()
        writer.writerow(features)
    summary_path = ROOT / "results_summary" / "final_classification_summary.csv"
    print(f"Saved feature extraction output to {args.out_dir}")
    print("Final classifier validation summary is stored at:")
    print(summary_path)


if __name__ == "__main__":
    main()
