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

from lensing_pipeline.detection import detect_arc_mask
from lensing_pipeline.ellipse import extract_arc_parameters, fit_ellipse_from_mask
from lensing_pipeline.metrics import segmentation_metrics
from lensing_pipeline.visualization import save_detection_overlay, save_grayscale_png


def load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final classical gravitational arc detector on one image.")
    parser.add_argument("--input", type=Path, default=ROOT / "examples" / "lens_00932_observed.png")
    parser.add_argument("--truth-mask", type=Path, default=ROOT / "examples" / "lens_00932_true_mask.png")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "arc_detection_demo")
    parser.add_argument("--threshold-percentile", type=float, default=91.2)
    parser.add_argument("--min-component-size", type=int, default=8)
    parser.add_argument("--suppress-centre-radius", type=float, default=0.08)
    parser.add_argument("--log-sigma", type=float, default=1.4)
    parser.add_argument("--log-percentile", type=float, default=98.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    image = load_gray(args.input)
    truth = load_gray(args.truth_mask) > 0.5 if args.truth_mask and args.truth_mask.exists() else None
    mask = detect_arc_mask(
        image,
        threshold_percentile=args.threshold_percentile,
        min_component_size=args.min_component_size,
        suppress_centre_radius=args.suppress_centre_radius,
        log_sigma=args.log_sigma,
        log_percentile=args.log_percentile,
    )
    ellipse = fit_ellipse_from_mask(mask)
    params = extract_arc_parameters(mask, image.shape)
    save_grayscale_png(mask.astype(float), args.out_dir / "detected_arc_mask.png")
    save_detection_overlay(image, mask, truth, ellipse, args.out_dir / "detection_overlay.png")
    row = {
        "input": str(args.input),
        "arc_pixels": int(mask.sum()),
        "centroid_x": params.centroid_x,
        "centroid_y": params.centroid_y,
        "semi_major": params.semi_major,
        "semi_minor": params.semi_minor,
        "axis_ratio": params.axis_ratio,
        "orientation_degrees": params.orientation_degrees,
        "einstein_radius_estimate_pixels": params.einstein_radius_estimate,
    }
    if truth is not None:
        row.update(segmentation_metrics(mask, truth))
    with (args.out_dir / "arc_detection_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved detection outputs to {args.out_dir}")
    print(row)


if __name__ == "__main__":
    main()
