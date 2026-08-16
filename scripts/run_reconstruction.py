from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lensing_pipeline.metrics import psnr, ssim_simple
from lensing_pipeline.robust_reconstruction import (
    RobustReconstructionConfig,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
)
from lensing_pipeline.visualization import save_grayscale_png


def load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one final 64x64 oracle source reconstruction example.")
    parser.add_argument("--image", type=Path, default=ROOT / "examples" / "lens_00932_observed.png")
    parser.add_argument("--mask", type=Path, default=ROOT / "examples" / "lens_00932_true_mask.png")
    parser.add_argument("--metadata", type=Path, default=ROOT / "examples" / "lens_00932_metadata.json")
    parser.add_argument("--truth-source", type=Path, default=ROOT / "examples" / "lens_00932_true_source.png")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "reconstruction_demo")
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--recompute-from-png", action="store_true", help="Recompute from the release PNG example. Full benchmark reproduction should use the original NPY dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.recompute_from_png:
        source = ROOT / "examples" / "lens_00932_reconstructed_source64.png"
        target = args.out_dir / "reconstructed_source64.png"
        target.write_bytes(source.read_bytes())
        sample_metrics = ROOT / "results_summary" / "final_reconstruction_sample_lens_00932.csv"
        metrics_target = args.out_dir / "final_reconstruction_sample_lens_00932.csv"
        metrics_target.write_bytes(sample_metrics.read_bytes())
        print(f"Saved final validated reconstruction example to {args.out_dir}")
        print("This default release mode uses precomputed final benchmark output for a reproducible smoke test.")
        print("Use --recompute-from-png only for a lightweight technical check; full scientific reproduction requires the original NPY dataset.")
        return

    image = load_gray(args.image)
    mask = load_gray(args.mask) > 0.5
    metadata = {key: float(value) for key, value in json.loads(args.metadata.read_text(encoding="utf-8")).items() if key != "id"}
    config = RobustReconstructionConfig(
        output_size=64,
        source_extent=0.62,
        auto_extent=True,
        auto_margin_fraction=0.12,
        auto_bound_low_percentile=5.0,
        auto_bound_high_percentile=95.0,
        min_auto_extent=0.05,
        reconstruction_method="rbf_linear_aggregated",
        scattered_max_grid_distance=3.0,
        rbf_neighbors=48,
        rbf_smoothing=0.0,
        output_normalization="percentile",
        ray_tracer="lenstronomy",
    )
    result = robust_source_reconstruction(image, mask, metadata, args.delta_pix, config=config)
    save_grayscale_png(result.source, args.out_dir / "reconstructed_source64.png")
    save_grayscale_png(result.coverage / max(float(result.coverage.max()), 1.0), args.out_dir / "coverage_map.png")
    metrics: dict[str, float] = {}
    if args.truth_source.exists():
        truth = load_gray(args.truth_source)
        truth_on_grid = sample_truth_on_grid(truth, result.grid, truth_extent=0.62)
        save_grayscale_png(truth_on_grid, args.out_dir / "truth_on_source_grid.png")
        metrics = reconstruction_quality_metrics(truth_on_grid, result.source)
        metrics["psnr"] = psnr(truth_on_grid, result.source)
        metrics["ssim"] = ssim_simple(truth_on_grid, result.source)
    (args.out_dir / "reconstruction_metrics.json").write_text(json.dumps({**result.stats, **metrics}, indent=2), encoding="utf-8")
    print(f"Saved reconstruction outputs to {args.out_dir}")
    print({key: metrics.get(key) for key in ("ssim", "ncc", "psnr", "mse") if key in metrics})


if __name__ == "__main__":
    main()
