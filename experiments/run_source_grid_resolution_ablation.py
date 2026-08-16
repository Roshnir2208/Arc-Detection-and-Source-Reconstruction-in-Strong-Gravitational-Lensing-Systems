from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from lensing_pipeline.metrics import match_photometric_scale
from lensing_pipeline.robust_reconstruction import (
    RobustReconstructionConfig,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
)
from run_cosmos_lensing_benchmark import augment_source, generate_case, make_compact_source, stretch
from run_reconstruction_improvement_ablation import forward_lens_reconstruction, morphology_errors
from run_sparse_coverage_targeted_fixes import extract_samples, support_fractions
from run_support_tv_reconstruction_validation import artefact_metrics, gaussian_psf_kernel, richardson_lucy


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 0 else 0.0


def forward_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    pred = match_photometric_scale(observed, predicted)
    residual = np.asarray(observed, dtype=float) - pred
    return {
        "forward_mse": float(np.mean(residual * residual)),
        "forward_ncc": ncc(observed, pred),
    }


def config_for(args: argparse.Namespace, source_size: int) -> RobustReconstructionConfig:
    return RobustReconstructionConfig(
        output_size=int(source_size),
        source_extent=args.source_extent,
        auto_extent=True,
        auto_margin_fraction=0.12,
        auto_bound_low_percentile=5.0,
        auto_bound_high_percentile=95.0,
        min_auto_extent=0.05,
        reconstruction_method="rbf_linear_aggregated",
        scattered_max_grid_distance=args.scattered_max_grid_distance,
        beta_aggregation_sigma_clip=2.5,
        beta_aggregation_min_bin_samples=2,
        rbf_smoothing=0.0,
        rbf_neighbors=args.rbf_neighbors,
        hole_fill="none",
        output_normalization="percentile",
        regularization="none",
    )


def method_row(
    case_id: str,
    resolution: int,
    result,
    truth: np.ndarray,
    observed: np.ndarray,
    metadata: dict[str, float | str],
    delta_pix: float,
    psf: np.ndarray,
    runtime: float,
    sample_count: int,
    fractions: dict[str, object],
) -> dict[str, object]:
    support = fractions.get("support", result.coverage > 0)
    predicted = forward_lens_reconstruction(result.source, result.grid, metadata, observed.shape, delta_pix, psf)
    native_pixels = int(resolution) * int(resolution)
    direct_supported_fraction = finite_float(fractions.get("direct_supported_fraction"), 0.0)
    row: dict[str, object] = {
        "id": case_id,
        "resolution": int(resolution),
        "method": f"rl15_rbf_linear_{resolution}",
        "runtime_seconds": runtime,
        "mapped_beta_sample_count": int(sample_count),
        "native_source_pixel_count": int(native_pixels),
        "samples_per_native_pixel": float(sample_count / max(native_pixels, 1)),
        "samples_per_directly_supported_pixel": float(sample_count / max(native_pixels * direct_supported_fraction, 1.0)),
        **reconstruction_quality_metrics(truth, result.source),
        **morphology_errors(truth, result.source),
        **forward_metrics(observed, predicted),
        **artefact_metrics(result.source, support),
        **{k: v for k, v in fractions.items() if not isinstance(v, np.ndarray)},
    }
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["resolution"])].append(row)
    metrics = [
        "ssim_aligned",
        "ncc_aligned",
        "psnr_aligned",
        "mse_aligned",
        "centroid_error",
        "effective_radius_error",
        "ellipticity_error",
        "asymmetry_error",
        "gini_error",
        "concentration_error",
        "forward_mse",
        "forward_ncc",
        "runtime_seconds",
        "samples_per_native_pixel",
        "samples_per_directly_supported_pixel",
        "direct_supported_fraction",
        "interpolated_fraction",
        "extrapolated_fraction",
        "artefact_high_frequency_energy",
        "artefact_dark_hole_fraction",
        "artefact_component_count",
    ]
    out: list[dict[str, object]] = []
    for resolution, resolution_rows in sorted(grouped.items()):
        item: dict[str, object] = {"resolution": resolution, "method": f"rl15_rbf_linear_{resolution}", "count": len(resolution_rows)}
        for metric in metrics:
            values = np.array([finite_float(row.get(metric)) for row in resolution_rows], dtype=float)
            values = values[np.isfinite(values)]
            item[f"mean_{metric}"] = float(np.mean(values)) if len(values) else math.nan
            item[f"median_{metric}"] = float(np.median(values)) if len(values) else math.nan
            item[f"std_{metric}"] = float(np.std(values)) if len(values) else math.nan
        out.append(item)
    return out


def select_resolution(summary: list[dict[str, object]]) -> dict[str, object]:
    if not summary:
        return {"selected_resolution": "", "reason": "no_results"}
    max_ssim = max(finite_float(row.get("median_ssim_aligned")) for row in summary)
    max_ncc = max(finite_float(row.get("median_ncc_aligned")) for row in summary)
    candidates = []
    for row in summary:
        ssim = finite_float(row.get("median_ssim_aligned"))
        ncc_value = finite_float(row.get("median_ncc_aligned"))
        direct = finite_float(row.get("median_direct_supported_fraction"))
        artefact = finite_float(row.get("median_artefact_high_frequency_energy"))
        morphology = finite_float(row.get("median_centroid_error")) + finite_float(row.get("median_effective_radius_error")) + finite_float(row.get("median_ellipticity_error"))
        close_to_best = (ssim >= max_ssim - 0.01) and (ncc_value >= max_ncc - 0.01)
        score = (1.5 * ssim) + ncc_value + (0.25 * direct) - (0.25 * artefact) - (0.01 * morphology)
        candidates.append((close_to_best, score, -int(row["resolution"]), row))
    candidates.sort(reverse=True)
    selected = candidates[0][3]
    return {
        "selected_resolution": int(selected["resolution"]),
        "selected_method": selected["method"],
        "reason": "lowest/highest balanced resolution preserving near-best SSIM/NCC with morphology and artefact checks",
        "median_ssim_aligned": selected.get("median_ssim_aligned"),
        "median_ncc_aligned": selected.get("median_ncc_aligned"),
        "median_samples_per_native_pixel": selected.get("median_samples_per_native_pixel"),
        "median_direct_supported_fraction": selected.get("median_direct_supported_fraction"),
    }


def display_native(image: np.ndarray, tile: int = 132) -> Image.Image:
    arr = stretch(image)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)


def save_resolution_panel(path: Path, case_id: str, truth_by_res: dict[int, np.ndarray], source_by_res: dict[int, np.ndarray], resolutions: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tile = 132
    label_h = 24
    footer_h = 22
    panels: list[tuple[str, np.ndarray]] = [("true source", truth_by_res[max(resolutions)])]
    for resolution in resolutions:
        panels.append((f"{resolution}x{resolution}", source_by_res[resolution]))
    canvas = Image.new("RGB", (tile * len(panels), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(panels):
        x = idx * tile
        draw.text((x + 4, 4), label, fill="black")
        canvas.paste(display_native(arr, tile=tile), (x, label_h))
    draw.text((4, tile + label_h + 4), f"{case_id} | display-only upsampling for visual comparison", fill="black")
    canvas.save(path)


def save_galleries(out_dir: Path, rows: list[dict[str, object]], records: dict[str, dict[str, object]], selected_resolution: int) -> None:
    selected = [row for row in rows if int(row["resolution"]) == int(selected_resolution)]
    selected.sort(key=lambda row: finite_float(row.get("ssim_aligned")), reverse=True)
    groups = {
        "best_examples": selected[:5],
        "worst_examples": selected[-5:][::-1],
    }
    mid = len(selected) // 2
    groups["median_examples"] = selected[max(0, mid - 2) : min(len(selected), mid + 3)]
    for folder, group in groups.items():
        for idx, row in enumerate(group, start=1):
            rec = records[str(row["id"])]
            save_resolution_panel(
                out_dir / folder / f"{idx:02d}_{row['id']}_resolution_ablation.png",
                str(row["id"]),
                rec["truth_by_res"],
                rec["source_by_res"],
                rec["resolutions"],
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate native source-grid resolution for RL15 + RBF linear reconstruction.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "cosmos_sources")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "source_grid_resolution_ablation")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--resolutions", type=str, default="32,48,64,96,128")
    parser.add_argument("--seed", type=int, default=25015168)
    parser.add_argument("--num-pix", type=int, default=128)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--source-extent", type=float, default=0.62)
    parser.add_argument("--noise-sigma", type=float, default=0.018)
    parser.add_argument("--psf-fwhm", type=float, default=0.08)
    parser.add_argument("--rl-iterations", type=int, default=15)
    parser.add_argument("--rbf-neighbors", type=int, default=48)
    parser.add_argument("--scattered-max-grid-distance", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resolutions = [int(item.strip()) for item in args.resolutions.split(",") if item.strip()]
    source_paths = sorted((args.source_dir / "sources_npy").glob("*.npy"))
    if not source_paths:
        raise SystemExit(f"No source .npy files found in {args.source_dir / 'sources_npy'}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    psf = gaussian_psf_kernel(args.psf_fwhm, args.delta_pix)
    rows: list[dict[str, object]] = []
    records: dict[str, dict[str, object]] = {}

    for index, source_path in enumerate(source_paths[: args.count]):
        source = np.load(source_path).astype(float)
        source = make_compact_source(augment_source(source, rng), rng, max(resolutions), 0.34, 0.58)
        case_id = f"grid_ablation_{index:05d}"
        observed, _clean, _lensed_source, true_mask, metadata = generate_case(
            source,
            source_path.stem,
            case_id,
            rng,
            args.num_pix,
            args.delta_pix,
            args.source_extent,
            args.noise_sigma,
        )
        deconvolved = richardson_lucy(observed, psf, args.rl_iterations)
        source_by_res: dict[int, np.ndarray] = {}
        truth_by_res: dict[int, np.ndarray] = {}
        for resolution in resolutions:
            start = time.perf_counter()
            result = robust_source_reconstruction(
                deconvolved,
                true_mask,
                metadata,  # type: ignore[arg-type]
                args.delta_pix,
                valid_pixel_mode="arc",
                config=config_for(args, resolution),
            )
            runtime = time.perf_counter() - start
            truth = sample_truth_on_grid(source, result.grid, args.source_extent)
            samples = extract_samples(result, deconvolved)
            fractions = support_fractions(
                samples["points"],
                samples["direct_coverage"],
                samples["target_points"],
                float(samples["pixel_scale"]),
                args.scattered_max_grid_distance,
            )
            rows.append(
                method_row(
                    case_id,
                    resolution,
                    result,
                    truth,
                    observed,
                    metadata,
                    args.delta_pix,
                    psf,
                    runtime,
                    int(len(samples["points"])),
                    fractions,
                )
            )
            source_by_res[resolution] = result.source
            truth_by_res[resolution] = truth
        records[case_id] = {"source_by_res": source_by_res, "truth_by_res": truth_by_res, "resolutions": resolutions}
        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{min(args.count, len(source_paths))}")

    summary = summarize(rows)
    decision = select_resolution(summary)
    write_csv(args.out_dir / "per_image_metrics.csv", rows)
    write_csv(args.out_dir / "resolution_summary.csv", summary)
    (args.out_dir / "selected_resolution.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    save_galleries(args.out_dir, rows, records, int(decision["selected_resolution"]))

    plt.figure(figsize=(8, 4.8), dpi=180)
    plt.plot([row["resolution"] for row in summary], [finite_float(row.get("median_ssim_aligned")) for row in summary], marker="o", label="SSIM")
    plt.plot([row["resolution"] for row in summary], [finite_float(row.get("median_ncc_aligned")) for row in summary], marker="o", label="NCC")
    plt.xlabel("Native source-grid resolution")
    plt.ylabel("Median metric")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "resolution_vs_ssim_ncc.png")
    plt.close()

    plt.figure(figsize=(8, 4.8), dpi=180)
    plt.plot([row["resolution"] for row in summary], [finite_float(row.get("median_samples_per_native_pixel")) for row in summary], marker="o", label="samples/native pixel")
    plt.plot([row["resolution"] for row in summary], [finite_float(row.get("median_direct_supported_fraction")) for row in summary], marker="o", label="direct support fraction")
    plt.xlabel("Native source-grid resolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out_dir / "resolution_vs_sampling_density.png")
    plt.close()

    lines = [
        "# Source-Grid Resolution Ablation",
        "",
        "Method held fixed: `RL15 + RBF linear`.",
        "",
        f"Selected native resolution: `{decision['selected_resolution']}x{decision['selected_resolution']}`",
        "",
        f"Reason: {decision['reason']}",
        "",
        "Scientific metrics were calculated on each native grid after resampling the true source onto the same physical source-plane grid. Display panels may upsample native images only for viewing.",
        "",
        "No detection, lens parameters, RL settings, RBF method, classifier code, or support-mask logic was changed.",
    ]
    (args.out_dir / "SOURCE_GRID_RESOLUTION_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved source-grid resolution ablation to {args.out_dir}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
