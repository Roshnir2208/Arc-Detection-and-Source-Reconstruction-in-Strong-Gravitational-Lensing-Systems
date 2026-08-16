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
from scipy import ndimage, signal

from lensing_pipeline.detection import detect_arc_mask, normalize_image
from lensing_pipeline.metrics import match_photometric_scale
from lensing_pipeline.robust_reconstruction import (
    RobustReconstructionConfig,
    RobustReconstructionResult,
    SupportCleanupConfig,
    TVRegularizationConfig,
    apply_support_mask,
    clean_source_support_mask,
    morphology_stats,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
    tv_regularize_supported_source,
)
from run_cosmos_lensing_benchmark import augment_source, generate_case, make_compact_source, stretch
from run_reconstruction_improvement_ablation import forward_lens_reconstruction, morphology_errors


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def finite_float(value: object, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def gaussian_psf_kernel(fwhm_arcsec: float, delta_pix: float, truncation: float = 3.0) -> np.ndarray:
    sigma_pix = max(float(fwhm_arcsec) / max(float(delta_pix), 1e-12) / 2.354820045, 0.05)
    radius = max(2, int(math.ceil(truncation * sigma_pix)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma_pix * sigma_pix))
    kernel /= max(float(kernel.sum()), 1e-12)
    return kernel


def richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int) -> np.ndarray:
    observed = np.clip(np.asarray(image, dtype=float), 0.0, None)
    if iterations <= 0:
        return observed
    estimate = np.full_like(observed, max(float(np.mean(observed)), 1e-6), dtype=float)
    psf_mirror = psf[::-1, ::-1]
    initial_flux = float(observed.sum())
    for _ in range(int(iterations)):
        conv = signal.fftconvolve(estimate, psf, mode="same")
        estimate *= signal.fftconvolve(observed / np.maximum(conv, 1e-8), psf_mirror, mode="same")
        estimate = np.clip(estimate, 0.0, None)
    final_flux = float(estimate.sum())
    if final_flux > 0 and initial_flux > 0:
        estimate *= initial_flux / final_flux
    return np.clip(estimate, 0.0, None)


def config_for(args: argparse.Namespace) -> RobustReconstructionConfig:
    return RobustReconstructionConfig(
        output_size=args.source_size,
        source_extent=args.source_extent,
        auto_extent=True,
        auto_margin_fraction=0.12,
        auto_bound_low_percentile=5.0,
        auto_bound_high_percentile=95.0,
        min_auto_extent=0.05,
        reconstruction_method=args.baseline_method,
        scattered_max_grid_distance=args.scattered_max_grid_distance,
        beta_aggregation_sigma_clip=2.5,
        beta_aggregation_min_bin_samples=2,
        rbf_smoothing=args.rbf_smoothing,
        rbf_neighbors=args.rbf_neighbors,
        hole_fill="none",
        output_normalization="percentile",
        regularization="none",
    )


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


def artefact_metrics(source: np.ndarray, support: np.ndarray) -> dict[str, float]:
    source = np.asarray(source, dtype=float)
    support = np.asarray(support, dtype=bool)
    active = (source > max(float(np.percentile(source[source > 0], 45.0)), 0.02)) if np.any(source > 0) else np.zeros_like(support)
    labels, count = ndimage.label(active & support)
    holes = support & (source <= 1e-8)
    hole_labels, hole_count = ndimage.label(holes)
    gradient = np.hypot(ndimage.sobel(source, axis=0), ndimage.sobel(source, axis=1))
    boundary = ndimage.binary_dilation(support) ^ ndimage.binary_erosion(support)
    return {
        "artefact_component_count": float(count),
        "artefact_dark_hole_fraction": float(np.mean(holes[support])) if np.any(support) else 0.0,
        "artefact_dark_hole_components": float(hole_count),
        "artefact_boundary_gradient_mean": float(np.mean(gradient[boundary])) if np.any(boundary) else 0.0,
        "artefact_high_frequency_energy": float(np.mean(np.abs(source - ndimage.median_filter(source, size=3))[support])) if np.any(support) else 0.0,
    }


def make_result(base: RobustReconstructionResult, source: np.ndarray, coverage: np.ndarray, method: str, stats: dict[str, float]) -> RobustReconstructionResult:
    return RobustReconstructionResult(
        source=np.clip(source, 0.0, None),
        coverage=coverage,
        valid_mask=base.valid_mask,
        beta_x=base.beta_x,
        beta_y=base.beta_y,
        grid=base.grid,
        stats={**base.stats, **stats, "reconstruction_method": method},
    )


def metric_row(
    case_id: str,
    method: str,
    result: RobustReconstructionResult,
    truth: np.ndarray,
    observed: np.ndarray,
    metadata: dict[str, float | str],
    delta_pix: float,
    psf: np.ndarray,
    runtime: float,
    support: np.ndarray,
) -> dict[str, object]:
    predicted = forward_lens_reconstruction(result.source, result.grid, metadata, observed.shape, delta_pix, psf)
    quality = reconstruction_quality_metrics(truth, result.source)
    row: dict[str, object] = {
        "id": case_id,
        "method": method,
        "runtime_seconds": runtime,
        **quality,
        **morphology_errors(truth, result.source),
        **forward_metrics(observed, predicted),
        **artefact_metrics(result.source, support),
    }
    row.update({f"stat_{key}": value for key, value in result.stats.items()})
    return row


def save_gray(path: Path, image: np.ndarray, percentile: float = 99.5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((stretch(image, percentile=percentile) * 255).astype(np.uint8), mode="L").save(path)


def save_diagnostic_maps(out_dir: Path, case_id: str, maps: dict[str, np.ndarray], tv_gradient: np.ndarray | None) -> None:
    folder = out_dir / "diagnostic_maps" / case_id
    for name, image in maps.items():
        arr = np.asarray(image, dtype=float)
        if np.any(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=np.nanmax(arr[np.isfinite(arr)]) if np.any(np.isfinite(arr)) else 0.0)
        save_gray(folder / f"{name}.png", arr)
    if tv_gradient is not None:
        save_gray(folder / "tv_gradient_magnitude.png", tv_gradient)


def panel(path: Path, case_id: str, observed: np.ndarray, baseline: np.ndarray, improved: np.ndarray, truth: np.ndarray, support: np.ndarray, method: str, metadata: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_scale = max(float(np.max(truth)), float(np.max(baseline)), float(np.max(improved)), 1e-10)
    residual = np.clip(truth / source_scale, 0.0, 1.0) - np.clip(improved / source_scale, 0.0, 1.0)
    panels = [
        ("observed lens", stretch(observed)),
        ("RL15+RBF baseline", np.clip(baseline / source_scale, 0.0, 1.0)),
        ("improved source", np.clip(improved / source_scale, 0.0, 1.0)),
        ("true source", np.clip(truth / source_scale, 0.0, 1.0)),
        ("source residual", np.clip(0.5 + 0.5 * residual / max(float(np.max(np.abs(residual))), 1e-10), 0.0, 1.0)),
        ("support map", support.astype(float)),
    ]
    tile = 148
    label_h = 24
    footer_h = 26
    canvas = Image.new("RGB", (tile * len(panels), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(panels):
        x = idx * tile
        draw.text((x + 4, 4), label[:22], fill="black")
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")
        canvas.paste(img.resize((tile, tile), Image.Resampling.BICUBIC), (x, label_h))
    draw.text((4, tile + label_h + 4), f"{case_id} | {method} | {metadata}"[:145], fill="black")
    canvas.save(path)


def summarise(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
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
        "artefact_dark_hole_fraction",
        "artefact_component_count",
        "artefact_high_frequency_energy",
        "runtime_seconds",
    ]
    out: list[dict[str, object]] = []
    for method, method_rows in grouped.items():
        item: dict[str, object] = {"method": method, "count": len(method_rows)}
        for metric in metrics:
            values = np.array([finite_float(r.get(metric)) for r in method_rows], dtype=float)
            values = values[np.isfinite(values)]
            item[f"mean_{metric}"] = float(np.mean(values)) if len(values) else math.nan
            item[f"median_{metric}"] = float(np.median(values)) if len(values) else math.nan
            item[f"std_{metric}"] = float(np.std(values)) if len(values) else math.nan
        out.append(item)
    out.sort(key=lambda row: (finite_float(row.get("median_ssim_aligned")), finite_float(row.get("median_ncc_aligned"))), reverse=True)
    return out


def select_method(summary: list[dict[str, object]], baseline: str) -> dict[str, object]:
    baseline_row = next((row for row in summary if row["method"] == baseline), None)
    if baseline_row is None:
        return {"selected_method": summary[0]["method"] if summary else "", "reason": "baseline_missing"}
    candidates = []
    for row in summary:
        method = str(row["method"])
        if method == baseline:
            continue
        ssim_delta = finite_float(row.get("median_ssim_aligned")) - finite_float(baseline_row.get("median_ssim_aligned"))
        ncc_delta = finite_float(row.get("median_ncc_aligned")) - finite_float(baseline_row.get("median_ncc_aligned"))
        centroid_delta = finite_float(row.get("median_centroid_error")) - finite_float(baseline_row.get("median_centroid_error"))
        size_delta = finite_float(row.get("median_effective_radius_error")) - finite_float(baseline_row.get("median_effective_radius_error"))
        artefact_delta = finite_float(row.get("median_artefact_high_frequency_energy")) - finite_float(baseline_row.get("median_artefact_high_frequency_energy"))
        acceptable = (ssim_delta >= -0.003) and (ncc_delta >= -0.003) and (centroid_delta <= 2.0) and (size_delta <= 2.0)
        candidates.append((acceptable, ssim_delta + ncc_delta - max(artefact_delta, 0.0), row, ssim_delta, ncc_delta))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if candidates and candidates[0][0]:
        row = candidates[0][2]
        return {
            "selected_method": row["method"],
            "baseline_method": baseline,
            "reason": "best acceptable support/TV method by median SSIM/NCC without material morphology degradation",
            "median_ssim_delta_vs_baseline": candidates[0][3],
            "median_ncc_delta_vs_baseline": candidates[0][4],
        }
    return {
        "selected_method": baseline,
        "baseline_method": baseline,
        "reason": "support/TV did not beat baseline under the multi-objective acceptance rule",
    }


def save_ranked_examples(out_dir: Path, rows: list[dict[str, object]], selected_method: str, records: dict[tuple[str, str], dict[str, object]]) -> None:
    selected = [row for row in rows if row["method"] == selected_method]
    selected.sort(key=lambda row: finite_float(row.get("ssim_aligned")), reverse=True)
    groups = {
        "best_examples": selected[:5],
        "worst_examples": selected[-5:][::-1],
    }
    mid = len(selected) // 2
    groups["median_examples"] = selected[max(0, mid - 2) : min(len(selected), mid + 3)]
    for folder, group_rows in groups.items():
        for index, row in enumerate(group_rows, start=1):
            record = records[(str(row["id"]), str(row["method"]))]
            panel(
                out_dir / folder / f"{index:02d}_{row['id']}_{row['method']}.png",
                str(row["id"]),
                record["observed"],
                record["baseline_source"],
                record["source"],
                record["truth"],
                record["support"],
                str(row["method"]),
                f"SSIM={finite_float(row.get('ssim_aligned')):.3f}, NCC={finite_float(row.get('ncc_aligned')):.3f}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate support-mask cleanup and TV regularisation for source reconstruction.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "cosmos_sources")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "reconstruction_final_regularised")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=25015168)
    parser.add_argument("--source-size", type=int, default=128)
    parser.add_argument("--num-pix", type=int, default=128)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--source-extent", type=float, default=0.62)
    parser.add_argument("--noise-sigma", type=float, default=0.018)
    parser.add_argument("--psf-fwhm", type=float, default=0.08)
    parser.add_argument("--rl-iterations", type=int, default=15)
    parser.add_argument("--baseline-method", type=str, default="rbf_thin_plate_spline_aggregated")
    parser.add_argument("--rbf-smoothing", type=float, default=0.0)
    parser.add_argument("--rbf-neighbors", type=int, default=48)
    parser.add_argument("--scattered-max-grid-distance", type=float, default=3.0)
    parser.add_argument("--lambda-tv", type=str, default="0.001,0.003,0.01,0.03,0.1")
    parser.add_argument("--save-images", type=int, default=12)
    parser.add_argument("--mask-source", choices=["true", "detected"], default="true")
    parser.add_argument("--threshold-percentile", type=float, default=89.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = sorted((args.source_dir / "sources_npy").glob("*.npy"))
    if not source_paths:
        raise SystemExit(f"No source .npy files found in {args.source_dir / 'sources_npy'}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    psf = gaussian_psf_kernel(args.psf_fwhm, args.delta_pix)
    lambdas = [float(item.strip()) for item in args.lambda_tv.split(",") if item.strip()]
    baseline_name = f"rl{args.rl_iterations}_{args.baseline_method}"

    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    records: dict[tuple[str, str], dict[str, object]] = {}

    for index, source_path in enumerate(source_paths[: args.count]):
        source = np.load(source_path).astype(float)
        source = make_compact_source(augment_source(source, rng), rng, args.source_size, 0.34, 0.58)
        case_id = f"support_tv_{index:05d}"
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
        if args.mask_source == "true":
            arc_mask = true_mask
        else:
            arc_mask = detect_arc_mask(observed, threshold_percentile=args.threshold_percentile, min_component_size=8)
        deconvolved = richardson_lucy(observed, psf, args.rl_iterations)

        start = time.perf_counter()
        baseline = robust_source_reconstruction(
            deconvolved,
            arc_mask,
            metadata,  # type: ignore[arg-type]
            args.delta_pix,
            valid_pixel_mode="arc",
            config=config_for(args),
        )
        baseline_runtime = time.perf_counter() - start
        truth_on_grid = sample_truth_on_grid(source, baseline.grid, args.source_extent)
        baseline_support = baseline.coverage > 0
        baseline_row = metric_row(case_id, baseline_name, baseline, truth_on_grid, observed, metadata, args.delta_pix, psf, baseline_runtime, baseline_support)
        rows.append(baseline_row)
        records[(case_id, baseline_name)] = {
            "observed": observed,
            "baseline_source": baseline.source,
            "source": baseline.source,
            "truth": truth_on_grid,
            "support": baseline_support,
        }

        support, support_stats, maps = clean_source_support_mask(
            baseline.coverage,
            baseline.beta_x,
            baseline.beta_y,
            baseline.grid,
            SupportCleanupConfig(),
        )
        support_source = apply_support_mask(baseline.source, support)
        support_result = make_result(baseline, support_source, support.astype(float), f"{baseline_name}_support_cleaned", support_stats)
        support_row = metric_row(case_id, f"{baseline_name}_support_cleaned", support_result, truth_on_grid, observed, metadata, args.delta_pix, psf, 0.0, support)
        rows.append(support_row)
        records[(case_id, f"{baseline_name}_support_cleaned")] = {
            "observed": observed,
            "baseline_source": baseline.source,
            "source": support_result.source,
            "truth": truth_on_grid,
            "support": support,
        }

        diagnostics.append(
            {
                "id": case_id,
                **support_stats,
                "diagnosis_sparse_coverage": support_stats["support_cleaned_fraction"] < 0.02,
                "diagnosis_invalid_support_removed": support_stats["support_removed_fraction"] > 0.0,
                "diagnosis_black_holes_from_unpopulated_support": finite_float(support_row.get("artefact_dark_hole_fraction")) < finite_float(baseline_row.get("artefact_dark_hole_fraction"), 1.0),
            }
        )

        last_gradient: np.ndarray | None = None
        for lambda_tv in lambdas:
            start_tv = time.perf_counter()
            tv_source, tv_stats, tv_gradient = tv_regularize_supported_source(
                support_source,
                support,
                TVRegularizationConfig(weight=lambda_tv, max_iterations=80),
            )
            tv_runtime = time.perf_counter() - start_tv
            last_gradient = tv_gradient
            tv_method = f"{baseline_name}_support_cleaned_tv{lambda_tv:g}"
            tv_result = make_result(baseline, tv_source, support.astype(float), tv_method, {**support_stats, **tv_stats})
            tv_row = metric_row(case_id, tv_method, tv_result, truth_on_grid, observed, metadata, args.delta_pix, psf, tv_runtime, support)
            rows.append(tv_row)
            records[(case_id, tv_method)] = {
                "observed": observed,
                "baseline_source": baseline.source,
                "source": tv_result.source,
                "truth": truth_on_grid,
                "support": support,
            }

        if index < args.save_images:
            maps = {**maps, "source_intensity": baseline.source}
            save_diagnostic_maps(args.out_dir, case_id, maps, last_gradient)
        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{min(args.count, len(source_paths))}")

    method_summary = summarise(rows)
    decision = select_method(method_summary, baseline_name)
    selected_method = str(decision["selected_method"])
    write_csv(args.out_dir / "per_image_metrics.csv", rows)
    write_csv(args.out_dir / "method_comparison.csv", method_summary)
    write_csv(args.out_dir / "artefact_diagnostics.csv", diagnostics)
    write_csv(args.out_dir / "morphology_metrics.csv", [{k: v for k, v in row.items() if k in {"id", "method", "centroid_error", "effective_radius_error", "ellipticity_error", "asymmetry_error", "gini_error", "concentration_error"}} for row in rows])
    write_csv(args.out_dir / "forward_metrics.csv", [{k: v for k, v in row.items() if k in {"id", "method", "forward_mse", "forward_ncc"}} for row in rows])
    (args.out_dir / "selected_method.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    save_ranked_examples(args.out_dir, rows, selected_method, records)

    note = [
        "# Final Reconstruction Decision",
        "",
        f"Selected method: `{selected_method}`",
        "",
        f"Baseline method: `{baseline_name}`",
        "",
        f"Decision rule: {decision.get('reason', '')}",
        "",
        "This validation only tested support-mask cleanup and TV regularisation. It did not alter detection, ellipse fitting, the inverse lens equation, or classifier labels.",
        "",
        "Artefact diagnosis:",
        "- beta sample density and nearest-sample distance maps were saved per selected example.",
        "- pixels outside cleaned support were forced to zero.",
        "- TV regularisation was restricted to cleaned support and flux-scaled to avoid unsupported structure.",
        "",
        "Important limitation: this is still inverse reconstruction from sparse lensed samples, not photorealistic recovery.",
    ]
    (args.out_dir / "FINAL_RECONSTRUCTION_DECISION.md").write_text("\n".join(note), encoding="utf-8")

    plt.figure(figsize=(8, 4.5), dpi=180)
    labels = [str(row["method"]) for row in method_summary]
    vals = [finite_float(row.get("median_ssim_aligned")) for row in method_summary]
    plt.bar(labels, vals)
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Median aligned SSIM")
    plt.tight_layout()
    plt.savefig(args.out_dir / "method_median_ssim.png")
    plt.close()
    print(f"Saved support+TV validation to {args.out_dir}")
    print(f"Selected method: {selected_method}")


if __name__ == "__main__":
    main()
