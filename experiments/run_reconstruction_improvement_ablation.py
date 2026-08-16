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
from lensing_pipeline.reconstruction import approximate_sie_ray_shooting
from lensing_pipeline.robust_reconstruction import (
    RobustReconstructionConfig,
    RobustReconstructionResult,
    SourceGrid,
    bilinear_weighted_accumulate,
    morphology_stats,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
)
from lensing_pipeline.visualization import save_grayscale_png
from run_cosmos_lensing_benchmark import augment_source, generate_case, make_compact_source, sample_source, stretch


BASE_METHODS = ["weighted", "griddata_linear", "clough_tocher"]
RBF_METHODS = [
    "rbf_linear_aggregated",
    "rbf_thin_plate_spline_aggregated",
    "rbf_cubic_aggregated",
    "rbf_multiquadric_aggregated",
    "rbf_inverse_multiquadric_aggregated",
    "rbf_gaussian_aggregated",
]
PRIMARY = ["ssim_aligned", "ncc_aligned", "psnr_aligned", "mse_aligned"]


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


def gaussian_psf_kernel(fwhm_arcsec: float, delta_pix: float, truncation: float = 3.0) -> np.ndarray:
    sigma_pix = max(float(fwhm_arcsec) / max(float(delta_pix), 1e-12) / 2.354820045, 0.05)
    radius = max(2, int(math.ceil(truncation * sigma_pix)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma_pix * sigma_pix))
    kernel /= max(float(kernel.sum()), 1e-12)
    return kernel


def richardson_lucy(image: np.ndarray, psf: np.ndarray, iterations: int, clip: bool = True) -> np.ndarray:
    if iterations <= 0:
        return np.clip(np.asarray(image, dtype=float), 0.0, None)
    observed = np.clip(np.asarray(image, dtype=float), 0.0, None)
    estimate = np.full_like(observed, max(float(np.mean(observed)), 1e-6), dtype=float)
    psf_mirror = psf[::-1, ::-1]
    initial_flux = float(observed.sum())
    for _ in range(int(iterations)):
        conv = signal.fftconvolve(estimate, psf, mode="same")
        relative_blur = observed / np.maximum(conv, 1e-8)
        estimate *= signal.fftconvolve(relative_blur, psf_mirror, mode="same")
        if clip:
            estimate = np.clip(estimate, 0.0, None)
    final_flux = float(estimate.sum())
    if final_flux > 0 and initial_flux > 0:
        estimate *= initial_flux / final_flux
    return np.clip(estimate, 0.0, None)


def config_for(args: argparse.Namespace, method: str, smoothing: float = 0.0, epsilon: float | None = None) -> RobustReconstructionConfig:
    rbf_epsilon = epsilon
    if rbf_epsilon is None and any(token in method for token in ("multiquadric", "gaussian")):
        pixel_scale = (2.0 * float(args.source_extent)) / max(int(args.source_size) - 1, 1)
        rbf_epsilon = max(pixel_scale * max(int(args.rbf_neighbors), 1), 1e-3)
    return RobustReconstructionConfig(
        output_size=args.source_size,
        source_extent=args.source_extent,
        auto_extent=True,
        auto_margin_fraction=0.12,
        auto_bound_low_percentile=args.low_percentile,
        auto_bound_high_percentile=args.high_percentile,
        min_auto_extent=0.05,
        reconstruction_method=method,
        scattered_max_grid_distance=args.scattered_max_grid_distance,
        beta_aggregation_sigma_clip=2.5,
        beta_aggregation_min_bin_samples=2,
        rbf_smoothing=smoothing,
        rbf_epsilon=rbf_epsilon,
        rbf_neighbors=args.rbf_neighbors,
        hole_fill="none",
        output_normalization="percentile",
        regularization="none",
    )


def gini(image: np.ndarray) -> float:
    values = np.sort(np.clip(np.asarray(image, dtype=float).ravel(), 0.0, None))
    if len(values) == 0 or float(values.sum()) <= 0:
        return 0.0
    idx = np.arange(1, len(values) + 1, dtype=float)
    return float(np.sum((2.0 * idx - len(values) - 1.0) * values) / (len(values) * values.sum()))


def m20(image: np.ndarray) -> float:
    arr = np.clip(np.asarray(image, dtype=float), 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return math.nan
    yy, xx = np.indices(arr.shape)
    cx = float((xx * arr).sum() / total)
    cy = float((yy * arr).sum() / total)
    moment = arr * ((xx - cx) ** 2 + (yy - cy) ** 2)
    total_moment = float(moment.sum())
    if total_moment <= 0:
        return math.nan
    order = np.argsort(arr.ravel())[::-1]
    flux_sorted = arr.ravel()[order]
    moment_sorted = moment.ravel()[order]
    cutoff = np.searchsorted(np.cumsum(flux_sorted), 0.2 * total, side="left")
    return float(np.log10(max(float(moment_sorted[: cutoff + 1].sum()), 1e-12) / total_moment))


def extended_morphology(image: np.ndarray) -> dict[str, float]:
    stats = morphology_stats(image)
    stats["ellipticity"] = float(1.0 - finite_float(stats.get("axis_ratio"), 1.0))
    stats["gini"] = gini(image)
    stats["m20"] = m20(image)
    return stats


def morphology_errors(truth: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    truth_m = extended_morphology(truth)
    recon_m = extended_morphology(match_photometric_scale(truth, reconstruction))
    return {
        "centroid_error": float(np.hypot(truth_m["centroid_x"] - recon_m["centroid_x"], truth_m["centroid_y"] - recon_m["centroid_y"])),
        "effective_radius_error": float(abs(truth_m["size"] - recon_m["size"])),
        "ellipticity_error": float(abs(truth_m["ellipticity"] - recon_m["ellipticity"])),
        "axis_ratio_error": float(abs(truth_m["axis_ratio"] - recon_m["axis_ratio"])),
        "concentration_error": float(abs(truth_m["concentration"] - recon_m["concentration"])),
        "asymmetry_error": float(abs(truth_m["asymmetry"] - recon_m["asymmetry"])),
        "gini_error": float(abs(truth_m["gini"] - recon_m["gini"])),
        "m20_error": float(abs(truth_m["m20"] - recon_m["m20"])) if np.isfinite(truth_m["m20"]) and np.isfinite(recon_m["m20"]) else math.nan,
    }


def source_grid_sample(source: np.ndarray, grid: SourceGrid, beta_x: np.ndarray, beta_y: np.ndarray) -> np.ndarray:
    n = grid.output_size
    sx = (beta_x - (grid.center_x - grid.extent)) / (2.0 * grid.extent) * (n - 1)
    sy = (beta_y - (grid.center_y - grid.extent)) / (2.0 * grid.extent) * (n - 1)
    return ndimage.map_coordinates(source, [sy, sx], order=1, mode="constant", cval=0.0)


def forward_lens_reconstruction(source: np.ndarray, grid: SourceGrid, metadata: dict[str, float | str], image_shape: tuple[int, int], delta_pix: float, psf: np.ndarray | None) -> np.ndarray:
    coords_x = (np.arange(image_shape[1], dtype=float) - (image_shape[1] - 1) / 2.0) * delta_pix
    coords_y = (np.arange(image_shape[0], dtype=float) - (image_shape[0] - 1) / 2.0) * delta_pix
    theta_x, theta_y = np.meshgrid(coords_x, coords_y)
    beta_x, beta_y = approximate_sie_ray_shooting(theta_x, theta_y, metadata)  # type: ignore[arg-type]
    predicted = source_grid_sample(source, grid, beta_x, beta_y)
    if psf is not None:
        predicted = signal.fftconvolve(predicted, psf, mode="same")
    return np.clip(predicted, 0.0, None)


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float).ravel()
    bb = np.asarray(b, dtype=float).ravel()
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    return float(np.dot(aa, bb) / denom) if denom > 0 else 0.0


def forward_metrics(observed: np.ndarray, predicted: np.ndarray, noise_sigma: float) -> dict[str, float]:
    pred = match_photometric_scale(observed, predicted)
    residual = np.asarray(observed, dtype=float) - pred
    return {
        "forward_mse": float(np.mean(residual * residual)),
        "forward_ncc": ncc(observed, pred),
        "forward_chi_square": float(np.mean((residual / max(noise_sigma, 1e-8)) ** 2)),
    }


def image_plane_loss(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Return photometrically matched image-plane MSE plus signed residual."""
    pred = match_photometric_scale(observed, predicted)
    residual = np.asarray(observed, dtype=float) - pred
    return float(np.mean(residual * residual)), pred, residual


def aggregate_image_residual_to_source(
    residual: np.ndarray,
    result: RobustReconstructionResult,
) -> tuple[np.ndarray, np.ndarray]:
    """Back-project signed image residuals onto the existing source grid."""
    yy, xx = np.nonzero(result.valid_mask)
    if len(xx) == 0:
        return np.zeros_like(result.source), np.zeros_like(result.source)
    values = np.asarray(residual, dtype=float)[yy, xx]
    delta_source, delta_coverage, _valid = bilinear_weighted_accumulate(result.beta_x, result.beta_y, values, result.grid)
    return delta_source, delta_coverage


def iterative_forward_backward_refine(
    initial: RobustReconstructionResult,
    observed: np.ndarray,
    metadata: dict[str, float | str],
    delta_pix: float,
    psf: np.ndarray,
    eta: float,
    max_iterations: int,
    early_stop_patience: int = 2,
) -> tuple[RobustReconstructionResult, dict[str, float | str]]:
    """Improve a source estimate by relensing it, back-projecting signed residuals, and accepting only loss-reducing updates."""
    support = (np.asarray(initial.source) > 0.0) & np.isfinite(initial.source)
    if not np.any(support):
        return initial, {
            "method3_iterations_requested": float(max_iterations),
            "method3_eta": float(eta),
            "method3_iterations_accepted": 0.0,
            "method3_initial_forward_mse": math.nan,
            "method3_best_forward_mse": math.nan,
            "method3_status": "skipped_empty_support",
        }

    current = np.clip(np.asarray(initial.source, dtype=float), 0.0, None)
    predicted = forward_lens_reconstruction(current, initial.grid, metadata, observed.shape, delta_pix, psf)
    current_loss, _current_pred, residual = image_plane_loss(observed, predicted)
    initial_loss = current_loss
    best_source = current.copy()
    best_loss = current_loss
    accepted = 0
    non_improving = 0
    for iteration in range(int(max_iterations)):
        delta_source, delta_coverage = aggregate_image_residual_to_source(residual, initial)
        update = np.zeros_like(current)
        update[support & (delta_coverage > 0)] = delta_source[support & (delta_coverage > 0)]
        candidate = current.copy()
        candidate[support] = np.clip(candidate[support] + float(eta) * update[support], 0.0, None)
        predicted_candidate = forward_lens_reconstruction(candidate, initial.grid, metadata, observed.shape, delta_pix, psf)
        candidate_loss, _candidate_pred, candidate_residual = image_plane_loss(observed, predicted_candidate)
        if candidate_loss < current_loss - 1e-12:
            current = candidate
            current_loss = candidate_loss
            residual = candidate_residual
            accepted += 1
            non_improving = 0
            if candidate_loss < best_loss:
                best_loss = candidate_loss
                best_source = candidate.copy()
        else:
            non_improving += 1
            if non_improving >= int(early_stop_patience):
                break

    stats = {
        **initial.stats,
        "reconstruction_method": f"{initial.stats.get('reconstruction_method', 'unknown')}+iterative_refinement",
        "method3_iterations_requested": float(max_iterations),
        "method3_eta": float(eta),
        "method3_iterations_accepted": float(accepted),
        "method3_initial_forward_mse": float(initial_loss),
        "method3_best_forward_mse": float(best_loss),
        "method3_forward_mse_delta": float(best_loss - initial_loss),
        "method3_status": "accepted" if accepted > 0 else "no_loss_reducing_update",
    }
    refined = RobustReconstructionResult(
        source=np.clip(best_source, 0.0, None),
        coverage=initial.coverage,
        valid_mask=initial.valid_mask,
        beta_x=initial.beta_x,
        beta_y=initial.beta_y,
        grid=initial.grid,
        stats=stats,
    )
    return refined, stats


def metric_row(case_id: str, method: str, truth: np.ndarray, result: RobustReconstructionResult, runtime: float, observed: np.ndarray, metadata: dict[str, float | str], delta_pix: float, psf: np.ndarray | None, noise_sigma: float) -> dict[str, object]:
    quality = reconstruction_quality_metrics(truth, result.source)
    morph = morphology_errors(truth, result.source)
    forward = forward_metrics(observed, forward_lens_reconstruction(result.source, result.grid, metadata, observed.shape, delta_pix, psf), noise_sigma)
    row: dict[str, object] = {
        "id": case_id,
        "method": method,
        "runtime_seconds": runtime,
        **quality,
        **morph,
        **forward,
    }
    row.update({f"stat_{key}": value for key, value in result.stats.items()})
    return row


def save_gray(path: Path, image: np.ndarray, shared_scale: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if shared_scale and shared_scale > 0:
        arr = np.clip(np.asarray(image, dtype=float) / shared_scale, 0.0, 1.0)
    else:
        arr = stretch(image)
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


def signed_display(signed: np.ndarray) -> np.ndarray:
    max_abs = max(float(np.max(np.abs(signed))), 1e-10)
    return np.clip(0.5 + 0.5 * signed / max_abs, 0.0, 1.0)


def save_example_panel(path: Path, case_id: str, observed: np.ndarray, deconvolved: np.ndarray, baseline: np.ndarray, improved: np.ndarray, truth: np.ndarray, method: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = max(float(np.max(truth)), float(np.max(baseline)), float(np.max(improved)), 1e-10)
    residual = np.abs(np.clip(truth / scale, 0.0, 1.0) - np.clip(improved / scale, 0.0, 1.0))
    panels = [
        ("observed image", stretch(observed)),
        ("deconvolved image", stretch(deconvolved)),
        ("baseline CT", np.clip(baseline / scale, 0.0, 1.0)),
        (method, np.clip(improved / scale, 0.0, 1.0)),
        ("true source", np.clip(truth / scale, 0.0, 1.0)),
        ("abs residual", stretch(residual)),
    ]
    tile = 150
    label_h = 24
    footer_h = 20
    canvas = Image.new("RGB", (tile * len(panels), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(panels):
        x = idx * tile
        draw.text((x + 4, 4), label[:22], fill="black")
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")
        canvas.paste(img.resize((tile, tile), Image.Resampling.BICUBIC), (x, label_h))
    draw.text((4, tile + label_h + 3), case_id, fill="black")
    canvas.save(path)


def save_method3_panel(
    path: Path,
    case_id: str,
    observed: np.ndarray,
    initial: np.ndarray,
    refined: np.ndarray,
    truth: np.ndarray,
    relensed_refined: np.ndarray,
    method: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    source_scale = max(float(np.max(truth)), float(np.max(initial)), float(np.max(refined)), 1e-10)
    image_scale = max(float(np.max(observed)), float(np.max(relensed_refined)), 1e-10)
    source_residual = np.clip(truth / source_scale, 0.0, 1.0) - np.clip(refined / source_scale, 0.0, 1.0)
    relensed_scaled = match_photometric_scale(observed, relensed_refined)
    image_residual = observed - relensed_scaled
    panels = [
        ("observed lens", np.clip(observed / image_scale, 0.0, 1.0)),
        ("initial source", np.clip(initial / source_scale, 0.0, 1.0)),
        ("refined source", np.clip(refined / source_scale, 0.0, 1.0)),
        ("true source", np.clip(truth / source_scale, 0.0, 1.0)),
        ("source residual", signed_display(source_residual)),
        ("relensed refined", np.clip(relensed_scaled / image_scale, 0.0, 1.0)),
        ("image residual", signed_display(image_residual)),
    ]
    tile = 142
    label_h = 24
    footer_h = 22
    canvas = Image.new("RGB", (tile * len(panels), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(panels):
        x = idx * tile
        draw.text((x + 4, 4), label[:20], fill="black")
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")
        canvas.paste(img.resize((tile, tile), Image.Resampling.BICUBIC), (x, label_h))
    draw.text((4, tile + label_h + 4), f"{case_id} | {method}"[:140], fill="black")
    canvas.save(path)


def save_ranked_examples(out_dir: Path, records: list[dict[str, object]], selected_method: str) -> None:
    selected = [r for r in records if r["method"] == selected_method]
    if not selected:
        selected = list(records)
    selected.sort(key=lambda r: finite_float(r["ssim_aligned"]), reverse=True)
    groups = {
        "best_examples": selected[:5],
        "worst_examples": selected[-5:][::-1],
    }
    mid = len(selected) // 2
    groups["median_examples"] = selected[max(0, mid - 2) : min(len(selected), mid + 3)]
    for folder, rows in groups.items():
        for idx, row in enumerate(rows, start=1):
            save_example_panel(
                out_dir / folder / f"{idx:02d}_{row['id']}_{selected_method}.png",
                str(row["id"]),
                row["observed"],
                row["deconvolved"],
                row["baseline"],
                row["reconstruction"],
                row["truth"],
                selected_method,
            )


def save_method3_ranked_examples(out_dir: Path, records: list[dict[str, object]]) -> None:
    selected = [r for r in records if str(r.get("method", "")).endswith("_iterative_refined_selected")]
    selected.sort(key=lambda r: finite_float(r["ssim_aligned"]), reverse=True)
    groups = {
        "method3_best_examples": selected[:5],
        "method3_worst_examples": selected[-5:][::-1],
    }
    mid = len(selected) // 2
    groups["method3_median_examples"] = selected[max(0, mid - 2) : min(len(selected), mid + 3)]
    for folder, rows in groups.items():
        for idx, row in enumerate(rows, start=1):
            save_method3_panel(
                out_dir / folder / f"{idx:02d}_{row['id']}_{row['method']}.png",
                str(row["id"]),
                row["observed"],
                row["initial"],
                row["refined"],
                row["truth"],
                row["relensed_refined"],
                str(row["method"]),
            )


def summarise_methods(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    out = []
    for method, method_rows in grouped.items():
        item: dict[str, object] = {"method": method, "count": len(method_rows)}
        for metric in ["ssim_aligned", "ncc_aligned", "psnr_aligned", "mse_aligned", "centroid_error", "effective_radius_error", "ellipticity_error", "asymmetry_error", "gini_error", "concentration_error", "forward_mse", "forward_ncc", "runtime_seconds"]:
            vals = np.array([finite_float(r.get(metric)) for r in method_rows], dtype=float)
            vals = vals[np.isfinite(vals)]
            item[f"mean_{metric}"] = float(np.mean(vals)) if len(vals) else math.nan
            item[f"median_{metric}"] = float(np.median(vals)) if len(vals) else math.nan
        out.append(item)
    out.sort(key=lambda r: finite_float(r.get("median_ssim_aligned")), reverse=True)
    return out


def parameter_selection(rows: list[dict[str, object]], baseline_method: str = "clough_tocher") -> list[dict[str, object]]:
    summaries = summarise_methods(rows)
    baseline = next((s for s in summaries if s["method"] == baseline_method), None)
    out = []
    for summary in summaries:
        method = str(summary["method"])
        method_rows = [r for r in rows if r["method"] == method]
        base_by_id = {r["id"]: r for r in rows if r["method"] == baseline_method}
        ssim_improved = []
        ncc_improved = []
        morph_improved = []
        forward_worse = []
        for row in method_rows:
            base = base_by_id.get(row["id"])
            if not base:
                continue
            ssim_improved.append(finite_float(row["ssim_aligned"]) > finite_float(base["ssim_aligned"]))
            ncc_improved.append(finite_float(row["ncc_aligned"]) > finite_float(base["ncc_aligned"]))
            morph = finite_float(row["centroid_error"]) + finite_float(row["effective_radius_error"]) + finite_float(row["ellipticity_error"])
            base_morph = finite_float(base["centroid_error"]) + finite_float(base["effective_radius_error"]) + finite_float(base["ellipticity_error"])
            morph_improved.append(morph < base_morph)
            forward_worse.append(finite_float(row["forward_mse"]) > finite_float(base["forward_mse"]))
        out.append(
            {
                **summary,
                "baseline_method": baseline_method,
                "delta_median_ssim_vs_baseline": finite_float(summary.get("median_ssim_aligned")) - finite_float(baseline.get("median_ssim_aligned")) if baseline else math.nan,
                "delta_median_ncc_vs_baseline": finite_float(summary.get("median_ncc_aligned")) - finite_float(baseline.get("median_ncc_aligned")) if baseline else math.nan,
                "percent_validation_images_improved_ssim": 100.0 * float(np.mean(ssim_improved)) if ssim_improved else math.nan,
                "percent_validation_images_improved_ncc": 100.0 * float(np.mean(ncc_improved)) if ncc_improved else math.nan,
                "percent_validation_images_improved_morphology": 100.0 * float(np.mean(morph_improved)) if morph_improved else math.nan,
                "percent_improvement_worse_forward_consistency": 100.0 * float(np.mean(forward_worse)) if forward_worse else math.nan,
            }
        )
    return out


def rbf_kernel_verification_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        method = str(row.get("method", ""))
        if "rbf_" in method and "iterative_refined" not in method:
            grouped[method].append(row)
    out = []
    for method, method_rows in grouped.items():
        kernels = sorted({str(r.get("stat_scattered_rbf_kernel", "")) for r in method_rows if str(r.get("stat_scattered_rbf_kernel", ""))})
        fallbacks = [r for r in method_rows if "fallback" in str(r.get("stat_scattered_scattered_method_used", ""))]
        out.append(
            {
                "method": method,
                "count": len(method_rows),
                "reported_rbf_kernels": ";".join(kernels),
                "fallback_count": len(fallbacks),
                "executed_without_fallback": len(fallbacks) == 0 and bool(kernels),
                "median_ssim_aligned": float(np.median([finite_float(r.get("ssim_aligned")) for r in method_rows])),
                "median_ncc_aligned": float(np.median([finite_float(r.get("ncc_aligned")) for r in method_rows])),
                "median_forward_mse": float(np.median([finite_float(r.get("forward_mse")) for r in method_rows])),
            }
        )
    out.sort(key=lambda r: str(r["method"]))
    return out


def method3_acceptance_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    out = []
    initial_pairs = [
        ("rl15_rbf_linear_aggregated_smooth0", "rl15_rbf_linear_aggregated_smooth0_iterative_refined_selected"),
        ("rl15_rbf_thin_plate_spline_aggregated_smooth0", "rl15_rbf_thin_plate_spline_aggregated_smooth0_iterative_refined_selected"),
    ]
    for initial_method, refined_method in initial_pairs:
        initial_by_id = {r["id"]: r for r in rows if r.get("method") == initial_method}
        refined_rows = [r for r in rows if r.get("method") == refined_method]
        both_source_and_forward = []
        accepted = []
        degraded_morphology = []
        for refined in refined_rows:
            initial = initial_by_id.get(refined["id"])
            if not initial:
                continue
            source_improved = finite_float(refined.get("ssim_aligned")) >= finite_float(initial.get("ssim_aligned")) and finite_float(refined.get("ncc_aligned")) >= finite_float(initial.get("ncc_aligned"))
            forward_improved = finite_float(refined.get("forward_mse")) < finite_float(initial.get("forward_mse"))
            both_source_and_forward.append(source_improved and forward_improved)
            accepted.append(str(refined.get("method3_acceptance_decision")) == "accepted")
            initial_shape = finite_float(initial.get("centroid_error")) + finite_float(initial.get("effective_radius_error")) + finite_float(initial.get("ellipticity_error"))
            refined_shape = finite_float(refined.get("centroid_error")) + finite_float(refined.get("effective_radius_error")) + finite_float(refined.get("ellipticity_error"))
            degraded_morphology.append(refined_shape > initial_shape * 1.10)
        out.append(
            {
                "initial_method": initial_method,
                "refined_method": refined_method,
                "count": len(refined_rows),
                "percent_improved_source_ssim_and_forward_mse": 100.0 * float(np.mean(both_source_and_forward)) if both_source_and_forward else math.nan,
                "percent_accepted_by_rule": 100.0 * float(np.mean(accepted)) if accepted else math.nan,
                "percent_material_shape_degradation": 100.0 * float(np.mean(degraded_morphology)) if degraded_morphology else math.nan,
            }
        )
    return out


def save_method_barplot(path: Path, summaries: list[dict[str, object]], metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = [str(r["method"]) for r in summaries]
    vals = [finite_float(r.get(f"median_{metric}")) for r in summaries]
    plt.figure(figsize=(9, 4.8), dpi=180)
    plt.bar(methods, vals, color="#386cb0")
    plt.xticks(rotation=35, ha="right")
    plt.ylabel(f"Median {metric}")
    plt.title(f"Method comparison: {metric}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate low-risk reconstruction improvements: RL deconvolution and RBF interpolation.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "cosmos_sources")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "reconstruction_improved")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=25015168)
    parser.add_argument("--source-size", type=int, default=128)
    parser.add_argument("--num-pix", type=int, default=128)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--source-extent", type=float, default=0.62)
    parser.add_argument("--noise-sigma", type=float, default=0.018)
    parser.add_argument("--psf-fwhm", type=float, default=0.08)
    parser.add_argument("--threshold-percentile", type=float, default=89.0)
    parser.add_argument("--min-component-size", type=int, default=8)
    parser.add_argument("--low-percentile", type=float, default=5.0)
    parser.add_argument("--high-percentile", type=float, default=95.0)
    parser.add_argument("--scattered-max-grid-distance", type=float, default=3.0)
    parser.add_argument("--rbf-neighbors", type=int, default=48)
    parser.add_argument("--rbf-smoothing-values", type=str, default="0,0.0005")
    parser.add_argument("--rl-iterations", type=str, default="0,3,5,10,15")
    parser.add_argument("--max-rbf-methods", type=int, default=6)
    parser.add_argument("--method3-iterations", type=str, default="1,3,5,10")
    parser.add_argument("--method3-etas", type=str, default="0.02,0.05,0.1,0.2")
    parser.add_argument("--disable-method3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = sorted((args.source_dir / "sources_npy").glob("*.npy"))
    if not source_paths:
        raise SystemExit(f"No source .npy files found in {args.source_dir / 'sources_npy'}")
    count = min(len(source_paths), args.count if args.count > 0 else len(source_paths))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    psf = gaussian_psf_kernel(args.psf_fwhm, args.delta_pix)
    rl_iterations = [int(x.strip()) for x in args.rl_iterations.split(",") if x.strip()]
    if 15 not in rl_iterations:
        rl_iterations.append(15)
        rl_iterations = sorted(set(rl_iterations))
    rbf_smoothing_values = [float(x.strip()) for x in args.rbf_smoothing_values.split(",") if x.strip()]
    method3_iterations = [int(x.strip()) for x in args.method3_iterations.split(",") if x.strip()]
    method3_etas = [float(x.strip()) for x in args.method3_etas.split(",") if x.strip()]
    rbf_methods = RBF_METHODS[: max(0, args.max_rbf_methods)]
    variants: list[tuple[str, str, int, float, float | None]] = []
    variants.extend((method, method, 0, 0.0, None) for method in BASE_METHODS)
    variants.extend((f"rl{it}_clough_tocher", "clough_tocher", it, 0.0, None) for it in rl_iterations if it > 0)
    for method in rbf_methods:
        for smoothing in rbf_smoothing_values:
            variants.append((f"{method}_smooth{smoothing:g}", method, 0, smoothing, None))
            for it in rl_iterations:
                if it > 0:
                    variants.append((f"rl{it}_{method}_smooth{smoothing:g}", method, it, smoothing, None))

    rows: list[dict[str, object]] = []
    visual_records: list[dict[str, object]] = []
    method3_visual_records: list[dict[str, object]] = []
    for index, source_path in enumerate(source_paths[:count]):
        source = normalize_image(augment_source(np.load(source_path), rng))
        source = make_compact_source(source, rng, args.source_size, 0.20, 0.34)
        case_id = f"improve_{index:05d}"
        image, _clean, lensed_source, _true_mask, metadata = generate_case(
            source,
            source_path.stem,
            case_id,
            rng,
            num_pix=args.num_pix,
            delta_pix=args.delta_pix,
            source_extent=args.source_extent,
            noise_sigma=args.noise_sigma,
        )
        metadata["psf_fwhm"] = float(args.psf_fwhm)
        metadata["noise_sigma"] = float(args.noise_sigma)
        blurred_lensed_source = np.clip(signal.fftconvolve(lensed_source, psf, mode="same"), 0.0, None)
        observed_source = normalize_image(blurred_lensed_source + rng.normal(0.0, args.noise_sigma, lensed_source.shape))
        observed_for_detection = normalize_image(signal.fftconvolve(image, psf, mode="same") + rng.normal(0.0, args.noise_sigma, image.shape))
        detected_mask = detect_arc_mask(observed_for_detection, args.threshold_percentile, min_component_size=args.min_component_size, suppress_centre_radius=0.08)
        deconvolved_cache: dict[int, np.ndarray] = {}
        baseline_result: RobustReconstructionResult | None = None
        baseline_truth: np.ndarray | None = None
        results_by_method: dict[str, tuple[RobustReconstructionResult, np.ndarray, np.ndarray, float]] = {}
        for variant_name, recon_method, rl_iter, smoothing, epsilon in variants:
            if rl_iter not in deconvolved_cache:
                deconvolved_cache[rl_iter] = richardson_lucy(observed_source, psf, rl_iter)
            recon_input = deconvolved_cache[rl_iter]
            start = time.perf_counter()
            result = robust_source_reconstruction(
                recon_input,
                detected_mask,
                metadata,
                args.delta_pix,
                valid_pixel_mode="arc",
                config=config_for(args, recon_method, smoothing=smoothing, epsilon=epsilon),
            )
            runtime = time.perf_counter() - start
            truth = sample_truth_on_grid(source, result.grid, args.source_extent)
            row = metric_row(case_id, variant_name, truth, result, runtime, observed_source, metadata, args.delta_pix, psf, args.noise_sigma)
            row.update(
                {
                    "source_id": source_path.stem,
                    "reconstruction_method": recon_method,
                    "rl_iterations": rl_iter,
                    "rbf_smoothing": smoothing,
                    "psf_fwhm": args.psf_fwhm,
                    "noise_sigma": args.noise_sigma,
                }
            )
            rows.append(row)
            results_by_method[variant_name] = (result, truth, recon_input, runtime)
            if variant_name == "clough_tocher":
                baseline_result = result
                baseline_truth = truth

        if not args.disable_method3:
            for initial_method in ("rl15_rbf_linear_aggregated_smooth0", "rl15_rbf_thin_plate_spline_aggregated_smooth0"):
                if initial_method not in results_by_method:
                    continue
                initial_result, initial_truth, _initial_input, _initial_runtime = results_by_method[initial_method]
                initial_row = next(r for r in rows if r["id"] == case_id and r["method"] == initial_method)
                best_refined_result: RobustReconstructionResult | None = None
                best_refined_row: dict[str, object] | None = None
                best_refined_method = ""
                best_image_loss = math.inf
                for eta in method3_etas:
                    for max_iter in method3_iterations:
                        method3_name = f"{initial_method}_iterative_eta{eta:g}_iter{max_iter}"
                        start = time.perf_counter()
                        refined_result, refine_stats = iterative_forward_backward_refine(
                            initial_result,
                            observed_source,
                            metadata,
                            args.delta_pix,
                            psf,
                            eta=eta,
                            max_iterations=max_iter,
                        )
                        runtime = time.perf_counter() - start
                        row = metric_row(case_id, method3_name, initial_truth, refined_result, runtime, observed_source, metadata, args.delta_pix, psf, args.noise_sigma)
                        row.update(
                            {
                                "source_id": source_path.stem,
                                "reconstruction_method": str(refined_result.stats.get("reconstruction_method", "iterative_refinement")),
                                "rl_iterations": 15,
                                "rbf_smoothing": 0.0,
                                "method3_initial_method": initial_method,
                                "method3_eta": eta,
                                "method3_iterations_requested": max_iter,
                                **{key: value for key, value in refine_stats.items() if key.startswith("method3_")},
                            }
                        )
                        rows.append(row)
                        image_loss = finite_float(row.get("forward_mse"), math.inf)
                        if image_loss < best_image_loss:
                            best_image_loss = image_loss
                            best_refined_result = refined_result
                            best_refined_row = row
                            best_refined_method = method3_name
                if best_refined_result is not None and best_refined_row is not None:
                    selected_method = f"{initial_method}_iterative_refined_selected"
                    selected_row = dict(best_refined_row)
                    selected_row["method"] = selected_method
                    selected_row["method3_selected_from"] = best_refined_method
                    source_ok = finite_float(selected_row.get("ssim_aligned")) >= finite_float(initial_row.get("ssim_aligned")) and finite_float(selected_row.get("ncc_aligned")) >= finite_float(initial_row.get("ncc_aligned"))
                    forward_ok = finite_float(selected_row.get("forward_mse")) < finite_float(initial_row.get("forward_mse"))
                    initial_shape = finite_float(initial_row.get("centroid_error")) + finite_float(initial_row.get("effective_radius_error")) + finite_float(initial_row.get("ellipticity_error"))
                    refined_shape = finite_float(selected_row.get("centroid_error")) + finite_float(selected_row.get("effective_radius_error")) + finite_float(selected_row.get("ellipticity_error"))
                    shape_ok = refined_shape <= initial_shape * 1.10
                    selected_row["method3_source_ssim_ncc_maintained_or_improved"] = source_ok
                    selected_row["method3_forward_mse_improved"] = forward_ok
                    selected_row["method3_shape_not_materially_worse"] = shape_ok
                    selected_row["method3_acceptance_decision"] = "accepted" if (source_ok and forward_ok and shape_ok) else "rejected"
                    rows.append(selected_row)
                    relensed_refined = forward_lens_reconstruction(best_refined_result.source, best_refined_result.grid, metadata, observed_source.shape, args.delta_pix, psf)
                    method3_visual_records.append(
                        {
                            **selected_row,
                            "observed": observed_source,
                            "initial": initial_result.source,
                            "refined": best_refined_result.source,
                            "truth": initial_truth,
                            "relensed_refined": relensed_refined,
                        }
                    )
        summaries_so_far = summarise_methods([r for r in rows if r["id"] == case_id])
        best_method = summaries_so_far[0]["method"]
        best_row = next(r for r in rows if r["id"] == case_id and r["method"] == best_method)
        if baseline_result is not None and baseline_truth is not None:
            best_result = None
            best_deconv = observed_source
            for variant_name, recon_method, rl_iter, smoothing, epsilon in variants:
                if variant_name == best_method:
                    best_deconv = deconvolved_cache[rl_iter]
                    best_result = robust_source_reconstruction(
                        best_deconv,
                        detected_mask,
                        metadata,
                        args.delta_pix,
                        valid_pixel_mode="arc",
                        config=config_for(args, recon_method, smoothing=smoothing, epsilon=epsilon),
                    )
                    break
            if best_result is not None:
                visual_records.append(
                    {
                        **best_row,
                        "observed": observed_source,
                        "deconvolved": best_deconv,
                        "baseline": baseline_result.source,
                        "reconstruction": best_result.source,
                        "truth": sample_truth_on_grid(source, best_result.grid, args.source_extent),
                    }
                )
                if index < 8:
                    save_example_panel(args.out_dir / "figures" / f"{case_id}_{best_method}.png", case_id, observed_source, best_deconv, baseline_result.source, best_result.source, sample_truth_on_grid(source, best_result.grid, args.source_extent), str(best_method))
        if (index + 1) % 10 == 0 or index + 1 == count:
            print(f"Processed {index + 1}/{count}", flush=True)

    summaries = summarise_methods(rows)
    selection = parameter_selection(rows)
    selected_method = str(selection[0]["method"]) if selection else "clough_tocher"
    save_ranked_examples(args.out_dir, visual_records, selected_method)
    save_method3_ranked_examples(args.out_dir, method3_visual_records)
    rbf_verification = rbf_kernel_verification_rows(rows)
    method3_summary = method3_acceptance_summary(rows)
    write_csv(args.out_dir / "per_image_metrics.csv", rows)
    write_csv(args.out_dir / "method_comparison.csv", summaries)
    write_csv(args.out_dir / "parameter_selection.csv", selection)
    write_csv(args.out_dir / "rbf_kernel_verification.csv", rbf_verification)
    write_csv(args.out_dir / "method3_acceptance_summary.csv", method3_summary)
    write_csv(args.out_dir / "morphology_metrics.csv", [{k: v for k, v in row.items() if k in {"id", "method", "centroid_error", "effective_radius_error", "ellipticity_error", "axis_ratio_error", "concentration_error", "asymmetry_error", "gini_error", "m20_error"}} for row in rows])
    write_csv(args.out_dir / "forward_consistency_metrics.csv", [{k: v for k, v in row.items() if k in {"id", "method", "forward_mse", "forward_ncc", "forward_chi_square", "runtime_seconds"}} for row in rows])
    save_method_barplot(args.out_dir / "figures" / "median_ssim_by_method.png", summaries, "ssim_aligned")
    save_method_barplot(args.out_dir / "figures" / "median_ncc_by_method.png", summaries, "ncc_aligned")
    summary = {
        "count": count,
        "implemented_methods": ["Method 1: Richardson-Lucy PSF deconvolution", "Method 2: RBF scattered-data interpolation", "Method 3: iterative forward-backward source refinement"],
        "method_3_status": "implemented_as_optional_ablation_without_true_source_during_refinement" if not args.disable_method3 else "disabled_by_command_line",
        "baseline_preserved": "clough_tocher",
        "selected_method_by_validation_median_ssim": selected_method,
        "selection_caveat": "Only replace Clough-Tocher if improvements in median SSIM/NCC and morphology preservation are reliable.",
        "method3_acceptance_criterion": "lower forward-image residual while maintaining/improving source SSIM/NCC and not materially worsening centroid/effective-radius/ellipticity",
        "psf": {"type": "Gaussian", "fwhm_arcsec": args.psf_fwhm, "delta_pix": args.delta_pix},
        "rl_iterations_tested": rl_iterations,
        "rbf_methods_tested": rbf_methods,
        "rbf_smoothing_values": rbf_smoothing_values,
        "method3_iterations_tested": method3_iterations,
        "method3_etas_tested": method3_etas,
        "rbf_kernel_verification": rbf_verification,
        "method3_acceptance_summary": method3_summary,
        "top_methods": summaries[:5],
        "output_files": [
            "method_comparison.csv",
            "parameter_selection.csv",
            "rbf_kernel_verification.csv",
            "method3_acceptance_summary.csv",
            "per_image_metrics.csv",
            "morphology_metrics.csv",
            "forward_consistency_metrics.csv",
            "best_examples/",
            "median_examples/",
            "worst_examples/",
            "method3_best_examples/",
            "method3_median_examples/",
            "method3_worst_examples/",
            "figures/",
        ],
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Selected validation method: {selected_method}")
    print(f"Saved reconstruction improvement ablation to {args.out_dir}")


if __name__ == "__main__":
    main()
