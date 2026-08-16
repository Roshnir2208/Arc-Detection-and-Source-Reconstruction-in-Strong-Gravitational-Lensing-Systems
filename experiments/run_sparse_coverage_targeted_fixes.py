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
from scipy import interpolate, ndimage, signal, spatial
from scipy.spatial import cKDTree

from lensing_pipeline.metrics import match_photometric_scale
from lensing_pipeline.robust_reconstruction import (
    RobustReconstructionConfig,
    RobustReconstructionResult,
    aggregate_beta_samples,
    bilinear_weighted_accumulate,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
    source_grid_coordinates,
)
from run_cosmos_lensing_benchmark import augment_source, generate_case, make_compact_source, stretch
from run_reconstruction_improvement_ablation import forward_lens_reconstruction, morphology_errors
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


def normalise_physical(source: np.ndarray) -> tuple[np.ndarray, float]:
    source = np.clip(np.asarray(source, dtype=float), 0.0, None)
    positive = source[source > 0]
    scale = float(np.percentile(positive, 99.5)) if len(positive) else 0.0
    if scale > 0:
        return np.clip(source / scale, 0.0, 1.0), scale
    return source, scale


def config_for(args: argparse.Namespace, method: str) -> RobustReconstructionConfig:
    return RobustReconstructionConfig(
        output_size=args.source_size,
        source_extent=args.source_extent,
        auto_extent=True,
        auto_margin_fraction=0.12,
        auto_bound_low_percentile=5.0,
        auto_bound_high_percentile=95.0,
        min_auto_extent=0.05,
        reconstruction_method=method,
        scattered_max_grid_distance=args.scattered_max_grid_distance,
        beta_aggregation_sigma_clip=2.5,
        beta_aggregation_min_bin_samples=2,
        rbf_smoothing=0.0,
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


def extract_samples(result: RobustReconstructionResult, deconvolved: np.ndarray) -> dict[str, object]:
    yy, xx = np.nonzero(result.valid_mask)
    values = np.asarray(deconvolved, dtype=float)[yy, xx]
    _weighted, direct_coverage, source_valid = bilinear_weighted_accumulate(result.beta_x, result.beta_y, values, result.grid)
    beta_x = result.beta_x[source_valid]
    beta_y = result.beta_y[source_valid]
    values = values[source_valid]
    beta_x, beta_y, values, aggregate_stats = aggregate_beta_samples(
        beta_x,
        beta_y,
        values,
        result.grid,
        sigma_clip=2.5,
        min_bin_samples=2,
    )
    points = np.column_stack([beta_x.astype(float), beta_y.astype(float)])
    if len(points) > 5000:
        subset = np.linspace(0, len(points) - 1, 5000).astype(int)
        points = points[subset]
        values = values[subset]
    grid_beta_x, grid_beta_y = source_grid_coordinates(result.grid)
    target_points = np.column_stack([grid_beta_x.ravel(), grid_beta_y.ravel()])
    pixel_scale = (2.0 * result.grid.extent) / max(result.grid.output_size - 1, 1)
    return {
        "points": points,
        "values": values.astype(float),
        "target_points": target_points,
        "direct_coverage": direct_coverage,
        "pixel_scale": pixel_scale,
        "aggregate_stats": aggregate_stats,
    }


def support_fractions(points: np.ndarray, direct_coverage: np.ndarray, target_points: np.ndarray, pixel_scale: float, max_grid_distance: float) -> dict[str, object]:
    tree = cKDTree(points)
    nearest_distance, _ = tree.query(target_points, k=1)
    support = nearest_distance <= max_grid_distance * pixel_scale
    try:
        inside_hull = spatial.Delaunay(points).find_simplex(target_points) >= 0
    except Exception:
        inside_hull = np.zeros(len(target_points), dtype=bool)
    direct_support = direct_coverage.ravel() > 0
    interpolated = support & ~direct_support & inside_hull
    extrapolated = support & ~inside_hull
    return {
        "support": support.reshape(direct_coverage.shape),
        "inside_hull": inside_hull.reshape(direct_coverage.shape),
        "nearest_distance_pixels": (nearest_distance / max(pixel_scale, 1e-12)).reshape(direct_coverage.shape),
        "direct_supported_fraction": float(np.mean(direct_support)),
        "interpolated_fraction": float(np.mean(interpolated)),
        "extrapolated_fraction": float(np.mean(extrapolated)),
        "support_masked_fraction": float(np.mean(~support)),
        "nearest_distance_pixels_p95": float(np.percentile(nearest_distance / max(pixel_scale, 1e-12), 95.0)),
    }


def adaptive_confidence(points: np.ndarray, target_points: np.ndarray, pixel_scale: float, max_grid_distance: float, neighbours: int) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(points)
    k = min(max(int(neighbours), 4), len(points))
    distances, _indices = tree.query(target_points, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
    local_radius = np.maximum(np.percentile(distances, 50.0, axis=1), 1.5 * pixel_scale)
    local_radius = np.minimum(local_radius, max_grid_distance * pixel_scale)
    close_count = np.sum(distances <= local_radius[:, None], axis=1)
    nearest = distances[:, 0]
    confidence = np.exp(-((nearest / np.maximum(local_radius, 1e-12)) ** 2)) * np.minimum(1.0, close_count / 6.0)
    support = (nearest <= max_grid_distance * pixel_scale) & (close_count >= 4)
    return np.where(support, confidence, 0.0), support


def rbf_linear_confidence_taper(samples: dict[str, object], output_shape: tuple[int, int], max_grid_distance: float, neighbours: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    points = samples["points"]
    values = samples["values"]
    target_points = samples["target_points"]
    pixel_scale = float(samples["pixel_scale"])
    interpolator = interpolate.RBFInterpolator(points, values, kernel="linear", neighbors=max(8, neighbours), smoothing=0.0)
    raw = interpolator(target_points).reshape(output_shape)
    confidence, support = adaptive_confidence(points, target_points, pixel_scale, max_grid_distance, neighbours=neighbours)
    confidence_map = confidence.reshape(output_shape)
    support_map = support.reshape(output_shape)
    source = np.where(support_map, np.clip(raw, 0.0, None) * confidence_map, 0.0)
    source, scale = normalise_physical(source)
    stats = {
        "confidence_mean_supported": float(np.mean(confidence_map[support_map])) if np.any(support_map) else 0.0,
        "confidence_scale_99p5": float(scale),
        "clipped_to_zero_fraction": float(np.mean((raw < 0.0) & support_map)),
    }
    return source, support_map, stats


def adaptive_local_weighted(samples: dict[str, object], output_shape: tuple[int, int], max_grid_distance: float, neighbours: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    points = samples["points"]
    values = samples["values"]
    target_points = samples["target_points"]
    pixel_scale = float(samples["pixel_scale"])
    tree = cKDTree(points)
    k = min(max(int(neighbours), 8), len(points))
    distances, indices = tree.query(target_points, k=k)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    local_radius = np.maximum(np.percentile(distances, 50.0, axis=1), 1.5 * pixel_scale)
    local_radius = np.minimum(local_radius, max_grid_distance * pixel_scale)
    usable = distances <= local_radius[:, None]
    close_count = np.sum(usable, axis=1)
    support = (distances[:, 0] <= max_grid_distance * pixel_scale) & (close_count >= 4)
    weights = np.where(usable, 1.0 / np.maximum(distances, 0.35 * pixel_scale) ** 2, 0.0)
    numerator = np.sum(weights * values[indices], axis=1)
    denominator = np.sum(weights, axis=1)
    interpolated = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    source = np.where(support, np.clip(interpolated, 0.0, None), 0.0).reshape(output_shape)
    source, scale = normalise_physical(source)
    support_map = support.reshape(output_shape)
    stats = {
        "confidence_mean_supported": float(np.mean(np.minimum(1.0, close_count[support] / 8.0))) if np.any(support) else 0.0,
        "confidence_scale_99p5": float(scale),
        "clipped_to_zero_fraction": 0.0,
    }
    return source, support_map, stats


def method_row(
    case_id: str,
    method: str,
    source: np.ndarray,
    support: np.ndarray,
    truth: np.ndarray,
    observed: np.ndarray,
    reference_result: RobustReconstructionResult,
    metadata: dict[str, float | str],
    delta_pix: float,
    psf: np.ndarray,
    runtime: float,
    fractions: dict[str, object],
    extra_stats: dict[str, float] | None = None,
) -> dict[str, object]:
    result = RobustReconstructionResult(
        source=np.clip(source, 0.0, None),
        coverage=support.astype(float),
        valid_mask=reference_result.valid_mask,
        beta_x=reference_result.beta_x,
        beta_y=reference_result.beta_y,
        grid=reference_result.grid,
        stats={},
    )
    predicted = forward_lens_reconstruction(result.source, result.grid, metadata, observed.shape, delta_pix, psf)
    row: dict[str, object] = {
        "id": case_id,
        "method": method,
        "runtime_seconds": runtime,
        **reconstruction_quality_metrics(truth, result.source),
        **morphology_errors(truth, result.source),
        **forward_metrics(observed, predicted),
        **artefact_metrics(result.source, support),
        **fractions,
    }
    if extra_stats:
        row.update(extra_stats)
    return row


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
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
        "runtime_seconds",
        "direct_supported_fraction",
        "interpolated_fraction",
        "extrapolated_fraction",
        "clipped_to_zero_fraction",
        "artefact_dark_hole_fraction",
        "artefact_component_count",
        "artefact_high_frequency_energy",
    ]
    out = []
    for method, method_rows in grouped.items():
        item: dict[str, object] = {"method": method, "count": len(method_rows)}
        for metric in metrics:
            values = np.array([finite_float(row.get(metric)) for row in method_rows], dtype=float)
            values = values[np.isfinite(values)]
            item[f"mean_{metric}"] = float(np.mean(values)) if len(values) else math.nan
            item[f"median_{metric}"] = float(np.median(values)) if len(values) else math.nan
            item[f"std_{metric}"] = float(np.std(values)) if len(values) else math.nan
        out.append(item)
    out.sort(key=lambda row: (finite_float(row.get("median_ssim_aligned")), finite_float(row.get("median_ncc_aligned"))), reverse=True)
    return out


def select_method(summary: list[dict[str, object]], baseline_method: str) -> dict[str, object]:
    baseline = next((row for row in summary if row["method"] == baseline_method), None)
    if baseline is None:
        return {"selected_method": summary[0]["method"] if summary else "", "reason": "baseline_missing"}
    candidates = []
    for row in summary:
        method = str(row["method"])
        ssim_delta = finite_float(row.get("median_ssim_aligned")) - finite_float(baseline.get("median_ssim_aligned"))
        ncc_delta = finite_float(row.get("median_ncc_aligned")) - finite_float(baseline.get("median_ncc_aligned"))
        centroid_delta = finite_float(row.get("median_centroid_error")) - finite_float(baseline.get("median_centroid_error"))
        size_delta = finite_float(row.get("median_effective_radius_error")) - finite_float(baseline.get("median_effective_radius_error"))
        artefact_delta = finite_float(row.get("median_artefact_high_frequency_energy")) - finite_float(baseline.get("median_artefact_high_frequency_energy"))
        acceptable = (ssim_delta >= -0.002) and (ncc_delta >= -0.002) and (centroid_delta <= 1.5) and (size_delta <= 1.5)
        score = ssim_delta + ncc_delta - max(artefact_delta, 0.0)
        candidates.append((acceptable, score, row, ssim_delta, ncc_delta, artefact_delta))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best = candidates[0]
    if best[0]:
        return {
            "selected_method": best[2]["method"],
            "baseline_method": baseline_method,
            "reason": "best method meeting SSIM/NCC and morphology-preservation acceptance rule",
            "median_ssim_delta_vs_baseline": best[3],
            "median_ncc_delta_vs_baseline": best[4],
            "median_artifact_energy_delta_vs_baseline": best[5],
        }
    return {
        "selected_method": baseline_method,
        "baseline_method": baseline_method,
        "reason": "no targeted sparse-coverage fix met the acceptance rule",
    }


def panel(path: Path, case_id: str, observed: np.ndarray, truth: np.ndarray, images: dict[str, np.ndarray], selected_metric: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = max([float(np.max(truth))] + [float(np.max(image)) for image in images.values()] + [1e-10])
    panels = [("observed lens", stretch(observed))]
    for name, image in images.items():
        panels.append((name, np.clip(image / scale, 0.0, 1.0)))
    panels.append(("true source", np.clip(truth / scale, 0.0, 1.0)))
    tile = 140
    label_h = 24
    footer_h = 24
    canvas = Image.new("RGB", (tile * len(panels), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (label, arr) in enumerate(panels):
        x = idx * tile
        draw.text((x + 4, 4), label[:22], fill="black")
        img = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").convert("RGB")
        canvas.paste(img.resize((tile, tile), Image.Resampling.BICUBIC), (x, label_h))
    draw.text((4, tile + label_h + 4), f"{case_id} | {selected_metric}"[:130], fill="black")
    canvas.save(path)


def save_galleries(out_dir: Path, rows: list[dict[str, object]], records: dict[str, dict[str, object]], selected_method: str) -> None:
    selected = [row for row in rows if row["method"] == selected_method]
    selected.sort(key=lambda row: finite_float(row.get("ssim_aligned")), reverse=True)
    groups = {
        "best_examples": selected[:5],
        "worst_examples": selected[-5:][::-1],
    }
    mid = len(selected) // 2
    groups["median_examples"] = selected[max(0, mid - 2) : min(len(selected), mid + 3)]
    for folder, group in groups.items():
        for idx, row in enumerate(group, start=1):
            record = records[str(row["id"])]
            panel(
                out_dir / folder / f"{idx:02d}_{row['id']}_{selected_method}.png",
                str(row["id"]),
                record["observed"],
                record["truth"],
                record["images"],
                f"{selected_method}: SSIM={finite_float(row.get('ssim_aligned')):.3f}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare targeted sparse-beta-coverage reconstruction fixes.")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data" / "cosmos_sources")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "sparse_coverage_targeted_fixes")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=25015168)
    parser.add_argument("--source-size", type=int, default=128)
    parser.add_argument("--num-pix", type=int, default=128)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--source-extent", type=float, default=0.62)
    parser.add_argument("--noise-sigma", type=float, default=0.018)
    parser.add_argument("--psf-fwhm", type=float, default=0.08)
    parser.add_argument("--rl-iterations", type=int, default=15)
    parser.add_argument("--rbf-neighbors", type=int, default=48)
    parser.add_argument("--adaptive-neighbors", type=int, default=12)
    parser.add_argument("--scattered-max-grid-distance", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = sorted((args.source_dir / "sources_npy").glob("*.npy"))
    if not source_paths:
        raise SystemExit(f"No source .npy files found in {args.source_dir / 'sources_npy'}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    psf = gaussian_psf_kernel(args.psf_fwhm, args.delta_pix)
    rows: list[dict[str, object]] = []
    records: dict[str, dict[str, object]] = {}
    baseline_method = "rl15_rbf_thin_plate_spline"

    for index, source_path in enumerate(source_paths[: args.count]):
        source = np.load(source_path).astype(float)
        source = make_compact_source(augment_source(source, rng), rng, args.source_size, 0.34, 0.58)
        case_id = f"sparse_fix_{index:05d}"
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

        method_results: dict[str, np.ndarray] = {}
        reference_result: RobustReconstructionResult | None = None
        truth = None
        for method_name, robust_method in [
            (baseline_method, "rbf_thin_plate_spline_aggregated"),
            ("rl15_rbf_linear", "rbf_linear_aggregated"),
        ]:
            start = time.perf_counter()
            result = robust_source_reconstruction(
                deconvolved,
                true_mask,
                metadata,  # type: ignore[arg-type]
                args.delta_pix,
                valid_pixel_mode="arc",
                config=config_for(args, robust_method),
            )
            runtime = time.perf_counter() - start
            if reference_result is None:
                reference_result = result
                truth = sample_truth_on_grid(source, result.grid, args.source_extent)
            samples = extract_samples(result, deconvolved)
            fractions = support_fractions(
                samples["points"],
                samples["direct_coverage"],
                samples["target_points"],
                float(samples["pixel_scale"]),
                args.scattered_max_grid_distance,
            )
            support = fractions["support"]
            clipped = float(np.mean((result.source <= 0.0) & support))
            row = method_row(
                case_id,
                method_name,
                result.source,
                support,
                truth,
                observed,
                result,
                metadata,
                args.delta_pix,
                psf,
                runtime,
                {k: v for k, v in fractions.items() if not isinstance(v, np.ndarray)},
                {"clipped_to_zero_fraction": clipped},
            )
            rows.append(row)
            method_results[method_name] = result.source

        assert reference_result is not None and truth is not None
        samples = extract_samples(reference_result, deconvolved)
        base_fractions = support_fractions(
            samples["points"],
            samples["direct_coverage"],
            samples["target_points"],
            float(samples["pixel_scale"]),
            args.scattered_max_grid_distance,
        )
        for method_name, builder in [
            ("adaptive_grid_rbf_linear_confidence", rbf_linear_confidence_taper),
            ("adaptive_grid_local_weighted", adaptive_local_weighted),
        ]:
            start = time.perf_counter()
            adaptive_source, adaptive_support, extra = builder(
                samples,
                reference_result.source.shape,
                args.scattered_max_grid_distance,
                args.adaptive_neighbors,
            )
            runtime = time.perf_counter() - start
            fractions = {k: v for k, v in base_fractions.items() if not isinstance(v, np.ndarray)}
            fractions["support_masked_fraction"] = float(np.mean(~adaptive_support))
            rows.append(
                method_row(
                    case_id,
                    method_name,
                    adaptive_source,
                    adaptive_support,
                    truth,
                    observed,
                    reference_result,
                    metadata,
                    args.delta_pix,
                    psf,
                    runtime,
                    fractions,
                    extra,
                )
            )
            method_results[method_name] = adaptive_source

        records[case_id] = {"observed": observed, "truth": truth, "images": method_results}
        if (index + 1) % 25 == 0:
            print(f"Processed {index + 1}/{min(args.count, len(source_paths))}")

    summary = summarize(rows)
    decision = select_method(summary, baseline_method)
    write_csv(args.out_dir / "per_image_metrics.csv", rows)
    write_csv(args.out_dir / "method_comparison.csv", summary)
    (args.out_dir / "selected_method.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    save_galleries(args.out_dir, rows, records, str(decision["selected_method"]))

    labels = [str(row["method"]) for row in summary]
    vals = [finite_float(row.get("median_ssim_aligned")) for row in summary]
    plt.figure(figsize=(8.5, 4.8), dpi=180)
    plt.bar(labels, vals)
    plt.ylabel("Median aligned SSIM")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(args.out_dir / "median_ssim_by_method.png")
    plt.close()

    report = [
        "# Sparse Coverage Targeted Fixes",
        "",
        f"Selected method: `{decision['selected_method']}`",
        "",
        f"Decision: {decision['reason']}",
        "",
        "Compared methods:",
        "- RL15 + RBF thin-plate spline baseline",
        "- RL15 + RBF linear",
        "- adaptive confidence-tapered RBF linear",
        "- adaptive local weighted interpolation",
        "",
        "No detector, lens parameter, or classifier logic was changed.",
    ]
    (args.out_dir / "SPARSE_COVERAGE_FIX_DECISION.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Saved targeted sparse-coverage comparison to {args.out_dir}")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
