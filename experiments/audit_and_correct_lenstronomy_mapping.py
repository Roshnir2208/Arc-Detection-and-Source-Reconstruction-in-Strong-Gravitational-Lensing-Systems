from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage, spatial
from scipy.spatial import cKDTree

from lensing_pipeline.ellipse import extract_arc_parameters
from lensing_pipeline.metrics import match_photometric_scale
from lensing_pipeline.reconstruction import approximate_sie_ray_shooting
from lensing_pipeline.robust_reconstruction import (
    image_pixels_to_angles,
    lenstronomy_sie_ray_shooting,
    map_to_source_plane,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
)
from reconstruct_lenstronomy_final_64 import config_for, stretch
from run_interim_inverse_reconstruction_benchmark import ellipse_parameters_to_sie_metadata
from run_reconstruction_improvement_ablation import forward_lens_reconstruction
from run_support_tv_reconstruction_validation import gaussian_psf_kernel, richardson_lucy


def read_metadata(path: Path) -> dict[str, dict[str, float]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["id"]: {key: float(value) for key, value in row.items() if key != "id"} for row in rows}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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
        number = float(value)  # type: ignore[arg-type]
    except Exception:
        return default
    return number if math.isfinite(number) else default


def representative_ids(metrics: list[dict[str, str]], requested: list[str], per_group: int) -> list[tuple[str, str]]:
    rows = [row for row in metrics if row.get("id") and math.isfinite(finite_float(row.get("ssim_aligned")))]
    rows.sort(key=lambda row: finite_float(row["ssim_aligned"]), reverse=True)
    out: list[tuple[str, str]] = [("requested", image_id) for image_id in requested]
    if rows:
        out.extend(("best", row["id"]) for row in rows[:per_group])
        mid = max(0, len(rows) // 2 - per_group // 2)
        out.extend(("median", row["id"]) for row in rows[mid : mid + per_group])
        out.extend(("worst", row["id"]) for row in rows[-per_group:])
    seen: set[str] = set()
    dedup: list[tuple[str, str]] = []
    for group, image_id in out:
        if image_id not in seen:
            seen.add(image_id)
            dedup.append((group, image_id))
    return dedup


def sector_labels(xx: np.ndarray, yy: np.ndarray, image_shape: tuple[int, int], sectors: int = 8) -> np.ndarray:
    cx = (image_shape[1] - 1) / 2.0
    cy = (image_shape[0] - 1) / 2.0
    angle = np.arctan2(yy.astype(float) - cy, xx.astype(float) - cx)
    return np.clip(np.floor(((angle + np.pi) / (2.0 * np.pi)) * sectors).astype(int), 0, sectors - 1)


def mapping_stats(beta_x: np.ndarray, beta_y: np.ndarray, values: np.ndarray, sectors: np.ndarray) -> dict[str, float]:
    points = np.column_stack([beta_x, beta_y])
    total_centroid = np.average(points, axis=0, weights=np.clip(values, 0, None) + 1e-8) if len(points) else np.zeros(2)
    total_scatter = float(np.sqrt(np.mean(np.sum((points - total_centroid) ** 2, axis=1)))) if len(points) else 0.0
    centroids: list[np.ndarray] = []
    within: list[float] = []
    for sector in sorted(set(sectors.tolist())):
        select = sectors == sector
        if np.count_nonzero(select) < 4:
            continue
        weights = np.clip(values[select], 0, None) + 1e-8
        centroid = np.average(points[select], axis=0, weights=weights)
        centroids.append(centroid)
        within.append(float(np.sqrt(np.mean(np.sum((points[select] - centroid) ** 2, axis=1)))))
    if len(centroids) > 1:
        dists = spatial.distance.pdist(np.vstack(centroids))
        max_sep = float(np.max(dists))
        between = float(np.sqrt(np.mean(np.sum((np.vstack(centroids) - np.mean(np.vstack(centroids), axis=0)) ** 2, axis=1))))
    else:
        max_sep = 0.0
        between = 0.0
    try:
        hull_area = float(spatial.ConvexHull(points).volume) if len(points) >= 4 else 0.0
    except Exception:
        hull_area = 0.0
    if len(points) >= 2:
        nearest, _ = cKDTree(points).query(points, k=2)
        nn = nearest[:, 1]
    else:
        nn = np.array([0.0])
    out = {
        "sector_count": float(len(centroids)),
        "flux_weighted_beta_x": float(total_centroid[0]),
        "flux_weighted_beta_y": float(total_centroid[1]),
        "total_beta_scatter": total_scatter,
        "mean_within_sector_scatter": float(np.mean(within)) if within else 0.0,
        "between_sector_scatter": between,
        "max_sector_centroid_separation": max_sep,
        "sector_collapse_metric": between / max(total_scatter, 1e-12),
        "convex_hull_area": hull_area,
        "nearest_spacing_p50": float(np.percentile(nn, 50)),
        "nearest_spacing_p95": float(np.percentile(nn, 95)),
    }
    for axis, arr in [("x", beta_x), ("y", beta_y)]:
        out[f"beta_{axis}_p05"] = float(np.percentile(arr, 5)) if len(arr) else math.nan
        out[f"beta_{axis}_p50"] = float(np.percentile(arr, 50)) if len(arr) else math.nan
        out[f"beta_{axis}_p95"] = float(np.percentile(arr, 95)) if len(arr) else math.nan
    return out


def exact_lens_kwargs(metadata: dict[str, float]) -> dict[str, float]:
    return {
        "theta_E": float(metadata["theta_E"]),
        "e1": float(metadata["lens_e1"]),
        "e2": float(metadata["lens_e2"]),
        "center_x": float(metadata.get("lens_center_x", 0.0)),
        "center_y": float(metadata.get("lens_center_y", 0.0)),
    }


def config_with(ray_tracer: str, args: argparse.Namespace):
    cfg_args = SimpleNamespace(
        source_size=args.source_size,
        source_extent=args.source_extent,
        scattered_max_grid_distance=args.scattered_max_grid_distance,
        rbf_neighbors=args.rbf_neighbors,
        ray_tracer=ray_tracer,
    )
    return config_for(cfg_args)


def save_mapping_comparison(
    path: Path,
    image: np.ndarray,
    mappings: dict[str, tuple[np.ndarray, np.ndarray]],
    sectors: np.ndarray,
    exact_source: np.ndarray,
    current_source: np.ndarray,
    truth: np.ndarray,
    exact_grid,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].imshow(stretch(image), cmap="gray", origin="upper")
    axes[0].set_title("observed lens")
    axes[0].axis("off")
    for idx, name in enumerate(["exact_lenstronomy", "current_custom", "ellipse_derived"], start=1):
        bx, by = mappings[name]
        sc = axes[idx].scatter(bx, by, c=sectors, s=4, cmap="tab10", alpha=0.7, linewidths=0)
        axes[idx].set_title(name.replace("_", " "))
        axes[idx].set_aspect("equal", adjustable="box")
        fig.colorbar(sc, ax=axes[idx], fraction=0.046)
    scale = max(float(np.percentile(np.concatenate([truth.ravel(), exact_source.ravel(), current_source.ravel()]), 99.5)), 1e-12)
    panels = [
        ("exact reconstruction", exact_source),
        ("current reconstruction", current_source),
        ("true source", truth),
        ("exact residual", truth - match_photometric_scale(truth, exact_source)),
    ]
    for ax, (title, arr) in zip(axes[4:], panels):
        if "residual" in title:
            m = max(float(np.max(np.abs(arr))), 1e-12)
            ax.imshow(np.clip(0.5 + 0.5 * arr / m, 0, 1), cmap="gray", origin="upper")
        else:
            ax.imshow(np.clip(arr / scale, 0, 1), cmap="gray", origin="upper")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_coordinate_table(
    rows: list[dict[str, object]],
    image_id: str,
    yy: np.ndarray,
    xx: np.ndarray,
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    sectors: np.ndarray,
    values: np.ndarray,
    count: int = 16,
) -> None:
    if len(xx) == 0:
        return
    chosen: list[int] = []
    for sector in sorted(set(sectors.tolist())):
        candidates = np.where(sectors == sector)[0]
        if len(candidates):
            chosen.append(int(candidates[np.argmax(values[candidates])]))
    chosen = chosen[:count]
    for idx in chosen:
        rows.append(
            {
                "id": image_id,
                "sector": int(sectors[idx]),
                "row": int(yy[idx]),
                "col": int(xx[idx]),
                "theta_x": float(theta_x[idx]),
                "theta_y": float(theta_y[idx]),
                "alpha_x_exact": float(theta_x[idx] - beta_x[idx]),
                "alpha_y_exact": float(theta_y[idx] - beta_y[idx]),
                "beta_x_exact": float(beta_x[idx]),
                "beta_y_exact": float(beta_y[idx]),
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact Lenstronomy beta mapping against the current custom helper.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "lenstronomy_recon_1000")
    parser.add_argument("--reconstruction-dir", type=Path, default=ROOT / "outputs" / "lenstronomy_final_64_reconstruction_1000")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "lenstronomy_exact_mapping_audit")
    parser.add_argument("--ids", nargs="*", default=["lens_00000"])
    parser.add_argument("--per-group", type=int, default=2)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--rl-iterations", type=int, default=15)
    parser.add_argument("--source-size", type=int, default=64)
    parser.add_argument("--source-extent", type=float, default=0.62)
    parser.add_argument("--rbf-neighbors", type=int, default=48)
    parser.add_argument("--scattered-max-grid-distance", type=float, default=3.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata = read_metadata(args.dataset / "metadata.csv")
    metrics = read_csv(args.reconstruction_dir / "reconstruction_64_per_image_metrics.csv")
    selected = representative_ids(metrics, args.ids, args.per_group)
    psf_fwhm = float(next(iter(metadata.values())).get("psf_fwhm", 0.08)) if metadata else 0.08
    psf = gaussian_psf_kernel(psf_fwhm, args.delta_pix)

    rows: list[dict[str, object]] = []
    pixel_rows: list[dict[str, object]] = []
    for group, image_id in selected:
        if image_id not in metadata:
            continue
        image = np.load(args.dataset / "images_npy" / f"{image_id}.npy")
        mask = np.load(args.dataset / "masks_npy" / f"{image_id}.npy").astype(bool)
        truth_native = np.load(args.dataset / "sources_npy" / f"{image_id}.npy")
        deconvolved = richardson_lucy(image, psf, args.rl_iterations)
        arc = extract_arc_parameters(mask, image_shape=image.shape)
        ellipse_meta = ellipse_parameters_to_sie_metadata(arc, image.shape, args.delta_pix)

        yy, xx = np.nonzero(mask)
        theta_x, theta_y = image_pixels_to_angles(xx, yy, mask.shape, args.delta_pix)
        exact_beta_x, exact_beta_y = lenstronomy_sie_ray_shooting(theta_x, theta_y, metadata[image_id])
        custom_beta_x, custom_beta_y = approximate_sie_ray_shooting(theta_x, theta_y, metadata[image_id])
        ellipse_beta_x, ellipse_beta_y = approximate_sie_ray_shooting(theta_x, theta_y, ellipse_meta)
        values = deconvolved[yy, xx]
        sectors = sector_labels(xx, yy, image.shape)
        exact = robust_source_reconstruction(deconvolved, mask, metadata[image_id], args.delta_pix, valid_pixel_mode="arc", config=config_with("lenstronomy", args))
        current = robust_source_reconstruction(deconvolved, mask, metadata[image_id], args.delta_pix, valid_pixel_mode="arc", config=config_with("approximate", args))
        ellipse = robust_source_reconstruction(deconvolved, mask, ellipse_meta, args.delta_pix, valid_pixel_mode="arc", config=config_with("approximate", args))
        truth_exact = sample_truth_on_grid(truth_native, exact.grid, args.source_extent)
        truth_current = sample_truth_on_grid(truth_native, current.grid, args.source_extent)
        save_coordinate_table(pixel_rows, image_id, yy, xx, theta_x, theta_y, exact_beta_x, exact_beta_y, sectors, values)

        exact_stats = mapping_stats(exact_beta_x, exact_beta_y, values, sectors)
        custom_stats = mapping_stats(custom_beta_x, custom_beta_y, values, sectors)
        ellipse_stats = mapping_stats(ellipse_beta_x, ellipse_beta_y, values, sectors)
        beta_diff = np.hypot(exact_beta_x - custom_beta_x, exact_beta_y - custom_beta_y)
        lens_center_from_mask_x = (float(np.mean(xx)) - (image.shape[1] - 1) / 2.0) * args.delta_pix
        lens_center_from_mask_y = (float(np.mean(yy)) - (image.shape[0] - 1) / 2.0) * args.delta_pix
        exact_quality = reconstruction_quality_metrics(truth_exact, exact.source)
        current_quality = reconstruction_quality_metrics(truth_current, current.source)
        exact_relensed = forward_lens_reconstruction(exact.source, exact.grid, metadata[image_id], image.shape, args.delta_pix, psf)
        current_relensed = forward_lens_reconstruction(current.source, current.grid, metadata[image_id], image.shape, args.delta_pix, psf)
        exact_forward = float(np.mean((image - match_photometric_scale(image, exact_relensed)) ** 2))
        current_forward = float(np.mean((image - match_photometric_scale(image, current_relensed)) ** 2))

        row: dict[str, object] = {
            "id": image_id,
            "representative_group": group,
            "theta_E_true": float(metadata[image_id]["theta_E"]),
            "true_lens_e1": float(metadata[image_id]["lens_e1"]),
            "true_lens_e2": float(metadata[image_id]["lens_e2"]),
            "ellipse_lens_e1": float(ellipse_meta["lens_e1"]),
            "ellipse_lens_e2": float(ellipse_meta["lens_e2"]),
            "true_lens_center_x": float(metadata[image_id]["lens_center_x"]),
            "true_lens_center_y": float(metadata[image_id]["lens_center_y"]),
            "image_center_x_arcsec": 0.0,
            "image_center_y_arcsec": 0.0,
            "mask_centroid_x_arcsec": lens_center_from_mask_x,
            "mask_centroid_y_arcsec": lens_center_from_mask_y,
            "ellipse_center_x_arcsec": float(ellipse_meta["lens_center_x"]),
            "ellipse_center_y_arcsec": float(ellipse_meta["lens_center_y"]),
            "exact_kwargs": json.dumps(exact_lens_kwargs(metadata[image_id])),
            "custom_vs_exact_beta_median_difference": float(np.median(beta_diff)),
            "custom_vs_exact_beta_p95_difference": float(np.percentile(beta_diff, 95)),
            "exact_ssim": exact_quality["ssim_aligned"],
            "current_custom_ssim": current_quality["ssim_aligned"],
            "exact_ncc": exact_quality["ncc_aligned"],
            "current_custom_ncc": current_quality["ncc_aligned"],
            "exact_forward_mse": exact_forward,
            "current_custom_forward_mse": current_forward,
        }
        for prefix, stats in [("exact", exact_stats), ("current_custom", custom_stats), ("ellipse", ellipse_stats)]:
            row.update({f"{prefix}_{key}": value for key, value in stats.items()})
        rows.append(row)

        save_mapping_comparison(
            args.out_dir / f"{image_id}_exact_vs_custom_mapping.png",
            image,
            {
                "exact_lenstronomy": (exact_beta_x, exact_beta_y),
                "current_custom": (custom_beta_x, custom_beta_y),
                "ellipse_derived": (ellipse_beta_x, ellipse_beta_y),
            },
            sectors,
            exact.source,
            current.source,
            truth_exact,
            exact.grid,
        )

    write_csv(args.out_dir / "lens_mapping_audit_metrics.csv", rows)
    write_csv(args.out_dir / "lens_mapping_pixel_trace.csv", pixel_rows)
    if rows:
        exact_collapse = np.array([finite_float(row["exact_sector_collapse_metric"]) for row in rows])
        custom_collapse = np.array([finite_float(row["current_custom_sector_collapse_metric"]) for row in rows])
        beta_p95 = np.array([finite_float(row["custom_vs_exact_beta_p95_difference"]) for row in rows])
        exact_better = int(np.sum(exact_collapse < custom_collapse))
        root_cause = (
            "The current known-parameter reconstruction used a custom approximate SIE helper rather than Lenstronomy's exact SIE ray shooting. "
            "The exact and custom beta coordinates differ materially, so different arc sectors can be sent to separated source-plane regions."
            if float(np.nanmedian(beta_p95)) > 1e-3
            else "No large exact-vs-custom beta difference was found; remaining issues are likely parameter/source-grid related."
        )
        summary = {
            "count": len(rows),
            "median_exact_sector_collapse_metric": float(np.nanmedian(exact_collapse)),
            "median_current_custom_sector_collapse_metric": float(np.nanmedian(custom_collapse)),
            "exact_lower_collapse_count": exact_better,
            "median_custom_vs_exact_beta_p95_difference_arcsec": float(np.nanmedian(beta_p95)),
            "root_cause": root_cause,
            "final_pipeline_structure": {
                "known_parameter_oracle": "Lenstronomy SIE ray_shooting with stored generating theta_E/e1/e2/center_x/center_y",
                "estimated_automatic": "Ellipse-derived geometric approximation; not exact physical SIE inference",
            },
        }
    else:
        summary = {"count": 0, "root_cause": "No systems audited."}
    (args.out_dir / "lens_mapping_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Lens Mapping Audit",
        "",
        "## Root Cause",
        str(summary["root_cause"]),
        "",
        "## Coordinate-System Checks",
        "- Pixel coordinates are NumPy row/column indices.",
        "- Angular coordinates are computed as theta_x = (col - (width - 1)/2) * deltaPix and theta_y = (row - (height - 1)/2) * deltaPix.",
        "- The oracle experiment does not substitute the image centre, mask centroid, or ellipse centre for the true generating lens centre.",
        "- Exact Lenstronomy kwargs use theta_E, e1, e2, center_x, center_y, matching the synthetic generator.",
        "",
        "## Parameter-Convention Checks",
        "- Generating lens ellipticity is stored directly as lens_e1/lens_e2.",
        "- Ellipse-derived e1/e2 is only a geometric approximation from detected arc axis ratio and orientation.",
        "- The ellipse-derived model must not be described as exact physical SIE inference.",
        "",
        "## Sector-Collapse Statistics",
    ]
    if rows:
        lines.extend(
            [
                f"- Median exact sector-collapse metric: {summary['median_exact_sector_collapse_metric']:.6f}",
                f"- Median current-custom sector-collapse metric: {summary['median_current_custom_sector_collapse_metric']:.6f}",
                f"- Exact mapping has lower sector-collapse metric in {summary['exact_lower_collapse_count']} / {summary['count']} audited systems.",
                f"- Median 95th-percentile custom-vs-exact beta difference: {summary['median_custom_vs_exact_beta_p95_difference_arcsec']:.6f} arcsec",
            ]
        )
    lines.extend(
        [
            "",
            "## Before/After Reconstruction Examples",
            "The files `*_exact_vs_custom_mapping.png` show: observed lens, beta clouds for exact/current/ellipse mappings, exact reconstruction, current reconstruction, true source, and exact residual.",
            "",
            "## Final Pipeline Structure",
            "- Known-parameter/oracle reconstruction: exact Lenstronomy SIE ray shooting using stored simulation parameters.",
            "- Estimated/automatic reconstruction: ellipse-derived geometric approximation from detected arcs.",
            "- Classifiers should not be rerun until final reconstructed sources are regenerated with the selected mapping mode.",
        ]
    )
    (args.out_dir / "LENS_MAPPING_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved lens mapping audit to {args.out_dir}")


if __name__ == "__main__":
    main()
