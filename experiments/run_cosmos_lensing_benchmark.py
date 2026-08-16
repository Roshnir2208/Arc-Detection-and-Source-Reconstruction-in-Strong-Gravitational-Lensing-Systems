from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from lensing_pipeline.detection import detect_arc_mask, normalize_image
from lensing_pipeline.metrics import match_photometric_scale, psnr, segmentation_metrics, ssim_simple
from lensing_pipeline.reconstruction import (
    approximate_sie_ray_shooting,
    fit_forward_lensed_sersic_source,
    reconstruct_source_with_lens_model,
)
from lensing_pipeline.robust_reconstruction import (
    RobustReconstructionConfig,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    sample_truth_on_grid,
)
from lensing_pipeline.visualization import save_grayscale_png


def ellipticity_from_q_phi(axis_ratio: float, angle: float) -> tuple[float, float]:
    q = float(np.clip(axis_ratio, 0.2, 1.0))
    ellipticity = (1.0 - q) / (1.0 + q)
    return float(ellipticity * np.cos(2.0 * angle)), float(ellipticity * np.sin(2.0 * angle))


def sample_source(source: np.ndarray, beta_x: np.ndarray, beta_y: np.ndarray, source_extent: float) -> np.ndarray:
    size = source.shape[0]
    sx = (beta_x + source_extent) / (2.0 * source_extent) * (size - 1)
    sy = (beta_y + source_extent) / (2.0 * source_extent) * (size - 1)
    return ndimage.map_coordinates(source, [sy, sx], order=1, mode="constant", cval=0.0)


def augment_source(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    angle = float(rng.uniform(0.0, 360.0))
    rotated = ndimage.rotate(source, angle=angle, reshape=False, order=1, mode="constant", cval=0.0)
    zoom = float(rng.uniform(0.72, 1.12))
    zoomed = ndimage.zoom(rotated, zoom=zoom, order=1)
    out = np.zeros_like(source)
    h = min(out.shape[0], zoomed.shape[0])
    w = min(out.shape[1], zoomed.shape[1])
    y_src = max((zoomed.shape[0] - h) // 2, 0)
    x_src = max((zoomed.shape[1] - w) // 2, 0)
    y_dst = max((out.shape[0] - h) // 2, 0)
    x_dst = max((out.shape[1] - w) // 2, 0)
    out[y_dst : y_dst + h, x_dst : x_dst + w] = zoomed[y_src : y_src + h, x_src : x_src + w]
    shift = rng.uniform(-0.08 * source.shape[0], 0.08 * source.shape[0], size=2)
    out = ndimage.shift(out, shift=shift, order=1, mode="constant", cval=0.0)
    return normalize_image(out)


def make_compact_source(
    source: np.ndarray,
    rng: np.random.Generator,
    output_size: int,
    min_fraction: float,
    max_fraction: float,
) -> np.ndarray:
    source = normalize_image(source)
    positive = source[source > 0]
    if len(positive) == 0:
        return np.zeros((output_size, output_size), dtype=float)

    threshold = max(float(np.percentile(positive, 58.0)), 0.02)
    active = source > threshold
    labels, component_count = ndimage.label(active)
    if component_count > 0:
        best_label = 0
        best_flux = -1.0
        for label in range(1, component_count + 1):
            flux = float(source[labels == label].sum())
            if flux > best_flux:
                best_flux = flux
                best_label = label
        active = ndimage.binary_dilation(labels == best_label, iterations=4)
    if np.count_nonzero(active) < 8:
        active = source > max(float(np.percentile(positive, 40.0)), 0.01)

    yy, xx = np.nonzero(active)
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    pad = max(4, int(0.18 * max(y1 - y0, x1 - x0)))
    y0 = max(0, y0 - pad)
    y1 = min(source.shape[0], y1 + pad)
    x0 = max(0, x0 - pad)
    x1 = min(source.shape[1], x1 + pad)
    crop = source[y0:y1, x0:x1]
    crop = normalize_image(crop)

    target_span = int(round(output_size * float(rng.uniform(min_fraction, max_fraction))))
    target_span = max(18, min(target_span, output_size - 12))
    scale = target_span / max(crop.shape)
    resized = ndimage.zoom(crop, zoom=scale, order=3)
    resized = normalize_image(resized)

    canvas = np.zeros((output_size, output_size), dtype=float)
    h = min(canvas.shape[0], resized.shape[0])
    w = min(canvas.shape[1], resized.shape[1])
    y_dst = max((output_size - h) // 2 + int(rng.normal(0.0, output_size * 0.025)), 0)
    x_dst = max((output_size - w) // 2 + int(rng.normal(0.0, output_size * 0.025)), 0)
    y_dst = min(y_dst, output_size - h)
    x_dst = min(x_dst, output_size - w)
    y_src = max((resized.shape[0] - h) // 2, 0)
    x_src = max((resized.shape[1] - w) // 2, 0)
    canvas[y_dst : y_dst + h, x_dst : x_dst + w] = resized[y_src : y_src + h, x_src : x_src + w]
    canvas = ndimage.gaussian_filter(canvas, sigma=0.35)
    return normalize_image(canvas)


def generate_case(
    source: np.ndarray,
    source_id: str,
    case_id: str,
    rng: np.random.Generator,
    num_pix: int,
    delta_pix: float,
    source_extent: float,
    noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | str]]:
    theta_e = float(rng.uniform(0.95, 1.35))
    lens_q = float(rng.uniform(0.58, 0.90))
    lens_angle = float(rng.uniform(0.0, np.pi))
    e1, e2 = ellipticity_from_q_phi(lens_q, lens_angle)
    metadata: dict[str, float | str] = {
        "id": case_id,
        "source_id": source_id,
        "theta_E": theta_e,
        "lens_e1": e1,
        "lens_e2": e2,
        "lens_center_x": float(rng.uniform(-0.025, 0.025)),
        "lens_center_y": float(rng.uniform(-0.025, 0.025)),
        "lens_axis_ratio": lens_q,
        "lens_angle_degrees": float(np.degrees(lens_angle)),
        "source_extent": source_extent,
    }
    coords = (np.arange(num_pix, dtype=float) - (num_pix - 1) / 2.0) * delta_pix
    image_x, image_y = np.meshgrid(coords, coords)
    beta_x, beta_y = approximate_sie_ray_shooting(image_x, image_y, metadata)  # type: ignore[arg-type]
    lensed_source = normalize_image(sample_source(source, beta_x, beta_y, source_extent))

    lens_radius = np.hypot(image_x - float(metadata["lens_center_x"]), image_y - float(metadata["lens_center_y"]))
    lens_light = np.exp(-np.power(np.maximum(lens_radius, 1e-4) / float(rng.uniform(0.20, 0.34)), 0.38))
    lens_light *= float(rng.uniform(0.08, 0.18))
    clean = normalize_image(lensed_source + lens_light)
    image = normalize_image(clean + rng.normal(0.0, noise_sigma, clean.shape))
    positive = lensed_source[lensed_source > 0]
    threshold = max(float(np.percentile(positive, 42.0)), 0.012) if len(positive) else 0.012
    mask = lensed_source > threshold
    return image, clean, lensed_source, mask, metadata


def stretch(image: np.ndarray, percentile: float = 99.5, gamma: float = 0.72) -> np.ndarray:
    arr = np.clip(np.asarray(image, dtype=float), 0.0, None)
    positive = arr[arr > 0]
    if len(positive):
        scale = float(np.percentile(positive, percentile))
        if scale > 0:
            arr = arr / scale
    return np.power(np.clip(arr, 0.0, 1.0), gamma)


def save_panel(
    case_id: str,
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    true_source: np.ndarray,
    reconstruction: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["lensed image", "true arc mask", "detected mask", "true COSMOS source", "reconstructed source"]
    arrays = [stretch(image), true_mask.astype(float), pred_mask.astype(float), stretch(true_source), stretch(reconstruction)]
    tile = 150
    label_h = 22
    footer_h = 18
    panel = Image.new("RGB", (tile * len(arrays), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, arr) in enumerate(zip(labels, arrays)):
        if "mask" in label:
            base = Image.fromarray((stretch(image) * 255).astype(np.uint8), mode="L").convert("RGBA")
            red = Image.new("RGBA", base.size, (255, 0, 0, 145))
            alpha = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            base.alpha_composite(Image.composite(red, Image.new("RGBA", base.size, (0, 0, 0, 0)), alpha))
            img = base.convert("RGB")
        else:
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
        x = idx * tile
        draw.text((x + 4, 4), label, fill="black")
        panel.paste(img.resize((tile, tile), Image.Resampling.BICUBIC), (x, label_h))
    draw.text((4, tile + label_h + 2), case_id, fill="black")
    panel.save(path)


def save_robust_panel(
    case_id: str,
    image: np.ndarray,
    true_mask: np.ndarray,
    pred_mask: np.ndarray,
    valid_mask: np.ndarray,
    coverage: np.ndarray,
    true_source_on_grid: np.ndarray,
    reconstruction: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = [
        "lensed image",
        "true mask",
        "detected mask",
        "valid pixels",
        "coverage map",
        "truth on grid",
        "raw reconstruction",
        "residual",
    ]
    residual = np.abs(match_photometric_scale(true_source_on_grid, reconstruction) - true_source_on_grid)
    arrays = [
        stretch(image),
        true_mask.astype(float),
        pred_mask.astype(float),
        valid_mask.astype(float),
        stretch(coverage),
        stretch(true_source_on_grid),
        stretch(reconstruction),
        stretch(residual),
    ]
    tile = 128
    label_h = 20
    footer_h = 18
    panel = Image.new("RGB", (tile * len(arrays), tile + label_h + footer_h), "white")
    draw = ImageDraw.Draw(panel)
    for idx, (label, arr) in enumerate(zip(labels, arrays)):
        if "mask" in label or label == "valid pixels":
            base = Image.fromarray((stretch(image) * 255).astype(np.uint8), mode="L").convert("RGBA")
            red = Image.new("RGBA", base.size, (255, 0, 0, 145))
            alpha = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
            base.alpha_composite(Image.composite(red, Image.new("RGBA", base.size, (0, 0, 0, 0)), alpha))
            img = base.convert("RGB")
        else:
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L").convert("RGB")
        x = idx * tile
        draw.text((x + 4, 4), label, fill="black")
        panel.paste(img.resize((tile, tile), Image.Resampling.BICUBIC), (x, label_h))
    draw.text((4, tile + label_h + 2), case_id, fill="black")
    panel.save(path)


def save_beta_scatter(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    values: np.ndarray,
    path: Path,
    size: int = 420,
    padding: int = 28,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(canvas)
    if len(beta_x) == 0:
        draw.text((12, 12), "no beta samples", fill="black")
        canvas.save(path)
        return
    x0, x1 = np.percentile(beta_x, [1.0, 99.0])
    y0, y1 = np.percentile(beta_y, [1.0, 99.0])
    if abs(x1 - x0) < 1e-10:
        x0 -= 1.0
        x1 += 1.0
    if abs(y1 - y0) < 1e-10:
        y0 -= 1.0
        y1 += 1.0
    vals = np.asarray(values, dtype=float)
    v0, v1 = np.percentile(vals, [5.0, 99.0]) if len(vals) else (0.0, 1.0)
    for bx, by, value in zip(beta_x, beta_y, vals):
        px = padding + int(np.clip((bx - x0) / (x1 - x0), 0.0, 1.0) * (size - 2 * padding))
        py = size - padding - int(np.clip((by - y0) / (y1 - y0), 0.0, 1.0) * (size - 2 * padding))
        t = float(np.clip((value - v0) / max(v1 - v0, 1e-10), 0.0, 1.0))
        colour = (int(255 * t), int(70 + 160 * t), int(255 * (1.0 - t)))
        draw.ellipse((px - 1, py - 1, px + 1, py + 1), fill=colour)
    draw.rectangle((padding, padding, size - padding, size - padding), outline="black", width=1)
    draw.text((8, 8), "beta_x vs beta_y; colour=image flux", fill="black")
    draw.text((8, size - 20), f"x p01={x0:.3g}, p99={x1:.3g}; y p01={y0:.3g}, p99={y1:.3g}", fill="black")
    canvas.save(path)


def save_mask_overlay(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base = Image.fromarray((stretch(image) * 255).astype(np.uint8), mode="L").convert("RGBA")
    red = Image.new("RGBA", base.size, (255, 0, 0, 145))
    alpha = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    base.alpha_composite(Image.composite(red, Image.new("RGBA", base.size, (0, 0, 0, 0)), alpha))
    base.convert("RGB").save(path)


def save_contact_sheet(paths: list[Path], out_path: Path, limit: int = 12) -> None:
    selected = paths[:limit]
    if not selected:
        return
    panels = [Image.open(path).convert("RGB") for path in selected]
    sheet = Image.new("RGB", (panels[0].width, panels[0].height * len(panels)), "white")
    for idx, panel in enumerate(panels):
        sheet.paste(panel, (0, idx * panel.height))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)


def enhance_reconstruction_display(
    source: np.ndarray,
    output_size: int,
    threshold_fraction: float = 0.08,
    padding_fraction: float = 0.45,
    gamma: float = 0.72,
) -> tuple[np.ndarray, dict[str, float | str]]:
    source = np.clip(np.asarray(source, dtype=float), 0.0, 1.0)
    if not np.any(source > 0):
        return np.zeros((output_size, output_size), dtype=float), {
            "source_display_enhanced": 0.0,
            "source_display_crop_size": "",
        }

    positive = source[source > 0]
    background = float(np.percentile(positive, 45.0)) if len(positive) else 0.0
    source = np.clip(source - background, 0.0, None)
    source = ndimage.median_filter(source, size=3)
    source = ndimage.gaussian_filter(source, sigma=0.55)
    if np.any(source > 0):
        scale = float(np.percentile(source[source > 0], 99.0))
        if scale > 0:
            source = np.clip(source / scale, 0.0, 1.0)

    active = source > max(float(source.max()) * threshold_fraction, 1e-8)
    labels, component_count = ndimage.label(active)
    if component_count > 0:
        component_scores: list[tuple[float, int]] = []
        for label in range(1, component_count + 1):
            component = labels == label
            if int(component.sum()) >= 5:
                component_scores.append((float(source[component].sum()), label))
        component_scores.sort(reverse=True)
        kept = np.zeros_like(active)
        for _, label in component_scores[:8]:
            kept |= labels == label
        if np.any(kept):
            active = ndimage.binary_dilation(kept, iterations=2)
            source = np.where(active, source, 0.0)

    if np.count_nonzero(active) < 8:
        active = source > 0
    yy, xx = np.nonzero(active)
    y0 = int(max(0, yy.min()))
    y1 = int(min(source.shape[0] - 1, yy.max()))
    x0 = int(max(0, xx.min()))
    x1 = int(min(source.shape[1] - 1, xx.max()))
    span = max(y1 - y0 + 1, x1 - x0 + 1, 8)
    padding = int(np.ceil(span * padding_fraction))
    cy = int(round((y0 + y1) / 2))
    cx = int(round((x0 + x1) / 2))
    half = int(np.ceil(span / 2 + padding))
    crop = source[max(0, cy - half) : min(source.shape[0], cy + half + 1), max(0, cx - half) : min(source.shape[1], cx + half + 1)]
    if crop.size == 0:
        crop = source

    rendered = ndimage.zoom(crop, (output_size / crop.shape[0], output_size / crop.shape[1]), order=3)
    rendered = rendered[:output_size, :output_size]
    if rendered.shape != (output_size, output_size):
        padded = np.zeros((output_size, output_size), dtype=float)
        padded[: rendered.shape[0], : rendered.shape[1]] = rendered
        rendered = padded
    rendered = ndimage.gaussian_filter(rendered, sigma=0.35)
    if np.any(rendered > 0):
        scale = float(np.percentile(rendered[rendered > 0], 99.3))
        if scale > 0:
            rendered = rendered / scale
    rendered = np.power(np.clip(rendered, 0.0, 1.0), gamma)
    return np.clip(rendered, 0.0, 1.0), {
        "source_display_enhanced": 1.0,
        "source_display_crop_size": float(max(crop.shape)),
    }


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["id"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark lensing reconstruction using real COSMOS galaxies as source truth.")
    parser.add_argument("--source-dir", type=Path, required=True, help="Prepared COSMOS source library with sources_npy.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "cosmos_lensing_benchmark")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=25015168)
    parser.add_argument("--num-pix", type=int, default=128)
    parser.add_argument("--source-size", type=int, default=192)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--source-extent", type=float, default=0.62)
    parser.add_argument("--noise-sigma", type=float, default=0.018)
    parser.add_argument("--save-images", type=int, default=24)
    parser.add_argument("--threshold-percentile", type=float, default=89.0)
    parser.add_argument(
        "--deposition",
        choices=["bilinear", "linear-grid", "regularized-inversion", "forward-sersic", "hybrid-forward-pixel", "robust-weighted"],
        default="linear-grid",
    )
    parser.add_argument("--source-smooth-radius", type=int, default=1)
    parser.add_argument("--mask-source", choices=["detector", "true", "source-positive"], default="detector")
    parser.add_argument("--image-source", choices=["noisy", "clean", "lensed-source"], default="noisy")
    parser.add_argument("--source-positive-threshold", type=float, default=0.002)
    parser.add_argument("--inversion-lambda", type=float, default=0.01)
    parser.add_argument("--inversion-regularization", choices=["gradient", "curvature", "zeroth"], default="gradient")
    parser.add_argument("--compact-source", action="store_true")
    parser.add_argument("--compact-min-fraction", type=float, default=0.22)
    parser.add_argument("--compact-max-fraction", type=float, default=0.40)
    parser.add_argument("--robust-valid-pixels", choices=["arc", "flux", "arc_and_flux", "arc_or_flux"], default="arc")
    parser.add_argument("--robust-flux-threshold", type=float, default=0.002)
    parser.add_argument("--robust-dilation-radius", type=int, default=0)
    parser.add_argument("--robust-auto-extent", action="store_true")
    parser.add_argument("--robust-fixed-extent", action="store_true")
    parser.add_argument("--robust-margin-fraction", type=float, default=0.18)
    parser.add_argument("--robust-bound-low-percentile", type=float, default=1.0)
    parser.add_argument("--robust-bound-high-percentile", type=float, default=99.0)
    parser.add_argument("--robust-hole-fill", choices=["none", "local", "nearest_local"], default="none")
    parser.add_argument("--robust-max-gap-pixels", type=float, default=3.0)
    parser.add_argument("--enhance-review-display", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = sorted((args.source_dir / "sources_npy").glob("*.npy"))
    if not source_paths:
        raise SystemExit(f"No prepared COSMOS sources found in {args.source_dir / 'sources_npy'}")
    rng = np.random.default_rng(args.seed)

    dirs = {
        "images": args.out_dir / "images_npy",
        "clean": args.out_dir / "clean_npy",
        "lensed": args.out_dir / "lensed_source_npy",
        "masks": args.out_dir / "masks_npy",
        "sources": args.out_dir / "sources_npy",
        "figures": args.out_dir / "figures",
        "coverage": args.out_dir / "coverage_png",
        "recon": args.out_dir / "sources_reconstructed_png",
        "panels": args.out_dir / "review_panels",
        "diagnostics": args.out_dir / "diagnostic_panels",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    metadata_rows: list[dict[str, float | str | int]] = []
    metric_rows: list[dict[str, float | str | int]] = []
    panel_paths: list[Path] = []
    for index in range(args.count):
        source_path = source_paths[int(rng.integers(0, len(source_paths)))]
        source_id = source_path.stem
        case_id = f"cosmos_lens_{index:05d}"
        source = np.load(source_path)
        source = normalize_image(augment_source(source, rng))
        if args.compact_source:
            source = make_compact_source(
                source,
                rng,
                output_size=args.source_size,
                min_fraction=args.compact_min_fraction,
                max_fraction=args.compact_max_fraction,
            )
        image, clean, lensed_source, true_mask, metadata = generate_case(
            source,
            source_id,
            case_id,
            rng,
            num_pix=args.num_pix,
            delta_pix=args.delta_pix,
            source_extent=args.source_extent,
            noise_sigma=args.noise_sigma,
        )
        np.save(dirs["images"] / f"{case_id}.npy", image.astype(np.float32))
        np.save(dirs["clean"] / f"{case_id}.npy", clean.astype(np.float32))
        np.save(dirs["lensed"] / f"{case_id}.npy", lensed_source.astype(np.float32))
        np.save(dirs["masks"] / f"{case_id}.npy", true_mask.astype(bool))
        np.save(dirs["sources"] / f"{case_id}.npy", source.astype(np.float32))
        save_mask_overlay(image, true_mask, dirs["figures"] / f"{case_id}_true_mask.png")

        if args.mask_source == "true":
            pred_mask = true_mask
        elif args.mask_source == "source-positive":
            pred_mask = lensed_source > args.source_positive_threshold
        else:
            pred_mask = detect_arc_mask(
                image,
                threshold_percentile=args.threshold_percentile,
                min_component_size=8,
                suppress_centre_radius=0.08,
            )
        reconstruction_input = {
            "noisy": image,
            "clean": clean,
            "lensed-source": lensed_source,
        }[args.image_source]
        forward_stats: dict[str, float | str] = {
            "forward_source_fit_used": 0.0,
            "forward_source_fit_status": "",
            "forward_fit_rms": "",
            "forward_fit_cost": "",
        }
        robust_result = None
        true_source_for_metrics = source
        if args.deposition == "robust-weighted":
            robust_config = RobustReconstructionConfig(
                output_size=args.source_size,
                source_extent=args.source_extent,
                auto_extent=not args.robust_fixed_extent,
                auto_margin_fraction=args.robust_margin_fraction,
                auto_bound_low_percentile=args.robust_bound_low_percentile,
                auto_bound_high_percentile=args.robust_bound_high_percentile,
                hole_fill=args.robust_hole_fill,
                max_interpolation_gap_pixels=args.robust_max_gap_pixels,
            )
            robust_result = robust_source_reconstruction(
                reconstruction_input,
                pred_mask,
                {key: float(value) for key, value in metadata.items() if key not in {"id", "source_id"}},
                delta_pix=args.delta_pix,
                valid_pixel_mode=args.robust_valid_pixels,
                flux_threshold=args.robust_flux_threshold,
                dilation_radius=args.robust_dilation_radius,
                config=robust_config,
            )
            reconstruction = robust_result.source
            true_source_for_metrics = sample_truth_on_grid(source, robust_result.grid, args.source_extent)
            save_grayscale_png(stretch(robust_result.coverage), dirs["coverage"] / f"{case_id}_coverage.png")
            beta_values = reconstruction_input[robust_result.valid_mask]
            save_beta_scatter(
                robust_result.beta_x,
                robust_result.beta_y,
                beta_values,
                dirs["diagnostics"] / f"{case_id}_beta_scatter.png",
            )
        elif args.deposition in {"forward-sersic", "hybrid-forward-pixel"}:
            lens_center = (
                (args.num_pix - 1) / 2.0 + float(metadata["lens_center_x"]) / args.delta_pix,
                (args.num_pix - 1) / 2.0 + float(metadata["lens_center_y"]) / args.delta_pix,
            )
            forward_reconstruction, forward_stats = fit_forward_lensed_sersic_source(
                reconstruction_input,
                pred_mask,
                lens_center=lens_center,
                theta_e_pixels=float(metadata["theta_E"]) / args.delta_pix,
                axis_ratio=float(metadata["lens_axis_ratio"]),
                position_angle_degrees=float(metadata["lens_angle_degrees"]),
                output_size=args.source_size,
                source_extent_pixels=args.source_extent / args.delta_pix,
            )
            if args.deposition == "forward-sersic":
                reconstruction = forward_reconstruction
            else:
                pixel_reconstruction = reconstruct_source_with_lens_model(
                    reconstruction_input,
                    pred_mask,
                    {key: float(value) for key, value in metadata.items() if key not in {"id", "source_id"}},
                    delta_pix=args.delta_pix,
                    output_size=args.source_size,
                    source_extent=args.source_extent,
                    deposition="regularized_inversion",
                    smooth_radius=args.source_smooth_radius,
                    inversion_lambda=args.inversion_lambda,
                    inversion_regularization_type=args.inversion_regularization,
                )
                pixel_reconstruction = ndimage.median_filter(pixel_reconstruction, size=3)
                pixel_reconstruction = ndimage.gaussian_filter(pixel_reconstruction, sigma=0.65)
                if np.any(pixel_reconstruction > 0):
                    pixel_reconstruction = pixel_reconstruction / max(float(np.percentile(pixel_reconstruction[pixel_reconstruction > 0], 99.3)), 1e-8)
                forward_reconstruction = normalize_image(forward_reconstruction)
                reconstruction = normalize_image(0.68 * np.clip(pixel_reconstruction, 0.0, 1.0) + 0.32 * forward_reconstruction)
        else:
            reconstruction = reconstruct_source_with_lens_model(
                reconstruction_input,
                pred_mask,
                {key: float(value) for key, value in metadata.items() if key not in {"id", "source_id"}},
                delta_pix=args.delta_pix,
                output_size=args.source_size,
                source_extent=args.source_extent,
                deposition=args.deposition.replace("-", "_"),
                smooth_radius=args.source_smooth_radius,
                inversion_lambda=args.inversion_lambda,
                inversion_regularization_type=args.inversion_regularization,
            )
        reconstruction_scaled = match_photometric_scale(true_source_for_metrics, reconstruction)
        reconstruction_display = reconstruction_scaled
        display_stats: dict[str, float | str] = {"source_display_enhanced": 0.0, "source_display_crop_size": ""}
        if args.enhance_review_display:
            reconstruction_display, display_stats = enhance_reconstruction_display(
                reconstruction_scaled,
                output_size=args.source_size,
                threshold_fraction=0.08,
                padding_fraction=0.45,
                gamma=0.72,
            )
        save_grayscale_png(reconstruction_display, dirs["recon"] / f"{case_id}_reconstructed_source.png")
        det = segmentation_metrics(pred_mask, true_mask)
        quality = reconstruction_quality_metrics(true_source_for_metrics, reconstruction_scaled)
        row = {
                "id": case_id,
                "source_id": source_id,
                "precision": round(det["precision"], 6),
                "recall": round(det["recall"], 6),
                "f1": round(det["f1"], 6),
                "iou": round(det["iou"], 6),
                "psnr": round(psnr(true_source_for_metrics, reconstruction_scaled), 6),
                "ssim": round(ssim_simple(true_source_for_metrics, reconstruction_scaled), 6),
                "mse_raw": round(quality["mse_raw"], 6),
                "psnr_aligned": round(quality["psnr_aligned"], 6),
                "ssim_aligned": round(quality["ssim_aligned"], 6),
                "ncc_aligned": round(quality["ncc_aligned"], 6),
                "centroid_error": round(quality["centroid_error"], 6),
                "source_size_error": round(quality["source_size_error"], 6),
                "axis_ratio_error": round(quality["axis_ratio_error"], 6),
                "orientation_error": round(quality["orientation_error"], 6),
                "flux_error_fraction": round(quality["flux_error_fraction"], 6),
                "concentration_error": round(quality["concentration_error"], 6),
                "asymmetry_error": round(quality["asymmetry_error"], 6),
                "detected_pixels": int(pred_mask.sum()),
                "true_arc_pixels": int(true_mask.sum()),
                "mask_source": args.mask_source,
                "image_source": args.image_source,
                "source_positive_threshold": args.source_positive_threshold,
                "inversion_lambda": args.inversion_lambda,
                "inversion_regularization": args.inversion_regularization,
                "forward_source_fit_used": forward_stats["forward_source_fit_used"],
                "forward_source_fit_status": forward_stats["forward_source_fit_status"],
                "forward_fit_rms": forward_stats["forward_fit_rms"],
                "forward_fit_cost": forward_stats["forward_fit_cost"],
                "source_display_enhanced": display_stats["source_display_enhanced"],
                "source_display_crop_size": display_stats["source_display_crop_size"],
            }
        if robust_result is not None:
            row.update({key: round(float(value), 6) if isinstance(value, float) else value for key, value in robust_result.stats.items()})
        metric_rows.append(row)
        metadata_rows.append(metadata)
        if index < args.save_images:
            panel_path = dirs["panels"] / f"{case_id}_review.png"
            save_panel(case_id, image, true_mask, pred_mask, true_source_for_metrics, reconstruction_display, panel_path)
            panel_paths.append(panel_path)
            if robust_result is not None:
                save_robust_panel(
                    case_id,
                    image,
                    true_mask,
                    pred_mask,
                    robust_result.valid_mask,
                    robust_result.coverage,
                    true_source_for_metrics,
                    reconstruction_scaled,
                    dirs["diagnostics"] / f"{case_id}_diagnostic.png",
                )
        if (index + 1) % 25 == 0 or index + 1 == args.count:
            print(f"Processed {index + 1}/{args.count}", flush=True)

    write_csv(args.out_dir / "metadata.csv", metadata_rows)
    write_csv(args.out_dir / "cosmos_lensing_metrics.csv", metric_rows)
    summary: dict[str, float | int] = {"count": len(metric_rows)}
    for key in ["precision", "recall", "f1", "iou", "psnr", "ssim", "psnr_aligned", "ssim_aligned", "ncc_aligned"]:
        values = np.array([float(row[key]) for row in metric_rows], dtype=float)
        summary[f"mean_{key}"] = round(float(values.mean()), 6)
        summary[f"median_{key}"] = round(float(np.median(values)), 6)
    write_csv(args.out_dir / "cosmos_lensing_summary.csv", [summary])
    save_contact_sheet(panel_paths, args.out_dir / "cosmos_lensing_contact_sheet.png")
    print(f"Saved COSMOS real-source lensing benchmark to {args.out_dir}")


if __name__ == "__main__":
    main()
