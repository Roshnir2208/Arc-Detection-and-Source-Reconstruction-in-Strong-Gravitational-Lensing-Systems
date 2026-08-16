from __future__ import annotations

from dataclasses import dataclass
import os

import numpy as np
from scipy import ndimage, signal
from scipy import interpolate
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from lensing_pipeline.metrics import match_photometric_scale, psnr, ssim_simple
from lensing_pipeline.reconstruction import approximate_sie_ray_shooting


@dataclass(frozen=True)
class SourceGrid:
    center_x: float
    center_y: float
    extent: float
    output_size: int


@dataclass(frozen=True)
class RobustReconstructionConfig:
    output_size: int = 192
    source_extent: float = 0.62
    auto_extent: bool = True
    auto_margin_fraction: float = 0.18
    auto_bound_low_percentile: float = 1.0
    auto_bound_high_percentile: float = 99.0
    min_auto_extent: float = 0.06
    aggregation: str = "weighted_mean"
    reconstruction_method: str = "clough_tocher"
    scattered_max_grid_distance: float = 3.0
    beta_aggregation_sigma_clip: float = 2.5
    beta_aggregation_min_bin_samples: int = 2
    rbf_epsilon: float | None = None
    rbf_smoothing: float = 0.0
    rbf_neighbors: int = 48
    hole_fill: str = "none"
    local_fill_iterations: int = 1
    max_interpolation_gap_pixels: float = 3.0
    output_normalization: str = "percentile"
    regularization: str = "none"
    regularization_lambda: float = 0.0
    ray_tracer: str = "approximate"


@dataclass
class RobustReconstructionResult:
    source: np.ndarray
    coverage: np.ndarray
    valid_mask: np.ndarray
    beta_x: np.ndarray
    beta_y: np.ndarray
    grid: SourceGrid
    stats: dict[str, float | str]


@dataclass(frozen=True)
class SupportCleanupConfig:
    min_density_fraction: float = 0.03
    max_nearest_distance_pixels: float = 3.0
    closing_radius: int = 1
    min_island_pixels: int = 12
    max_hole_pixels: int = 24


@dataclass(frozen=True)
class TVRegularizationConfig:
    weight: float = 0.01
    max_iterations: int = 80


def select_valid_pixels(
    image: np.ndarray,
    arc_mask: np.ndarray,
    mode: str = "arc",
    threshold: float = 0.002,
    dilation_radius: int = 0,
) -> np.ndarray:
    """Select image-plane pixels that are allowed to contribute to source reconstruction."""
    image = np.asarray(image, dtype=float)
    mask = np.asarray(arc_mask, dtype=bool)
    if dilation_radius > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation_radius)
    if mode == "arc":
        valid = mask
    elif mode == "flux":
        valid = image > threshold
    elif mode == "arc_and_flux":
        valid = mask & (image > threshold)
    elif mode == "arc_or_flux":
        valid = mask | (image > threshold)
    else:
        raise ValueError(f"Unknown valid-pixel mode: {mode}")
    return np.asarray(valid, dtype=bool)


def image_pixels_to_angles(
    xx: np.ndarray,
    yy: np.ndarray,
    image_shape: tuple[int, int],
    delta_pix: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert NumPy pixel coordinates into centred angular coordinates."""
    height, width = image_shape
    theta_x = (xx.astype(float) - (width - 1) / 2.0) * delta_pix
    theta_y = (yy.astype(float) - (height - 1) / 2.0) * delta_pix
    return theta_x, theta_y


def lenstronomy_sie_ray_shooting(
    theta_x: np.ndarray,
    theta_y: np.ndarray,
    lens_metadata: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Ray-shoot with the exact Lenstronomy SIE parameter convention used for simulation."""
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    try:
        from lenstronomy.LensModel.lens_model import LensModel
    except ImportError as exc:
        raise RuntimeError("lenstronomy is required for exact oracle ray-shooting.") from exc
    lens_model = LensModel(lens_model_list=["SIE"])
    kwargs_lens = [
        {
            "theta_E": float(lens_metadata["theta_E"]),
            "e1": float(lens_metadata["lens_e1"]),
            "e2": float(lens_metadata["lens_e2"]),
            "center_x": float(lens_metadata.get("lens_center_x", 0.0)),
            "center_y": float(lens_metadata.get("lens_center_y", 0.0)),
        }
    ]
    beta_x, beta_y = lens_model.ray_shooting(np.asarray(theta_x, dtype=float), np.asarray(theta_y, dtype=float), kwargs_lens)
    return np.asarray(beta_x, dtype=float), np.asarray(beta_y, dtype=float)


def map_to_source_plane(
    valid_mask: np.ndarray,
    lens_metadata: dict[str, float],
    delta_pix: float,
    ray_tracer: str = "approximate",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply beta = theta - alpha(theta) to selected image-plane pixels."""
    yy, xx = np.nonzero(valid_mask)
    theta_x, theta_y = image_pixels_to_angles(xx, yy, valid_mask.shape, delta_pix)
    if ray_tracer == "approximate":
        beta_x, beta_y = approximate_sie_ray_shooting(theta_x, theta_y, lens_metadata)
    elif ray_tracer == "lenstronomy":
        beta_x, beta_y = lenstronomy_sie_ray_shooting(theta_x, theta_y, lens_metadata)
    else:
        raise ValueError(f"Unknown ray tracer: {ray_tracer}")
    return xx, yy, beta_x, beta_y


def source_grid_from_beta(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    output_size: int,
    fixed_extent: float,
    auto_extent: bool,
    margin_fraction: float,
    low_percentile: float,
    high_percentile: float,
    min_auto_extent: float,
) -> SourceGrid:
    if len(beta_x) == 0 or not auto_extent:
        return SourceGrid(0.0, 0.0, float(fixed_extent), int(output_size))
    low = float(np.clip(low_percentile, 0.0, 49.0))
    high = float(np.clip(high_percentile, 51.0, 100.0))
    x0, x1 = np.percentile(beta_x, [low, high]).astype(float)
    y0, y1 = np.percentile(beta_y, [low, high]).astype(float)
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    span = max(x1 - x0, y1 - y0)
    extent = max(0.5 * span * (1.0 + margin_fraction), min_auto_extent)
    return SourceGrid(cx, cy, float(extent), int(output_size))


def beta_percentile_stats(beta_x: np.ndarray, beta_y: np.ndarray) -> dict[str, float]:
    """Summarise mapped source-plane coordinates for diagnosing outliers and grid scale."""
    if len(beta_x) == 0:
        return {f"beta_{axis}_{name}": np.nan for axis in ("x", "y") for name in ("min", "p01", "p05", "p50", "p95", "p99", "max")}
    percentiles = [1.0, 5.0, 50.0, 95.0, 99.0]
    x_vals = np.percentile(beta_x, percentiles)
    y_vals = np.percentile(beta_y, percentiles)
    stats = {
        "beta_x_min": float(np.min(beta_x)),
        "beta_x_p01": float(x_vals[0]),
        "beta_x_p05": float(x_vals[1]),
        "beta_x_p50": float(x_vals[2]),
        "beta_x_p95": float(x_vals[3]),
        "beta_x_p99": float(x_vals[4]),
        "beta_x_max": float(np.max(beta_x)),
        "beta_y_min": float(np.min(beta_y)),
        "beta_y_p01": float(y_vals[0]),
        "beta_y_p05": float(y_vals[1]),
        "beta_y_p50": float(y_vals[2]),
        "beta_y_p95": float(y_vals[3]),
        "beta_y_p99": float(y_vals[4]),
        "beta_y_max": float(np.max(beta_y)),
    }
    return stats


def bilinear_weighted_accumulate(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    values: np.ndarray,
    grid: SourceGrid,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate source flux with bilinear weights and return weighted mean plus coverage."""
    n = grid.output_size
    sx = (beta_x - (grid.center_x - grid.extent)) / (2.0 * grid.extent) * (n - 1)
    sy = (beta_y - (grid.center_y - grid.extent)) / (2.0 * grid.extent) * (n - 1)
    valid = (sx >= 0.0) & (sx < n - 1) & (sy >= 0.0) & (sy < n - 1)

    flux = np.zeros((n, n), dtype=float)
    coverage = np.zeros_like(flux)
    if not np.any(valid):
        return flux, coverage, valid

    x0 = np.floor(sx[valid]).astype(int)
    y0 = np.floor(sy[valid]).astype(int)
    fx = sx[valid] - x0
    fy = sy[valid] - y0
    vals = values[valid].astype(float)
    deposits = (
        (y0, x0, (1.0 - fx) * (1.0 - fy)),
        (y0, x0 + 1, fx * (1.0 - fy)),
        (y0 + 1, x0, (1.0 - fx) * fy),
        (y0 + 1, x0 + 1, fx * fy),
    )
    for yy, xx, weight in deposits:
        np.add.at(flux, (yy, xx), vals * weight)
        np.add.at(coverage, (yy, xx), weight)

    source = np.divide(flux, coverage, out=np.zeros_like(flux), where=coverage > 1e-10)
    return source, coverage, valid


def source_grid_coordinates(grid: SourceGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return regular source-grid beta coordinates matching the reconstruction image."""
    n = grid.output_size
    gy, gx = np.mgrid[:n, :n]
    beta_x = grid.center_x + (gx / max(n - 1, 1)) * 2.0 * grid.extent - grid.extent
    beta_y = grid.center_y + (gy / max(n - 1, 1)) * 2.0 * grid.extent - grid.extent
    return beta_x, beta_y


def _remove_small_boolean_components(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(np.asarray(mask, dtype=bool))
    if count == 0:
        return np.asarray(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= max(int(min_pixels), 1)
    keep[0] = False
    return keep[labels]


def _fill_small_boolean_holes(mask: np.ndarray, max_hole_pixels: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    filled = ndimage.binary_fill_holes(mask)
    holes = filled & ~mask
    labels, count = ndimage.label(holes)
    if count == 0:
        return mask
    sizes = np.bincount(labels.ravel())
    fill = np.zeros_like(mask, dtype=bool)
    for label in range(1, count + 1):
        if sizes[label] <= max(int(max_hole_pixels), 0):
            fill |= labels == label
    return mask | fill


def support_diagnostic_maps(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    grid: SourceGrid,
    coverage: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return source-plane support diagnostics without modifying the reconstruction."""
    coverage = np.asarray(coverage, dtype=float)
    grid_beta_x, grid_beta_y = source_grid_coordinates(grid)
    if len(beta_x) == 0:
        nearest = np.full_like(coverage, np.inf, dtype=float)
    else:
        points = np.column_stack([beta_x.astype(float), beta_y.astype(float)])
        targets = np.column_stack([grid_beta_x.ravel(), grid_beta_y.ravel()])
        distances, _indices = cKDTree(points).query(targets, k=1)
        pixel_scale = (2.0 * grid.extent) / max(grid.output_size - 1, 1)
        nearest = (distances / max(pixel_scale, 1e-12)).reshape(coverage.shape)
    local_density = ndimage.gaussian_filter(coverage, sigma=1.25) if np.any(coverage > 0) else coverage.copy()
    support = coverage > 0
    return {
        "beta_sample_density": coverage,
        "local_beta_sample_density": local_density,
        "nearest_beta_distance_pixels": nearest,
        "interpolation_support": support.astype(float),
        "invalid_or_extrapolated_pixels": (~support).astype(float),
    }


def clean_source_support_mask(
    coverage: np.ndarray,
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    grid: SourceGrid,
    config: SupportCleanupConfig | None = None,
) -> tuple[np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    """Build a conservative source support mask from sample density and nearest-sample distance."""
    if config is None:
        config = SupportCleanupConfig()
    maps = support_diagnostic_maps(beta_x, beta_y, grid, coverage)
    density = maps["local_beta_sample_density"]
    nearest = maps["nearest_beta_distance_pixels"]
    if not np.any(density > 0):
        empty = np.zeros_like(density, dtype=bool)
        return empty, {
            "support_initial_fraction": 0.0,
            "support_cleaned_fraction": 0.0,
            "support_removed_fraction": 0.0,
            "support_filled_fraction": 0.0,
            "support_component_count": 0.0,
        }, maps

    density_threshold = max(float(np.max(density)) * float(config.min_density_fraction), 1e-10)
    initial = (density >= density_threshold) & (nearest <= float(config.max_nearest_distance_pixels))
    cleaned = initial.copy()
    if config.closing_radius > 0:
        structure = ndimage.generate_binary_structure(2, 1)
        cleaned = ndimage.binary_closing(cleaned, structure=structure, iterations=int(config.closing_radius))
    cleaned = _remove_small_boolean_components(cleaned, config.min_island_pixels)
    before_holes = cleaned.copy()
    cleaned = _fill_small_boolean_holes(cleaned, config.max_hole_pixels)
    cleaned &= nearest <= float(config.max_nearest_distance_pixels)

    labels, count = ndimage.label(cleaned)
    stats = {
        "support_density_threshold": float(density_threshold),
        "support_initial_fraction": float(np.mean(initial)),
        "support_cleaned_fraction": float(np.mean(cleaned)),
        "support_removed_fraction": float(np.mean(initial & ~cleaned)),
        "support_filled_fraction": float(np.mean(cleaned & ~before_holes)),
        "support_component_count": float(count),
    }
    maps = {**maps, "cleaned_support": cleaned.astype(float), "initial_support": initial.astype(float)}
    return cleaned.astype(bool), stats, maps


def apply_support_mask(source: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    """Set unsupported source pixels to zero without changing supported pixels."""
    return np.where(np.asarray(support_mask, dtype=bool), np.asarray(source, dtype=float), 0.0)


def _tv_chambolle_fallback(image: np.ndarray, weight: float, max_iterations: int) -> np.ndarray:
    """Small Chambolle-style TV denoising fallback used when scikit-image is unavailable."""
    image = np.asarray(image, dtype=float)
    px = np.zeros_like(image)
    py = np.zeros_like(image)
    tau = 0.125
    for _ in range(max(int(max_iterations), 1)):
        div = np.zeros_like(image)
        div[:, :-1] += px[:, :-1]
        div[:, 1:] -= px[:, :-1]
        div[:-1, :] += py[:-1, :]
        div[1:, :] -= py[:-1, :]
        u = image + float(weight) * div
        grad_x = np.zeros_like(image)
        grad_y = np.zeros_like(image)
        grad_x[:, :-1] = np.diff(u, axis=1)
        grad_y[:-1, :] = np.diff(u, axis=0)
        denom = 1.0 + tau * np.sqrt(grad_x * grad_x + grad_y * grad_y)
        px = (px + tau * grad_x) / denom
        py = (py + tau * grad_y) / denom
    div = np.zeros_like(image)
    div[:, :-1] += px[:, :-1]
    div[:, 1:] -= px[:, :-1]
    div[:-1, :] += py[:-1, :]
    div[1:, :] -= py[:-1, :]
    return image + float(weight) * div


def tv_regularize_supported_source(
    source: np.ndarray,
    support_mask: np.ndarray,
    config: TVRegularizationConfig | None = None,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    """Apply conservative TV regularisation only inside valid source support."""
    if config is None:
        config = TVRegularizationConfig()
    source = np.clip(np.asarray(source, dtype=float), 0.0, None)
    support = np.asarray(support_mask, dtype=bool)
    if float(config.weight) <= 0.0 or not np.any(support):
        gradient = np.hypot(ndimage.sobel(source, axis=0), ndimage.sobel(source, axis=1))
        return apply_support_mask(source, support), {"tv_applied": 0.0, "tv_flux_scale": 1.0}, gradient

    working = apply_support_mask(source, support)
    try:
        from skimage.restoration import denoise_tv_chambolle

        denoised = denoise_tv_chambolle(working, weight=float(config.weight), max_num_iter=int(config.max_iterations), channel_axis=None)
    except Exception:
        denoised = _tv_chambolle_fallback(working, float(config.weight), int(config.max_iterations))

    denoised = apply_support_mask(np.clip(denoised, 0.0, None), support)
    original_flux = float(np.sum(working[support]))
    denoised_flux = float(np.sum(denoised[support]))
    flux_scale = 1.0
    if original_flux > 0.0 and denoised_flux > 0.0:
        flux_scale = original_flux / denoised_flux
        denoised[support] *= flux_scale
    gradient = np.hypot(ndimage.sobel(denoised, axis=0), ndimage.sobel(denoised, axis=1))
    stats = {
        "tv_applied": 1.0,
        "tv_weight": float(config.weight),
        "tv_iterations": float(config.max_iterations),
        "tv_flux_scale": float(flux_scale),
        "tv_gradient_mean_supported": float(np.mean(gradient[support])) if np.any(support) else 0.0,
        "tv_gradient_max_supported": float(np.max(gradient[support])) if np.any(support) else 0.0,
    }
    return np.clip(denoised, 0.0, None), stats, gradient


def aggregate_beta_samples(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    values: np.ndarray,
    grid: SourceGrid,
    sigma_clip: float = 2.5,
    min_bin_samples: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Robustly combine mapped samples that fall into the same source-grid bin."""
    if len(beta_x) == 0:
        return beta_x, beta_y, values, {"original_samples": 0.0, "aggregated_samples": 0.0, "rejected_outlier_count": 0.0}

    n = grid.output_size
    sx = (beta_x - (grid.center_x - grid.extent)) / (2.0 * grid.extent) * (n - 1)
    sy = (beta_y - (grid.center_y - grid.extent)) / (2.0 * grid.extent) * (n - 1)
    valid = (sx >= 0.0) & (sx < n) & (sy >= 0.0) & (sy < n) & np.isfinite(values)
    if not np.any(valid):
        return np.array([]), np.array([]), np.array([]), {"original_samples": float(len(beta_x)), "aggregated_samples": 0.0, "rejected_outlier_count": float(len(beta_x))}

    sx = sx[valid]
    sy = sy[valid]
    bx = beta_x[valid]
    by = beta_y[valid]
    vals = values[valid].astype(float)
    bin_x = np.clip(np.floor(sx).astype(int), 0, n - 1)
    bin_y = np.clip(np.floor(sy).astype(int), 0, n - 1)
    bin_id = bin_y * n + bin_x

    out_x: list[float] = []
    out_y: list[float] = []
    out_values: list[float] = []
    rejected = int(len(beta_x) - np.count_nonzero(valid))
    for current_bin in np.unique(bin_id):
        members = bin_id == current_bin
        member_values = vals[members]
        keep = np.ones(len(member_values), dtype=bool)
        if len(member_values) >= max(3, int(min_bin_samples)):
            median = float(np.median(member_values))
            mad = float(np.median(np.abs(member_values - median)))
            robust_sigma = 1.4826 * mad
            if robust_sigma > 1e-10:
                keep = np.abs(member_values - median) <= float(sigma_clip) * robust_sigma
        rejected += int(np.count_nonzero(~keep))
        if not np.any(keep):
            continue
        member_x = bx[members][keep]
        member_y = by[members][keep]
        member_values = member_values[keep]
        out_x.append(float(np.mean(member_x)))
        out_y.append(float(np.mean(member_y)))
        out_values.append(float(np.median(member_values)))

    stats = {
        "original_samples": float(len(beta_x)),
        "aggregated_samples": float(len(out_values)),
        "rejected_outlier_count": float(rejected),
    }
    return np.array(out_x, dtype=float), np.array(out_y, dtype=float), np.array(out_values, dtype=float), stats


def scattered_interpolate_source(
    beta_x: np.ndarray,
    beta_y: np.ndarray,
    values: np.ndarray,
    grid: SourceGrid,
    method: str = "griddata_linear",
    max_grid_distance: float = 3.0,
    sigma_clip: float = 2.5,
    min_bin_samples: int = 2,
    rbf_epsilon: float | None = None,
    rbf_smoothing: float = 0.0,
    rbf_neighbors: int = 48,
) -> tuple[np.ndarray, dict[str, float | str]]:
    """Interpolate irregular beta samples onto the source grid with nearest-sample support gating."""
    stats: dict[str, float | str] = {
        "scattered_method_requested": method,
        "scattered_method_used": method,
        "original_samples": float(len(beta_x)),
        "aggregated_samples": float(len(beta_x)),
        "rejected_outlier_count": 0.0,
    }
    if len(beta_x) < 4:
        return np.zeros((grid.output_size, grid.output_size), dtype=float), stats

    use_aggregation = method.endswith("_aggregated")
    base_method = method.removesuffix("_aggregated")
    if use_aggregation:
        beta_x, beta_y, values, aggregate_stats = aggregate_beta_samples(
            beta_x,
            beta_y,
            values,
            grid,
            sigma_clip=sigma_clip,
            min_bin_samples=min_bin_samples,
        )
        stats.update(aggregate_stats)
        stats["scattered_method_used"] = base_method
        if len(beta_x) < 4:
            return np.zeros((grid.output_size, grid.output_size), dtype=float), stats

    grid_beta_x, grid_beta_y = source_grid_coordinates(grid)
    points = np.column_stack([beta_x.astype(float), beta_y.astype(float)])
    target_points = np.column_stack([grid_beta_x.ravel(), grid_beta_y.ravel()])
    vals = values.astype(float)

    if base_method == "griddata_linear":
        interpolated = interpolate.griddata(points, vals, target_points, method="linear", fill_value=0.0)
    elif base_method == "clough_tocher":
        try:
            interpolator = interpolate.CloughTocher2DInterpolator(points, vals, fill_value=0.0)
            interpolated = interpolator(target_points)
        except Exception:
            interpolated = interpolate.griddata(points, vals, target_points, method="linear", fill_value=0.0)
            stats["scattered_method_used"] = "griddata_linear_fallback"
    elif base_method.startswith("rbf_"):
        if len(points) > 5000:
            # RBF interpolation is cubic in sample count; use a deterministic subset for tractability.
            subset = np.linspace(0, len(points) - 1, 5000).astype(int)
            points = points[subset]
            vals = vals[subset]
            stats["rbf_sample_limit_used"] = 5000.0
        kernel = base_method.removeprefix("rbf_")
        supported_kernels = {
            "linear",
            "thin_plate_spline",
            "cubic",
            "multiquadric",
            "inverse_multiquadric",
            "gaussian",
        }
        if kernel not in supported_kernels:
            raise ValueError(f"Unknown RBF kernel: {kernel}")
        kwargs: dict[str, float | int | str | None] = {
            "kernel": kernel,
            "neighbors": max(8, int(rbf_neighbors)),
            "smoothing": float(rbf_smoothing),
        }
        if rbf_epsilon is not None and kernel in {"multiquadric", "inverse_multiquadric", "gaussian"}:
            kwargs["epsilon"] = float(rbf_epsilon)
        try:
            interpolator = interpolate.RBFInterpolator(points, vals, **kwargs)
            interpolated = interpolator(target_points)
            stats["rbf_kernel"] = kernel
            stats["rbf_smoothing"] = float(rbf_smoothing)
            stats["rbf_neighbors"] = float(max(8, int(rbf_neighbors)))
            stats["rbf_epsilon"] = "" if rbf_epsilon is None else float(rbf_epsilon)
        except Exception as exc:
            if float(rbf_smoothing) == 0.0 and kernel in {"multiquadric", "inverse_multiquadric", "gaussian"}:
                try:
                    kwargs["smoothing"] = 1e-8
                    interpolator = interpolate.RBFInterpolator(points, vals, **kwargs)
                    interpolated = interpolator(target_points)
                    stats["rbf_kernel"] = kernel
                    stats["rbf_smoothing"] = 1e-8
                    stats["rbf_neighbors"] = float(max(8, int(rbf_neighbors)))
                    stats["rbf_epsilon"] = "" if rbf_epsilon is None else float(rbf_epsilon)
                    stats["rbf_retry_reason"] = f"{type(exc).__name__}: {exc}"
                    stats["rbf_retry_smoothing"] = 1e-8
                except Exception as retry_exc:
                    interpolated = interpolate.griddata(points, vals, target_points, method="linear", fill_value=0.0)
                    stats["scattered_method_used"] = "griddata_linear_fallback"
                    stats["rbf_kernel"] = kernel
                    stats["rbf_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
            else:
                interpolated = interpolate.griddata(points, vals, target_points, method="linear", fill_value=0.0)
                stats["scattered_method_used"] = "griddata_linear_fallback"
                stats["rbf_kernel"] = kernel
                stats["rbf_error"] = f"{type(exc).__name__}: {exc}"
    else:
        raise ValueError(f"Unknown scattered interpolation method: {method}")

    interpolated = np.nan_to_num(interpolated, nan=0.0, posinf=0.0, neginf=0.0).reshape((grid.output_size, grid.output_size))

    if max_grid_distance > 0:
        pixel_scale = (2.0 * grid.extent) / max(grid.output_size - 1, 1)
        tree = cKDTree(points)
        distances, _indices = tree.query(target_points, k=1)
        support = distances.reshape((grid.output_size, grid.output_size)) <= max_grid_distance * pixel_scale
        interpolated = np.where(support, interpolated, 0.0)
        stats["support_fraction"] = float(np.mean(support))

    return np.clip(interpolated, 0.0, None), stats


def fill_small_holes(source: np.ndarray, coverage: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Optional local-only fill for tiny holes inside already covered source regions."""
    if iterations <= 0 or not np.any(coverage > 0):
        return source
    filled = source.copy()
    support = coverage > 0
    for _ in range(iterations):
        local_weight = ndimage.uniform_filter(support.astype(float), size=3)
        local_value = ndimage.uniform_filter(filled * support, size=3)
        candidates = (~support) & (local_weight >= 0.45)
        filled[candidates] = local_value[candidates] / np.maximum(local_weight[candidates], 1e-10)
        support = support | candidates
    return filled


def fill_nearest_local(source: np.ndarray, coverage: np.ndarray, max_gap_pixels: float = 3.0) -> np.ndarray:
    """Fill empty source pixels from nearest covered pixels only within a strict local distance."""
    covered = coverage > 0
    if not np.any(covered):
        return source
    yy, xx = np.nonzero(covered)
    tree = cKDTree(np.column_stack([yy, xx]))
    grid_y, grid_x = np.mgrid[: source.shape[0], : source.shape[1]]
    points = np.column_stack([grid_y.ravel(), grid_x.ravel()])
    distances, indices = tree.query(points, k=1)
    fillable = (distances.reshape(source.shape) <= max_gap_pixels) & (~covered)
    nearest_values = source[yy[indices], xx[indices]].reshape(source.shape)
    filled = source.copy()
    filled[fillable] = nearest_values[fillable]
    return filled


def tikhonov_regularize_populated_source(
    source: np.ndarray,
    coverage: np.ndarray,
    regularization_lambda: float = 0.03,
) -> np.ndarray:
    """Apply mild L2 smoothness only across already populated source-plane pixels."""
    lam = float(regularization_lambda)
    populated = np.asarray(coverage) > 0
    if lam <= 0.0 or not np.any(populated):
        return np.asarray(source, dtype=float).copy()

    index_map = -np.ones(source.shape, dtype=int)
    yy, xx = np.nonzero(populated)
    index_map[yy, xx] = np.arange(len(yy), dtype=int)
    weights = np.maximum(np.asarray(coverage, dtype=float)[yy, xx], 1e-8)
    values = np.asarray(source, dtype=float)[yy, xx]

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = weights * values

    for node, (y, x) in enumerate(zip(yy, xx, strict=False)):
        degree = 0
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < source.shape[0] and 0 <= nx < source.shape[1]:
                neighbour = int(index_map[ny, nx])
                if neighbour >= 0:
                    rows.append(node)
                    cols.append(neighbour)
                    data.append(-lam)
                    degree += 1
        rows.append(node)
        cols.append(node)
        data.append(float(weights[node]) + lam * degree)

    system = sparse.csr_matrix((data, (rows, cols)), shape=(len(yy), len(yy)))
    try:
        solved = sparse_linalg.spsolve(system, rhs)
    except Exception:
        solved = sparse_linalg.lsmr(system, rhs)[0]

    regularized = np.zeros_like(np.asarray(source, dtype=float))
    regularized[yy, xx] = np.clip(np.asarray(solved, dtype=float), 0.0, None)
    return regularized


def robust_source_reconstruction(
    image: np.ndarray,
    arc_mask: np.ndarray,
    lens_metadata: dict[str, float],
    delta_pix: float,
    valid_pixel_mode: str = "arc",
    flux_threshold: float = 0.002,
    dilation_radius: int = 0,
    config: RobustReconstructionConfig | None = None,
) -> RobustReconstructionResult:
    """Ray-trace selected image pixels to source plane using weighted, coverage-aware accumulation."""
    if config is None:
        config = RobustReconstructionConfig()
    valid_mask = select_valid_pixels(image, arc_mask, valid_pixel_mode, flux_threshold, dilation_radius)
    xx, yy, beta_x, beta_y = map_to_source_plane(valid_mask, lens_metadata, delta_pix, ray_tracer=config.ray_tracer)
    grid = source_grid_from_beta(
        beta_x,
        beta_y,
        output_size=config.output_size,
        fixed_extent=config.source_extent,
        auto_extent=config.auto_extent,
        margin_fraction=config.auto_margin_fraction,
        low_percentile=config.auto_bound_low_percentile,
        high_percentile=config.auto_bound_high_percentile,
        min_auto_extent=config.min_auto_extent,
    )
    values = np.asarray(image, dtype=float)[yy, xx] if len(xx) else np.array([], dtype=float)
    source, coverage, source_valid = bilinear_weighted_accumulate(beta_x, beta_y, values, grid)
    scattered_stats: dict[str, float | str] = {
        "scattered_method_requested": config.reconstruction_method,
        "scattered_method_used": "weighted",
        "original_samples": float(len(values)),
        "aggregated_samples": float(len(values)),
        "rejected_outlier_count": 0.0,
        "support_fraction": float(np.mean(coverage > 0)) if coverage.size else 0.0,
    }
    if config.reconstruction_method != "weighted":
        source, scattered_stats = scattered_interpolate_source(
            beta_x[source_valid],
            beta_y[source_valid],
            values[source_valid],
            grid,
            method=config.reconstruction_method,
            max_grid_distance=config.scattered_max_grid_distance,
            sigma_clip=config.beta_aggregation_sigma_clip,
            min_bin_samples=config.beta_aggregation_min_bin_samples,
            rbf_epsilon=config.rbf_epsilon,
            rbf_smoothing=config.rbf_smoothing,
            rbf_neighbors=config.rbf_neighbors,
        )

    if config.hole_fill == "local":
        source = fill_small_holes(source, coverage, iterations=config.local_fill_iterations)
    elif config.hole_fill == "nearest_local":
        source = fill_nearest_local(source, coverage, max_gap_pixels=config.max_interpolation_gap_pixels)
    elif config.hole_fill != "none":
        raise ValueError(f"Unknown hole-fill mode: {config.hole_fill}")

    if config.regularization == "tikhonov":
        source = tikhonov_regularize_populated_source(source, coverage, config.regularization_lambda)
    elif config.regularization != "none":
        raise ValueError(f"Unknown regularization mode: {config.regularization}")

    if config.output_normalization == "percentile":
        positive = source[source > 0]
        if len(positive):
            scale = float(np.percentile(positive, 99.5))
            if scale > 0:
                source = np.clip(source / scale, 0.0, 1.0)
    elif config.output_normalization == "max":
        peak = float(np.max(source))
        if peak > 0:
            source = np.clip(source / peak, 0.0, 1.0)
    elif config.output_normalization != "none":
        raise ValueError(f"Unknown output normalization: {config.output_normalization}")
    populated = coverage > 0
    stats: dict[str, float | str] = {
        "valid_image_pixels": float(np.count_nonzero(valid_mask)),
        "mapped_source_pixels": float(np.count_nonzero(source_valid)),
        "populated_source_pixels": float(np.count_nonzero(populated)),
        "populated_source_fraction": float(np.mean(populated)),
        "coverage_max": float(np.max(coverage)) if coverage.size else 0.0,
        "coverage_mean_populated": float(np.mean(coverage[populated])) if np.any(populated) else 0.0,
        "source_grid_center_x": grid.center_x,
        "source_grid_center_y": grid.center_y,
        "source_grid_extent": grid.extent,
        "source_grid_output_size": float(grid.output_size),
        "valid_pixel_mode": valid_pixel_mode,
        "reconstruction_method": config.reconstruction_method,
        "scattered_max_grid_distance": float(config.scattered_max_grid_distance),
        "beta_aggregation_sigma_clip": float(config.beta_aggregation_sigma_clip),
        "beta_aggregation_min_bin_samples": float(config.beta_aggregation_min_bin_samples),
        "rbf_epsilon": "" if config.rbf_epsilon is None else float(config.rbf_epsilon),
        "rbf_smoothing": float(config.rbf_smoothing),
        "rbf_neighbors": float(config.rbf_neighbors),
        "hole_fill": config.hole_fill,
        "output_normalization": config.output_normalization,
        "regularization": config.regularization,
        "regularization_lambda": float(config.regularization_lambda),
        "ray_tracer": config.ray_tracer,
        "auto_bound_low_percentile": config.auto_bound_low_percentile,
        "auto_bound_high_percentile": config.auto_bound_high_percentile,
        **beta_percentile_stats(beta_x, beta_y),
        **{f"scattered_{key}": value for key, value in scattered_stats.items()},
    }
    return RobustReconstructionResult(source, coverage, valid_mask, beta_x, beta_y, grid, stats)


def sample_truth_on_grid(truth: np.ndarray, grid: SourceGrid, truth_extent: float) -> np.ndarray:
    """Sample a centred truth source image onto the reconstruction source grid."""
    n = grid.output_size
    gy, gx = np.mgrid[:n, :n]
    beta_x = grid.center_x + (gx / max(n - 1, 1)) * 2.0 * grid.extent - grid.extent
    beta_y = grid.center_y + (gy / max(n - 1, 1)) * 2.0 * grid.extent - grid.extent
    sx = (beta_x + truth_extent) / (2.0 * truth_extent) * (truth.shape[1] - 1)
    sy = (beta_y + truth_extent) / (2.0 * truth_extent) * (truth.shape[0] - 1)
    return ndimage.map_coordinates(truth, [sy, sx], order=1, mode="constant", cval=0.0)


def crop_to_support(a: np.ndarray, b: np.ndarray, padding: int = 6) -> tuple[np.ndarray, np.ndarray]:
    support = (np.asarray(a) > 0) | (np.asarray(b) > 0)
    if not np.any(support):
        return a, b
    yy, xx = np.nonzero(support)
    y0 = max(0, int(yy.min()) - padding)
    y1 = min(a.shape[0], int(yy.max()) + padding + 1)
    x0 = max(0, int(xx.min()) - padding)
    x1 = min(a.shape[1], int(xx.max()) + padding + 1)
    return a[y0:y1, x0:x1], b[y0:y1, x0:x1]


def translation_align(reference: np.ndarray, estimate: np.ndarray, max_shift: int = 8) -> tuple[np.ndarray, tuple[int, int]]:
    """Align estimate to reference by integer translation only."""
    ref = np.asarray(reference, dtype=float)
    est = np.asarray(estimate, dtype=float)
    corr = signal.fftconvolve(ref - ref.mean(), (est - est.mean())[::-1, ::-1], mode="same")
    cy, cx = np.array(corr.shape) // 2
    y0, y1 = max(0, cy - max_shift), min(corr.shape[0], cy + max_shift + 1)
    x0, x1 = max(0, cx - max_shift), min(corr.shape[1], cx + max_shift + 1)
    local = corr[y0:y1, x0:x1]
    py, px = np.unravel_index(int(np.argmax(local)), local.shape)
    shift_y = int((y0 + py) - cy)
    shift_x = int((x0 + px) - cx)
    aligned = ndimage.shift(est, shift=(shift_y, shift_x), order=1, mode="constant", cval=0.0)
    return aligned, (shift_y, shift_x)


def morphology_stats(image: np.ndarray) -> dict[str, float]:
    arr = np.clip(np.asarray(image, dtype=float), 0.0, None)
    positive = arr[arr > 0]
    if len(positive) == 0:
        return {"centroid_x": np.nan, "centroid_y": np.nan, "size": 0.0, "axis_ratio": np.nan, "orientation_deg": np.nan, "flux": 0.0, "concentration": np.nan, "asymmetry": np.nan}
    threshold = max(float(np.percentile(positive, 50.0)), 0.02 * float(arr.max()))
    active = arr >= threshold
    if np.count_nonzero(active) == 0:
        active = arr > 0
    yy, xx = np.nonzero(active)
    weights = arr[yy, xx]
    flux = float(np.sum(arr))
    cx = float(np.average(xx, weights=weights))
    cy = float(np.average(yy, weights=weights))
    dx = xx - cx
    dy = yy - cy
    cov_xx = float(np.average(dx * dx, weights=weights))
    cov_yy = float(np.average(dy * dy, weights=weights))
    cov_xy = float(np.average(dx * dy, weights=weights))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 1e-8)
    eigvecs = eigvecs[:, order]
    size = float(np.sqrt(eigvals[0] + eigvals[1]))
    axis_ratio = float(np.sqrt(eigvals[1] / eigvals[0]))
    orientation = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    centre_radius = np.hypot(xx - cx, yy - cy)
    inner = weights[centre_radius <= np.percentile(centre_radius, 30.0)]
    concentration = float(np.sum(inner) / max(np.sum(weights), 1e-10)) if len(inner) else np.nan
    rotated = np.rot90(arr, 2)
    asymmetry = float(np.sum(np.abs(arr - rotated)) / max(np.sum(np.abs(arr)), 1e-10))
    return {"centroid_x": cx, "centroid_y": cy, "size": size, "axis_ratio": axis_ratio, "orientation_deg": orientation, "flux": flux, "concentration": concentration, "asymmetry": asymmetry}


def reconstruction_quality_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float]:
    ref, est = crop_to_support(np.asarray(reference, dtype=float), np.asarray(estimate, dtype=float))
    est_scaled = match_photometric_scale(ref, est)
    aligned, shift = translation_align(ref, est_scaled)
    aligned = match_photometric_scale(ref, aligned)
    ref_m = morphology_stats(ref)
    est_m = morphology_stats(aligned)
    ncc_raw_denom = float(np.linalg.norm(ref.ravel()) * np.linalg.norm(est_scaled.ravel()))
    ncc_aligned_denom = float(np.linalg.norm(ref.ravel()) * np.linalg.norm(aligned.ravel()))
    ncc_raw = float(np.sum(ref * est_scaled) / ncc_raw_denom) if ncc_raw_denom > 0 else 0.0
    ncc_aligned = float(np.sum(ref * aligned) / ncc_aligned_denom) if ncc_aligned_denom > 0 else 0.0
    return {
        "mse_raw": float(np.mean((ref - est_scaled) ** 2)),
        "psnr_raw": psnr(ref, est_scaled),
        "ssim_raw": ssim_simple(ref, est_scaled),
        "ncc_raw": ncc_raw,
        "mse_aligned": float(np.mean((ref - aligned) ** 2)),
        "psnr_aligned": psnr(ref, aligned),
        "ssim_aligned": ssim_simple(ref, aligned),
        "ncc_aligned": ncc_aligned,
        "align_shift_y": float(shift[0]),
        "align_shift_x": float(shift[1]),
        "centroid_error": float(np.hypot(ref_m["centroid_x"] - est_m["centroid_x"], ref_m["centroid_y"] - est_m["centroid_y"])),
        "source_size_error": float(abs(ref_m["size"] - est_m["size"])),
        "axis_ratio_error": float(abs(ref_m["axis_ratio"] - est_m["axis_ratio"])),
        "orientation_error": float(abs(((ref_m["orientation_deg"] - est_m["orientation_deg"] + 90.0) % 180.0) - 90.0)),
        "flux_error_fraction": float(abs(ref_m["flux"] - est_m["flux"]) / max(ref_m["flux"], 1e-10)),
        "concentration_error": float(abs(ref_m["concentration"] - est_m["concentration"])),
        "asymmetry_error": float(abs(ref_m["asymmetry"] - est_m["asymmetry"])),
    }
