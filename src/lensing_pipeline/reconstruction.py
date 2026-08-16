from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.optimize import least_squares
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from scipy.spatial import cKDTree

from lensing_pipeline.detection import box_blur


# Evaluate a smooth Sersic-like galaxy profile at source-plane coordinates.
def sersic_source_intensity(
    x: np.ndarray,
    y: np.ndarray,
    center_x: float,
    center_y: float,
    radius_major: float,
    axis_ratio: float,
    angle_degrees: float,
    sersic_n: float,
    amplitude: float,
) -> np.ndarray:
    radius_major = max(float(radius_major), 1e-4)
    axis_ratio = float(np.clip(axis_ratio, 0.15, 1.0))
    sersic_n = float(np.clip(sersic_n, 0.5, 6.0))
    angle = np.radians(angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = x - center_x
    dy = y - center_y
    major = dx * cos_a + dy * sin_a
    minor = -dx * sin_a + dy * cos_a
    elliptical_radius = np.sqrt(major**2 + (minor / axis_ratio) ** 2)
    bn = 1.9992 * sersic_n - 0.3271
    profile = np.exp(-bn * (np.power(np.maximum(elliptical_radius, 1e-8) / radius_major, 1.0 / sersic_n) - 1.0))
    return np.clip(float(amplitude) * profile, 0.0, None)


# Render a fitted parametric source on a regular source-plane grid.
def render_sersic_source_grid(
    output_size: int,
    source_extent_pixels: float,
    params: dict[str, float],
) -> np.ndarray:
    grid_y, grid_x = np.mgrid[:output_size, :output_size]
    display_extent = max(4.0, min(float(source_extent_pixels), float(params["source_radius_major"]) * 4.5))
    source_x = params["source_center_x"] + (grid_x / max(output_size - 1, 1)) * 2.0 * display_extent - display_extent
    source_y = params["source_center_y"] + (grid_y / max(output_size - 1, 1)) * 2.0 * display_extent - display_extent
    source = sersic_source_intensity(
        source_x,
        source_y,
        center_x=params["source_center_x"],
        center_y=params["source_center_y"],
        radius_major=params["source_radius_major"],
        axis_ratio=params["source_axis_ratio"],
        angle_degrees=params["source_angle_degrees"],
        sersic_n=params["source_sersic_n"],
        amplitude=params["source_amplitude"],
    )
    if np.any(source > 0):
        scale = float(np.percentile(source[source > 0], 99.7))
        if scale > 0:
            source = source / scale
    return np.clip(ndimage.gaussian_filter(source, sigma=0.45), 0.0, 1.0)


# Deposit ray-traced image pixels into a source grid using bilinear weights.
def bilinear_deposit(
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    output_size: int,
    source_extent: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sx = (source_x + source_extent) / (2 * source_extent) * (output_size - 1)
    sy = (source_y + source_extent) / (2 * source_extent) * (output_size - 1)
    valid = (sx >= 0) & (sx < output_size - 1) & (sy >= 0) & (sy < output_size - 1)

    source = np.zeros((output_size, output_size), dtype=float)
    weights = np.zeros_like(source)
    if not np.any(valid):
        return source, weights, valid

    x0 = np.floor(sx[valid]).astype(int)
    y0 = np.floor(sy[valid]).astype(int)
    fx = sx[valid] - x0
    fy = sy[valid] - y0
    vals = values[valid]
    deposits = [
        (y0, x0, (1.0 - fx) * (1.0 - fy)),
        (y0, x0 + 1, fx * (1.0 - fy)),
        (y0 + 1, x0, (1.0 - fx) * fy),
        (y0 + 1, x0 + 1, fx * fy),
    ]
    for yy, xx, ww in deposits:
        np.add.at(source, (yy, xx), vals * ww)
        np.add.at(weights, (yy, xx), ww)

    filled = weights > 1e-8
    source[filled] /= weights[filled]
    return source, weights, valid


# Suppress isolated source-plane noise and normalise contrast after deposition.
def clean_deposited_source(
    source: np.ndarray,
    weights: np.ndarray,
    smooth_radius: int = 1,
    min_weight_fraction: float = 0.05,
    contrast_percentile: float = 99.0,
) -> np.ndarray:
    source = np.asarray(source, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0):
        return np.zeros_like(source, dtype=float)
    support = weights >= max(float(weights.max()) * min_weight_fraction, 1e-8)
    cleaned = np.where(support, source, 0.0)
    if smooth_radius > 0:
        support_blur = box_blur(support.astype(float), radius=smooth_radius)
        value_blur = box_blur(cleaned * support.astype(float), radius=smooth_radius)
        cleaned = np.divide(value_blur, support_blur, out=np.zeros_like(cleaned), where=support_blur > 1e-8)
        cleaned *= support_blur > 0.02
    positive = cleaned[cleaned > 0]
    if len(positive):
        scale = float(np.percentile(positive, contrast_percentile))
        if scale > 0:
            cleaned = cleaned / scale
    return np.clip(cleaned, 0.0, 1.0)


# Interpolate sparse ray-traced samples onto a regular source grid to reduce gaps.
def interpolate_source_grid(
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    output_size: int,
    source_extent: float,
    smooth_radius: int = 1,
    contrast_percentile: float = 99.0,
    max_gap_pixels: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    sx = (source_x + source_extent) / (2 * source_extent) * (output_size - 1)
    sy = (source_y + source_extent) / (2 * source_extent) * (output_size - 1)
    valid = (sx >= 0) & (sx <= output_size - 1) & (sy >= 0) & (sy <= output_size - 1)
    if np.count_nonzero(valid) < 4:
        return bilinear_deposit(source_x, source_y, values, output_size, source_extent)[:2]

    points = np.column_stack([sx[valid], sy[valid]])
    sample_values = values[valid].astype(float)
    grid_y, grid_x = np.mgrid[:output_size, :output_size]
    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    linear = griddata(points, sample_values, grid_points, method="linear").reshape(output_size, output_size)
    nearest = griddata(points, sample_values, grid_points, method="nearest").reshape(output_size, output_size)
    tree = cKDTree(points)
    distances, _ = tree.query(grid_points, k=1)
    support = distances.reshape(output_size, output_size) <= max_gap_pixels

    source = np.where(np.isfinite(linear), linear, nearest)
    source = np.where(support, source, 0.0)
    weights = support.astype(float)
    if smooth_radius > 0:
        support_blur = box_blur(weights, radius=smooth_radius)
        value_blur = box_blur(source * weights, radius=smooth_radius)
        source = np.divide(value_blur, support_blur, out=np.zeros_like(source), where=support_blur > 1e-8)
        weights = support_blur
    positive = source[source > 0]
    if len(positive):
        scale = float(np.percentile(positive, contrast_percentile))
        if scale > 0:
            source = source / scale
    return np.clip(source, 0.0, 1.0), weights


# Build the lensing matrix that maps source pixels into ray-traced image samples.
def build_bilinear_lensing_matrix(
    source_x: np.ndarray,
    source_y: np.ndarray,
    output_size: int,
    source_extent: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    sx = (source_x + source_extent) / (2 * source_extent) * (output_size - 1)
    sy = (source_y + source_extent) / (2 * source_extent) * (output_size - 1)
    valid = (sx >= 0) & (sx < output_size - 1) & (sy >= 0) & (sy < output_size - 1)
    valid_indices = np.nonzero(valid)[0]
    if len(valid_indices) == 0:
        return sparse.csr_matrix((0, output_size * output_size), dtype=float), valid

    x0 = np.floor(sx[valid]).astype(int)
    y0 = np.floor(sy[valid]).astype(int)
    fx = sx[valid] - x0
    fy = sy[valid] - y0
    rows = np.repeat(np.arange(len(valid_indices)), 4)
    cols = np.concatenate(
        [
            y0 * output_size + x0,
            y0 * output_size + (x0 + 1),
            (y0 + 1) * output_size + x0,
            (y0 + 1) * output_size + (x0 + 1),
        ]
    )
    data = np.concatenate(
        [
            (1.0 - fx) * (1.0 - fy),
            fx * (1.0 - fy),
            (1.0 - fx) * fy,
            fx * fy,
        ]
    )
    matrix = sparse.coo_matrix((data, (rows, cols)), shape=(len(valid_indices), output_size * output_size))
    return matrix.tocsr(), valid


# Construct finite-difference regularization used by semi-linear source inversion.
def source_regularization_matrix(output_size: int, kind: str = "gradient") -> sparse.csr_matrix:
    n_pix = output_size * output_size
    if kind == "zeroth":
        return sparse.identity(n_pix, format="csr")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    row = 0
    for y in range(output_size):
        for x in range(output_size):
            index = y * output_size + x
            if x + 1 < output_size:
                rows.extend([row, row])
                cols.extend([index, index + 1])
                data.extend([-1.0, 1.0])
                row += 1
            if y + 1 < output_size:
                rows.extend([row, row])
                cols.extend([index, index + output_size])
                data.extend([-1.0, 1.0])
                row += 1
    gradient = sparse.coo_matrix((data, (rows, cols)), shape=(row, n_pix)).tocsr()
    if kind == "curvature":
        return gradient.T @ gradient
    return gradient


# Solve a Warren-Dye/Suyu-style regularized linear inversion for source pixels.
def regularized_source_inversion(
    source_x: np.ndarray,
    source_y: np.ndarray,
    values: np.ndarray,
    output_size: int,
    source_extent: float,
    regularization_lambda: float = 0.03,
    regularization_type: str = "gradient",
    contrast_percentile: float = 99.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | str]]:
    lensing_matrix, valid = build_bilinear_lensing_matrix(source_x, source_y, output_size, source_extent)
    valid_values = values[valid].astype(float)
    if lensing_matrix.shape[0] < 4:
        source, weights, _ = bilinear_deposit(source_x, source_y, values, output_size, source_extent)
        return source, weights, {
            "inversion_used": 0.0,
            "inversion_lambda": regularization_lambda,
            "inversion_regularization_type": regularization_type,
            "inversion_valid_samples": float(lensing_matrix.shape[0]),
        }

    regularizer = source_regularization_matrix(output_size, kind=regularization_type)
    lambda_sqrt = float(np.sqrt(max(regularization_lambda, 0.0)))
    if lambda_sqrt > 0:
        augmented_matrix = sparse.vstack(
            [lensing_matrix, lambda_sqrt * regularizer],
            format="csr",
        )
        augmented_rhs = np.concatenate([valid_values, np.zeros(regularizer.shape[0], dtype=float)])
    else:
        augmented_matrix = lensing_matrix
        augmented_rhs = valid_values

    solution_result = sparse_linalg.lsmr(
        augmented_matrix,
        augmented_rhs,
        atol=1e-6,
        btol=1e-6,
        maxiter=max(500, output_size * 8),
    )
    solution = solution_result[0]
    source = np.asarray(solution, dtype=float).reshape(output_size, output_size)
    source = np.clip(source, 0.0, None)
    weights = np.asarray(lensing_matrix.sum(axis=0)).ravel().reshape(output_size, output_size)
    support = ndimage.gaussian_filter(weights, sigma=1.2)
    if np.any(support > 0):
        support_threshold = max(0.01 * float(support.max()), float(np.percentile(support[support > 0], 20.0)))
        source = np.where(support > support_threshold, source, 0.0)
        source *= np.clip(support / max(float(np.percentile(support[support > 0], 88.0)), 1e-8), 0.0, 1.0)
        source = ndimage.median_filter(source, size=3)
        source = ndimage.gaussian_filter(source, sigma=0.45)
        source = np.where(support > support_threshold, source, 0.0)
    positive = source[source > 0]
    if len(positive):
        scale = float(np.percentile(positive, contrast_percentile))
        if scale > 0:
            source = source / scale
    return np.clip(source, 0.0, 1.0), weights, {
        "inversion_used": 1.0,
        "inversion_lambda": regularization_lambda,
        "inversion_regularization_type": regularization_type,
        "inversion_valid_samples": float(lensing_matrix.shape[0]),
        "inversion_solver": "lsmr_augmented_system",
        "inversion_iterations": float(solution_result[2]),
        "inversion_residual_norm": float(solution_result[3]),
    }


# Crop and re-render a pixelized source around its active region for review.
def enhance_pixelized_source_display(
    source: np.ndarray,
    output_size: int | None = None,
    threshold_fraction: float = 0.12,
    padding_fraction: float = 0.35,
    gamma: float = 0.8,
) -> tuple[np.ndarray, dict[str, float | str]]:
    source = np.clip(np.asarray(source, dtype=float), 0.0, 1.0)
    if output_size is None:
        output_size = int(source.shape[0])
    if not np.any(source > 0):
        return np.zeros((output_size, output_size), dtype=float), {
            "source_display_enhanced": 0.0,
            "source_display_crop_size": "",
        }

    positive = source[source > 0]
    if len(positive):
        background = float(np.percentile(positive, 45.0))
        source = np.clip(source - background, 0.0, None)
    source = ndimage.median_filter(source, size=3)
    source = ndimage.gaussian_filter(source, sigma=0.55)
    if np.any(source > 0):
        scale = float(np.percentile(source[source > 0], 99.0))
        if scale > 0:
            source = np.clip(source / scale, 0.0, 1.0)

    peak = float(source.max())
    active = source > max(threshold_fraction * peak, 1e-8)
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
    y0 = max(0, cy - half)
    y1 = min(source.shape[0], cy + half + 1)
    x0 = max(0, cx - half)
    x1 = min(source.shape[1], cx + half + 1)
    crop = source[y0:y1, x0:x1]
    if crop.size == 0:
        crop = source

    zoom_y = output_size / crop.shape[0]
    zoom_x = output_size / crop.shape[1]
    rendered = ndimage.zoom(crop, (zoom_y, zoom_x), order=3)
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
    if gamma > 0:
        rendered = np.power(np.clip(rendered, 0.0, 1.0), gamma)
    return np.clip(rendered, 0.0, 1.0), {
        "source_display_enhanced": 1.0,
        "source_display_crop_size": float(max(crop.shape)),
    }


# Regularise a sparse ray-traced source map into a smoother visual estimate.
def regularize_sparse_source(
    source: np.ndarray,
    radius: int = 2,
    iterations: int = 3,
    active_threshold: float = 0.02,
    gamma: float = 0.75,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.clip(np.asarray(source, dtype=float), 0.0, 1.0)
    if radius <= 0 or iterations <= 0 or not np.any(source > 0):
        return source, {
            "regularization_used": 0.0,
            "regularization_radius": float(radius),
            "regularization_iterations": float(iterations),
            "regularization_active_fraction": float(np.mean(source > 0)),
        }

    peak = float(source.max())
    active = source > max(active_threshold * peak, 1e-8)
    current = source.copy()
    for _ in range(iterations):
        weights = box_blur(active.astype(float), radius=radius)
        numerator = box_blur(current * active, radius=radius)
        estimate = np.divide(numerator, weights, out=np.zeros_like(current), where=weights > 1e-8)
        support = weights > 0.01
        current = np.where(active, np.maximum(current, estimate), estimate * support)
        active = support | active

    if np.any(current > 0):
        scale = float(np.percentile(current[current > 0], 99.0))
        if scale > 0:
            current = current / scale
    current = np.clip(current, 0.0, 1.0)
    if gamma > 0:
        current = np.power(current, gamma)
    return np.clip(current, 0.0, 1.0), {
        "regularization_used": 1.0,
        "regularization_radius": float(radius),
        "regularization_iterations": float(iterations),
        "regularization_active_fraction": float(np.mean(active)),
    }


# Fit a smooth multi-component galaxy-like source model to noisy back-projected pixels.
def fit_smooth_galaxy_source(
    source: np.ndarray,
    blend_original: float = 0.2,
    min_active_fraction: float = 0.002,
    gamma: float = 0.85,
) -> tuple[np.ndarray, dict[str, float]]:
    source = np.clip(np.asarray(source, dtype=float), 0.0, 1.0)
    if not np.any(source > 0):
        return source, {
            "source_model_fit_used": 0.0,
            "source_model_active_fraction": 0.0,
            "source_model_axis_ratio": "",
            "source_model_angle_degrees": "",
        }

    peak = float(source.max())
    active = source > max(0.08 * peak, 1e-8)
    active_fraction = float(np.mean(active))
    if active_fraction < min_active_fraction:
        active = source > 0

    yy, xx = np.nonzero(active)
    weights = np.power(source[yy, xx], 1.25)
    if len(weights) < 5 or float(weights.sum()) <= 0:
        return source, {
            "source_model_fit_used": 0.0,
            "source_model_active_fraction": active_fraction,
            "source_model_axis_ratio": "",
            "source_model_angle_degrees": "",
        }

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
    eigvals = np.maximum(eigvals[order], 1.0)
    eigvecs = eigvecs[:, order]

    sigma_major = float(np.clip(np.sqrt(eigvals[0]) * 1.35, 2.0, source.shape[1] * 0.32))
    sigma_minor = float(np.clip(np.sqrt(eigvals[1]) * 1.35, 1.5, sigma_major))
    axis_ratio = sigma_minor / sigma_major if sigma_major > 0 else 1.0
    angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))

    grid_y, grid_x = np.mgrid[: source.shape[0], : source.shape[1]]
    rotated_major = (grid_x - cx) * eigvecs[0, 0] + (grid_y - cy) * eigvecs[1, 0]
    rotated_minor = (grid_x - cx) * eigvecs[0, 1] + (grid_y - cy) * eigvecs[1, 1]
    elliptical_radius = np.sqrt((rotated_major / sigma_major) ** 2 + (rotated_minor / sigma_minor) ** 2)

    # A mixed Sersic-like profile gives a less artificial source than a single Gaussian blob.
    core = np.exp(-0.5 * elliptical_radius**2)
    disk = np.exp(-1.35 * elliptical_radius)
    halo = np.exp(-0.65 * elliptical_radius)
    model = 0.45 * core + 0.45 * disk + 0.10 * halo

    # Preserve real source-plane evidence by adding smooth clumps fitted from bright components.
    evidence = ndimage.gaussian_filter(source, sigma=0.7)
    bright_threshold = float(np.percentile(evidence[evidence > 0], 86.0)) if np.any(evidence > 0) else peak
    bright = evidence >= bright_threshold
    labels, component_count = ndimage.label(bright)
    component_candidates: list[tuple[float, int]] = []
    for label in range(1, component_count + 1):
        component = labels == label
        if int(component.sum()) >= 3:
            component_candidates.append((float(evidence[component].sum()), label))
    component_candidates.sort(reverse=True)
    clump_count = 0
    for _, label in component_candidates[:6]:
        component = labels == label
        cyy, cxx = np.nonzero(component)
        cweights = np.maximum(evidence[cyy, cxx], 1e-8)
        ccx = float(np.average(cxx, weights=cweights))
        ccy = float(np.average(cyy, weights=cweights))
        cdx = cxx - ccx
        cdy = cyy - ccy
        spread = float(np.sqrt(np.average(cdx * cdx + cdy * cdy, weights=cweights)))
        clump_sigma = float(np.clip(spread * 2.4, 3.2, 8.0))
        amplitude = float(np.percentile(evidence[component], 90.0))
        clump = amplitude * np.exp(-0.5 * (((grid_x - ccx) ** 2 + (grid_y - ccy) ** 2) / (clump_sigma**2)))
        model += 0.38 * clump
        clump_count += 1

    model *= peak / max(float(model.max()), 1e-8)

    support = ndimage.binary_dilation(active, iterations=4)
    blurred_original = ndimage.gaussian_filter(source, sigma=1.8)
    blended = (1.0 - blend_original) * model + blend_original * blurred_original
    blended = np.where(model > 0.010 * peak, blended, 0.0)
    blended = np.where(support | (model > 0.05 * peak), blended, blended * 0.45)
    blended = ndimage.gaussian_filter(blended, sigma=1.35)
    if np.any(blended > 0):
        scale = float(np.percentile(blended[blended > 0], 99.5))
        if scale > 0:
            blended = blended / scale
    if gamma > 0:
        blended = np.power(np.clip(blended, 0.0, 1.0), gamma)
    return np.clip(blended, 0.0, 1.0), {
        "source_model_fit_used": 1.0,
        "source_model_active_fraction": active_fraction,
        "source_model_axis_ratio": float(axis_ratio),
        "source_model_angle_degrees": angle,
        "source_model_clump_count": float(clump_count),
    }


# Map detected image-plane arc pixels into a simple source-plane grid.
def simple_source_reconstruction(image: np.ndarray, mask: np.ndarray, output_size: int = 48) -> np.ndarray:
    """A controlled baseline that remaps detected arc pixels into a compact source plane."""
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.zeros((output_size, output_size), dtype=float)

    values = image[yy, xx]
    cx = (image.shape[1] - 1) / 2.0
    cy = (image.shape[0] - 1) / 2.0
    dx = xx - cx
    dy = yy - cy
    radius = np.hypot(dx, dy)
    angle = np.arctan2(dy, dx)

    radial = (radius - np.median(radius)) / (np.std(radius) + 1e-6)
    angular = np.angle(np.exp(1j * (angle - np.median(angle)))) / np.pi

    sx = np.clip(((angular + 1) * 0.5 * (output_size - 1)).astype(int), 0, output_size - 1)
    sy = np.clip(((radial + 2.5) / 5.0 * (output_size - 1)).astype(int), 0, output_size - 1)

    source = np.zeros((output_size, output_size), dtype=float)
    counts = np.zeros_like(source)
    np.add.at(source, (sy, sx), values)
    np.add.at(counts, (sy, sx), 1)

    filled = counts > 0
    source[filled] /= counts[filled]
    if np.any(filled):
        source[~filled] = float(source[filled].mean())
    return np.clip(box_blur(source, radius=2), 0.0, 1.0)


# Estimate the lens galaxy centre from the brightest smoothed light near the cutout centre.
def estimate_lens_centre(image: np.ndarray, search_fraction: float = 0.35) -> tuple[float, float]:
    image = np.asarray(image, dtype=float)
    smoothed = box_blur(image, radius=3)
    height, width = smoothed.shape
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    radius = min(height, width) * search_fraction

    yy, xx = np.mgrid[:height, :width]
    search = np.hypot(xx - cx, yy - cy) <= radius
    if not np.any(search):
        return cx, cy
    masked = np.where(search, smoothed, -np.inf)
    y_peak, x_peak = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return float(x_peak), float(y_peak)


# Ray-trace detected real-image pixels with a simple SIS lens estimated from the cutout itself.
def reconstruct_source_with_estimated_sis(
    image: np.ndarray,
    mask: np.ndarray,
    output_size: int = 128,
    lens_center: tuple[float, float] | None = None,
    theta_e_pixels: float | None = None,
    source_extent_pixels: float | None = None,
    smooth_radius: int = 1,
    contrast_percentile: float = 99.0,
    deposition: str = "nearest",
    inversion_lambda: float = 0.03,
    inversion_regularization_type: str = "gradient",
) -> tuple[np.ndarray, dict[str, float]]:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        empty = np.zeros((output_size, output_size), dtype=float)
        return empty, {
            "lens_center_x": np.nan,
            "lens_center_y": np.nan,
            "theta_e_pixels": np.nan,
            "source_extent_pixels": np.nan,
            "valid_ray_fraction": 0.0,
            "smooth_radius": float(smooth_radius),
            "contrast_percentile": float(contrast_percentile),
        }

    if lens_center is None:
        lens_cx, lens_cy = estimate_lens_centre(image)
    else:
        lens_cx, lens_cy = lens_center

    dx = xx.astype(float) - lens_cx
    dy = yy.astype(float) - lens_cy
    radius = np.hypot(dx, dy)
    valid_radius = radius > 1e-6
    if theta_e_pixels is None:
        theta_e_pixels = float(np.median(radius[valid_radius])) if np.any(valid_radius) else 0.0

    beta_x = dx.copy()
    beta_y = dy.copy()
    beta_x[valid_radius] = dx[valid_radius] - theta_e_pixels * dx[valid_radius] / radius[valid_radius]
    beta_y[valid_radius] = dy[valid_radius] - theta_e_pixels * dy[valid_radius] / radius[valid_radius]

    if source_extent_pixels is None:
        extent = float(np.percentile(np.hypot(beta_x, beta_y), 95)) if len(beta_x) else 1.0
        source_extent_pixels = max(3.0, extent * 1.25)

    values = image[yy, xx]
    if deposition == "bilinear":
        source, counts, valid = bilinear_deposit(beta_x, beta_y, values, output_size, source_extent_pixels)
        source = clean_deposited_source(source, counts, smooth_radius=smooth_radius, contrast_percentile=contrast_percentile)
    elif deposition == "linear_grid":
        source, counts = interpolate_source_grid(
            beta_x,
            beta_y,
            values,
            output_size,
            source_extent_pixels,
            smooth_radius=smooth_radius,
            contrast_percentile=contrast_percentile,
        )
        valid = (
            (beta_x >= -source_extent_pixels)
            & (beta_x <= source_extent_pixels)
            & (beta_y >= -source_extent_pixels)
            & (beta_y <= source_extent_pixels)
        )
    elif deposition == "regularized_inversion":
        source, counts, inversion_stats = regularized_source_inversion(
            beta_x,
            beta_y,
            values,
            output_size,
            source_extent_pixels,
            regularization_lambda=inversion_lambda,
            regularization_type=inversion_regularization_type,
            contrast_percentile=contrast_percentile,
        )
        valid = (
            (beta_x >= -source_extent_pixels)
            & (beta_x <= source_extent_pixels)
            & (beta_y >= -source_extent_pixels)
            & (beta_y <= source_extent_pixels)
        )
    else:
        sx = np.floor((beta_x + source_extent_pixels) / (2 * source_extent_pixels) * output_size).astype(int)
        sy = np.floor((beta_y + source_extent_pixels) / (2 * source_extent_pixels) * output_size).astype(int)
        valid = (sx >= 0) & (sx < output_size) & (sy >= 0) & (sy < output_size)

        source = np.zeros((output_size, output_size), dtype=float)
        counts = np.zeros_like(source)
        np.add.at(source, (sy[valid], sx[valid]), values[valid])
        np.add.at(counts, (sy[valid], sx[valid]), 1)

        filled = counts > 0
        source[filled] /= counts[filled]
        if smooth_radius > 0:
            source = box_blur(source, radius=smooth_radius)
        peak = float(np.percentile(source[source > 0], contrast_percentile)) if np.any(source > 0) else 0.0
        if peak > 0:
            source = source / peak

    return np.clip(source, 0.0, 1.0), {
        "lens_center_x": float(lens_cx),
        "lens_center_y": float(lens_cy),
        "theta_e_pixels": float(theta_e_pixels),
        "source_extent_pixels": float(source_extent_pixels),
        "valid_ray_fraction": float(np.mean(valid)) if len(valid) else 0.0,
        "smooth_radius": float(smooth_radius),
        "contrast_percentile": float(contrast_percentile),
        "deposition": deposition,
        **(inversion_stats if deposition == "regularized_inversion" else {}),
    }


# Ray-trace detected real-image pixels through a catalogue SIE-style elliptical lens model.
def reconstruct_source_with_catalog_sie(
    image: np.ndarray,
    mask: np.ndarray,
    lens_center: tuple[float, float],
    theta_e_pixels: float,
    axis_ratio: float,
    position_angle_degrees: float,
    output_size: int = 128,
    source_extent_pixels: float | None = None,
    smooth_radius: int = 1,
    contrast_percentile: float = 99.0,
    deposition: str = "nearest",
    inversion_lambda: float = 0.03,
    inversion_regularization_type: str = "gradient",
) -> tuple[np.ndarray, dict[str, float | str]]:
    yy, xx = np.nonzero(mask)
    q = float(np.clip(axis_ratio, 0.2, 1.0))
    if len(xx) == 0:
        empty = np.zeros((output_size, output_size), dtype=float)
        return empty, {
            "lens_model_type": "catalog_sie_elliptical_potential",
            "lens_center_x": float(lens_center[0]),
            "lens_center_y": float(lens_center[1]),
            "theta_e_pixels": float(theta_e_pixels),
            "lens_axis_ratio": q,
            "position_angle_degrees": float(position_angle_degrees),
            "source_extent_pixels": np.nan,
            "valid_ray_fraction": 0.0,
            "smooth_radius": float(smooth_radius),
            "contrast_percentile": float(contrast_percentile),
        }

    lens_cx, lens_cy = lens_center
    dx = xx.astype(float) - lens_cx
    dy = yy.astype(float) - lens_cy
    angle = np.radians(position_angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    major = dx * cos_a + dy * sin_a
    minor = -dx * sin_a + dy * cos_a
    elliptical_radius = np.sqrt(q * major**2 + minor**2 / q)
    valid_radius = elliptical_radius > 1e-6

    alpha_major = np.zeros_like(major)
    alpha_minor = np.zeros_like(minor)
    alpha_major[valid_radius] = theta_e_pixels * q * major[valid_radius] / elliptical_radius[valid_radius]
    alpha_minor[valid_radius] = theta_e_pixels * minor[valid_radius] / (q * elliptical_radius[valid_radius])

    alpha_x = alpha_major * cos_a - alpha_minor * sin_a
    alpha_y = alpha_major * sin_a + alpha_minor * cos_a
    beta_x = dx - alpha_x
    beta_y = dy - alpha_y

    if source_extent_pixels is None:
        extent = float(np.percentile(np.hypot(beta_x, beta_y), 95)) if len(beta_x) else 1.0
        source_extent_pixels = max(3.0, extent * 1.25)

    values = image[yy, xx]
    if deposition == "bilinear":
        source, counts, valid = bilinear_deposit(beta_x, beta_y, values, output_size, source_extent_pixels)
        source = clean_deposited_source(source, counts, smooth_radius=smooth_radius, contrast_percentile=contrast_percentile)
    elif deposition == "linear_grid":
        source, counts = interpolate_source_grid(
            beta_x,
            beta_y,
            values,
            output_size,
            source_extent_pixels,
            smooth_radius=smooth_radius,
            contrast_percentile=contrast_percentile,
        )
        valid = (
            (beta_x >= -source_extent_pixels)
            & (beta_x <= source_extent_pixels)
            & (beta_y >= -source_extent_pixels)
            & (beta_y <= source_extent_pixels)
        )
    elif deposition == "regularized_inversion":
        source, counts, inversion_stats = regularized_source_inversion(
            beta_x,
            beta_y,
            values,
            output_size,
            source_extent_pixels,
            regularization_lambda=inversion_lambda,
            regularization_type=inversion_regularization_type,
            contrast_percentile=contrast_percentile,
        )
        valid = (
            (beta_x >= -source_extent_pixels)
            & (beta_x <= source_extent_pixels)
            & (beta_y >= -source_extent_pixels)
            & (beta_y <= source_extent_pixels)
        )
    else:
        sx = np.floor((beta_x + source_extent_pixels) / (2 * source_extent_pixels) * output_size).astype(int)
        sy = np.floor((beta_y + source_extent_pixels) / (2 * source_extent_pixels) * output_size).astype(int)
        valid = (sx >= 0) & (sx < output_size) & (sy >= 0) & (sy < output_size)

        source = np.zeros((output_size, output_size), dtype=float)
        counts = np.zeros_like(source)
        np.add.at(source, (sy[valid], sx[valid]), values[valid])
        np.add.at(counts, (sy[valid], sx[valid]), 1)

        filled = counts > 0
        source[filled] /= counts[filled]
        if smooth_radius > 0:
            source = box_blur(source, radius=smooth_radius)
        peak = float(np.percentile(source[source > 0], contrast_percentile)) if np.any(source > 0) else 0.0
        if peak > 0:
            source = source / peak

    return np.clip(source, 0.0, 1.0), {
        "lens_model_type": "catalog_sie_elliptical_potential",
        "lens_center_x": float(lens_cx),
        "lens_center_y": float(lens_cy),
        "theta_e_pixels": float(theta_e_pixels),
        "lens_axis_ratio": q,
        "position_angle_degrees": float(position_angle_degrees),
        "source_extent_pixels": float(source_extent_pixels),
        "valid_ray_fraction": float(np.mean(valid)) if len(valid) else 0.0,
        "smooth_radius": float(smooth_radius),
        "contrast_percentile": float(contrast_percentile),
        "deposition": deposition,
        **(inversion_stats if deposition == "regularized_inversion" else {}),
    }


# Ray-trace image-plane pixels through the same catalogue SIE-style model used elsewhere.
def catalog_sie_raytrace_pixels(
    xx: np.ndarray,
    yy: np.ndarray,
    lens_center: tuple[float, float],
    theta_e_pixels: float,
    axis_ratio: float,
    position_angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    lens_cx, lens_cy = lens_center
    q = float(np.clip(axis_ratio, 0.2, 1.0))
    dx = xx.astype(float) - lens_cx
    dy = yy.astype(float) - lens_cy
    angle = np.radians(position_angle_degrees)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)

    major = dx * cos_a + dy * sin_a
    minor = -dx * sin_a + dy * cos_a
    elliptical_radius = np.sqrt(q * major**2 + minor**2 / q)
    valid_radius = elliptical_radius > 1e-6

    alpha_major = np.zeros_like(major, dtype=float)
    alpha_minor = np.zeros_like(minor, dtype=float)
    alpha_major[valid_radius] = theta_e_pixels * q * major[valid_radius] / elliptical_radius[valid_radius]
    alpha_minor[valid_radius] = theta_e_pixels * minor[valid_radius] / (q * elliptical_radius[valid_radius])

    alpha_x = alpha_major * cos_a - alpha_minor * sin_a
    alpha_y = alpha_major * sin_a + alpha_minor * cos_a
    return dx - alpha_x, dy - alpha_y


# Fit a forward-lensed Sersic source to the observed arc pixels.
def fit_forward_lensed_sersic_source(
    image: np.ndarray,
    mask: np.ndarray,
    lens_center: tuple[float, float],
    theta_e_pixels: float,
    axis_ratio: float,
    position_angle_degrees: float,
    output_size: int = 128,
    source_extent_pixels: float | None = None,
    max_fit_pixels: int = 3500,
) -> tuple[np.ndarray, dict[str, float | str]]:
    yy, xx = np.nonzero(mask)
    if len(xx) < 12:
        return np.zeros((output_size, output_size), dtype=float), {
            "forward_source_fit_used": 0.0,
            "forward_source_fit_status": "not_enough_arc_pixels",
            "forward_fit_pixels": float(len(xx)),
        }

    values = np.asarray(image[yy, xx], dtype=float)
    if len(values) > max_fit_pixels:
        order = np.argsort(values)[-max_fit_pixels:]
        xx = xx[order]
        yy = yy[order]
        values = values[order]

    beta_x, beta_y = catalog_sie_raytrace_pixels(
        xx,
        yy,
        lens_center=lens_center,
        theta_e_pixels=theta_e_pixels,
        axis_ratio=axis_ratio,
        position_angle_degrees=position_angle_degrees,
    )
    if source_extent_pixels is None:
        extent = float(np.percentile(np.hypot(beta_x, beta_y), 97.0)) if len(beta_x) else 1.0
        source_extent_pixels = max(4.0, extent * 1.35)

    positive = values[values > 0]
    scale = float(np.percentile(positive, 98.0)) if len(positive) else 1.0
    if scale <= 0:
        scale = 1.0
    fit_values = np.clip(values / scale, 0.0, 1.5)

    weights = np.maximum(fit_values, 0.08)
    cx = float(np.average(beta_x, weights=weights))
    cy = float(np.average(beta_y, weights=weights))
    dx = beta_x - cx
    dy = beta_y - cy
    cov_xx = float(np.average(dx * dx, weights=weights))
    cov_yy = float(np.average(dy * dy, weights=weights))
    cov_xy = float(np.average(dx * dy, weights=weights))
    covariance = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]], dtype=float)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = np.maximum(eigvals[order], 0.5)
    eigvecs = eigvecs[:, order]
    init_radius = float(np.clip(np.sqrt(eigvals[0]) * 0.75, 1.0, source_extent_pixels * 0.65))
    init_q = float(np.clip(np.sqrt(eigvals[1] / eigvals[0]), 0.25, 0.95))
    init_angle = float(np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))
    init_amplitude = float(np.clip(np.percentile(fit_values, 92.0), 0.2, 1.2))

    lower = np.array(
        [
            -source_extent_pixels,
            -source_extent_pixels,
            0.6,
            0.18,
            -180.0,
            0.55,
            0.05,
        ],
        dtype=float,
    )
    upper = np.array(
        [
            source_extent_pixels,
            source_extent_pixels,
            source_extent_pixels,
            1.0,
            180.0,
            5.5,
            2.5,
        ],
        dtype=float,
    )
    initial = np.array([cx, cy, init_radius, init_q, init_angle, 1.2, init_amplitude], dtype=float)
    initial = np.clip(initial, lower + 1e-4, upper - 1e-4)

    robust_sigma = max(float(np.std(fit_values)), 0.08)

    def residuals(params: np.ndarray) -> np.ndarray:
        model = sersic_source_intensity(
            beta_x,
            beta_y,
            center_x=float(params[0]),
            center_y=float(params[1]),
            radius_major=float(params[2]),
            axis_ratio=float(params[3]),
            angle_degrees=float(params[4]),
            sersic_n=float(params[5]),
            amplitude=float(params[6]),
        )
        return (model - fit_values) / robust_sigma

    result = least_squares(
        residuals,
        initial,
        bounds=(lower, upper),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=500,
    )
    params = {
        "source_center_x": float(result.x[0]),
        "source_center_y": float(result.x[1]),
        "source_radius_major": float(result.x[2]),
        "source_axis_ratio": float(result.x[3]),
        "source_angle_degrees": float(result.x[4]),
        "source_sersic_n": float(result.x[5]),
        "source_amplitude": float(result.x[6]),
    }
    source = render_sersic_source_grid(output_size, source_extent_pixels, params)
    model_values = sersic_source_intensity(
        beta_x,
        beta_y,
        center_x=params["source_center_x"],
        center_y=params["source_center_y"],
        radius_major=params["source_radius_major"],
        axis_ratio=params["source_axis_ratio"],
        angle_degrees=params["source_angle_degrees"],
        sersic_n=params["source_sersic_n"],
        amplitude=params["source_amplitude"],
    )
    rms = float(np.sqrt(np.mean((model_values - fit_values) ** 2))) if len(fit_values) else np.nan
    return source, {
        "forward_source_fit_used": 1.0,
        "forward_source_fit_status": "ran",
        "forward_fit_pixels": float(len(fit_values)),
        "forward_fit_cost": float(result.cost),
        "forward_fit_rms": rms,
        "forward_source_extent_pixels": float(source_extent_pixels),
        **params,
    }


# Ray-trace detected real-image pixels through lenstronomy's SIE lens model.
def reconstruct_source_with_lenstronomy_sie(
    image: np.ndarray,
    mask: np.ndarray,
    lens_center: tuple[float, float],
    theta_e_pixels: float,
    axis_ratio: float,
    position_angle_degrees: float,
    output_size: int = 128,
    source_extent_pixels: float | None = None,
    smooth_radius: int = 1,
    contrast_percentile: float = 99.0,
    deposition: str = "nearest",
    inversion_lambda: float = 0.03,
    inversion_regularization_type: str = "gradient",
) -> tuple[np.ndarray, dict[str, float | str]]:
    LensModel = _import_lens_model()
    from lenstronomy.Util import param_util

    lens_cx, lens_cy = lens_center
    q = float(np.clip(axis_ratio, 0.2, 1.0))
    phi = float(np.radians(position_angle_degrees))
    e1, e2 = param_util.phi_q2_ellipticity(phi, q)
    lens_model = LensModel(lens_model_list=["SIE"])
    kwargs_lens = [
        {
            "theta_E": float(theta_e_pixels),
            "e1": float(e1),
            "e2": float(e2),
            "center_x": 0.0,
            "center_y": 0.0,
        }
    ]

    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        empty = np.zeros((output_size, output_size), dtype=float)
        return empty, {
            "lens_model_type": "lenstronomy_sie",
            "lens_center_x": float(lens_cx),
            "lens_center_y": float(lens_cy),
            "theta_e_pixels": float(theta_e_pixels),
            "lens_axis_ratio": q,
            "position_angle_degrees": float(position_angle_degrees),
            "lens_e1": float(e1),
            "lens_e2": float(e2),
            "source_extent_pixels": np.nan,
            "valid_ray_fraction": 0.0,
            "smooth_radius": float(smooth_radius),
            "contrast_percentile": float(contrast_percentile),
            "deposition": deposition,
        }

    image_x = xx.astype(float) - lens_cx
    image_y = yy.astype(float) - lens_cy
    beta_x, beta_y = lens_model.ray_shooting(image_x, image_y, kwargs_lens)

    if source_extent_pixels is None:
        extent = float(np.percentile(np.hypot(beta_x, beta_y), 95)) if len(beta_x) else 1.0
        source_extent_pixels = max(3.0, extent * 1.25)

    values = image[yy, xx]
    if deposition == "bilinear":
        source, counts, valid = bilinear_deposit(beta_x, beta_y, values, output_size, source_extent_pixels)
        source = clean_deposited_source(source, counts, smooth_radius=smooth_radius, contrast_percentile=contrast_percentile)
    elif deposition == "linear_grid":
        source, _ = interpolate_source_grid(
            beta_x,
            beta_y,
            values,
            output_size,
            source_extent_pixels,
            smooth_radius=smooth_radius,
            contrast_percentile=contrast_percentile,
        )
        valid = (
            (beta_x >= -source_extent_pixels)
            & (beta_x <= source_extent_pixels)
            & (beta_y >= -source_extent_pixels)
            & (beta_y <= source_extent_pixels)
        )
    elif deposition == "regularized_inversion":
        source, counts, inversion_stats = regularized_source_inversion(
            beta_x,
            beta_y,
            values,
            output_size,
            source_extent_pixels,
            regularization_lambda=inversion_lambda,
            regularization_type=inversion_regularization_type,
            contrast_percentile=contrast_percentile,
        )
        valid = (
            (beta_x >= -source_extent_pixels)
            & (beta_x <= source_extent_pixels)
            & (beta_y >= -source_extent_pixels)
            & (beta_y <= source_extent_pixels)
        )
    else:
        sx = np.floor((beta_x + source_extent_pixels) / (2 * source_extent_pixels) * output_size).astype(int)
        sy = np.floor((beta_y + source_extent_pixels) / (2 * source_extent_pixels) * output_size).astype(int)
        valid = (sx >= 0) & (sx < output_size) & (sy >= 0) & (sy < output_size)

        source = np.zeros((output_size, output_size), dtype=float)
        counts = np.zeros_like(source)
        np.add.at(source, (sy[valid], sx[valid]), values[valid])
        np.add.at(counts, (sy[valid], sx[valid]), 1)

        filled = counts > 0
        source[filled] /= counts[filled]
        if smooth_radius > 0:
            source = box_blur(source, radius=smooth_radius)
        peak = float(np.percentile(source[source > 0], contrast_percentile)) if np.any(source > 0) else 0.0
        if peak > 0:
            source = source / peak

    return np.clip(source, 0.0, 1.0), {
        "lens_model_type": "lenstronomy_sie",
        "lens_center_x": float(lens_cx),
        "lens_center_y": float(lens_cy),
        "theta_e_pixels": float(theta_e_pixels),
        "lens_axis_ratio": q,
        "position_angle_degrees": float(position_angle_degrees),
        "lens_e1": float(e1),
        "lens_e2": float(e2),
        "source_extent_pixels": float(source_extent_pixels),
        "valid_ray_fraction": float(np.mean(valid)) if len(valid) else 0.0,
        "smooth_radius": float(smooth_radius),
        "contrast_percentile": float(contrast_percentile),
        "deposition": deposition,
        **(inversion_stats if deposition == "regularized_inversion" else {}),
    }


# Import only the lenstronomy lens model needed for physics-based source reconstruction.
def _import_lens_model():
    try:
        from lenstronomy.LensModel.lens_model import LensModel
    except ImportError as exc:
        raise RuntimeError(
            "lenstronomy is not installed. Run `pip install -r requirements.txt` in your project venv."
        ) from exc
    return LensModel


# Convert lenstronomy-style ellipticity components into an approximate axis ratio and angle.
def ellipticity_to_q_phi(e1: float, e2: float) -> tuple[float, float]:
    ellipticity = float(np.clip(np.hypot(e1, e2), 0.0, 0.8))
    q = float(np.clip((1.0 - ellipticity) / (1.0 + ellipticity), 0.2, 1.0))
    phi = 0.5 * float(np.arctan2(e2, e1))
    return q, phi


# Approximate SIE ray tracing used when lenstronomy is unavailable.
def approximate_sie_ray_shooting(
    image_x: np.ndarray,
    image_y: np.ndarray,
    lens_metadata: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    theta_e = float(lens_metadata["theta_E"])
    cx = float(lens_metadata.get("lens_center_x", 0.0))
    cy = float(lens_metadata.get("lens_center_y", 0.0))
    q, phi = ellipticity_to_q_phi(float(lens_metadata["lens_e1"]), float(lens_metadata["lens_e2"]))

    dx = image_x - cx
    dy = image_y - cy
    cos_a = np.cos(phi)
    sin_a = np.sin(phi)
    major = dx * cos_a + dy * sin_a
    minor = -dx * sin_a + dy * cos_a

    if q > 0.995:
        radius = np.hypot(major, minor)
        valid = radius > 1e-8
        alpha_major = np.zeros_like(major, dtype=float)
        alpha_minor = np.zeros_like(minor, dtype=float)
        alpha_major[valid] = theta_e * major[valid] / radius[valid]
        alpha_minor[valid] = theta_e * minor[valid] / radius[valid]
    else:
        eps = np.sqrt(max(1.0 - q * q, 1e-8))
        psi = np.sqrt(q * q * major * major + minor * minor)
        valid = psi > 1e-8
        alpha_major = np.zeros_like(major, dtype=float)
        alpha_minor = np.zeros_like(minor, dtype=float)
        prefactor = theta_e * q / eps
        alpha_major[valid] = prefactor * np.arctan(eps * major[valid] / psi[valid])
        alpha_minor[valid] = prefactor * np.arctanh(np.clip(eps * minor[valid] / psi[valid], -0.999999, 0.999999))

    alpha_x = alpha_major * cos_a - alpha_minor * sin_a
    alpha_y = alpha_major * sin_a + alpha_minor * cos_a
    return image_x - alpha_x, image_y - alpha_y


# Reconstruct the source by ray-shooting detected image pixels through the known simulated lens.
def reconstruct_source_with_lens_model(
    image: np.ndarray,
    mask: np.ndarray,
    lens_metadata: dict[str, float],
    delta_pix: float = 0.05,
    output_size: int = 64,
    source_extent: float = 0.6,
    deposition: str = "nearest",
    smooth_radius: int = 1,
    inversion_lambda: float = 0.03,
    inversion_regularization_type: str = "gradient",
    ray_tracer: str = "approximate",
) -> np.ndarray:
    yy, xx = np.nonzero(mask)
    if len(xx) == 0:
        return np.zeros((output_size, output_size), dtype=float)

    height, width = image.shape
    image_x = (xx.astype(float) - (width - 1) / 2.0) * delta_pix
    image_y = (yy.astype(float) - (height - 1) / 2.0) * delta_pix
    if ray_tracer == "lenstronomy":
        LensModel = _import_lens_model()
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
        source_x, source_y = lens_model.ray_shooting(image_x, image_y, kwargs_lens)
    else:
        source_x, source_y = approximate_sie_ray_shooting(image_x, image_y, lens_metadata)

    values = image[yy, xx]
    if deposition == "bilinear":
        source, counts, _ = bilinear_deposit(source_x, source_y, values, output_size, source_extent)
        return clean_deposited_source(source, counts, smooth_radius=smooth_radius, contrast_percentile=99.0)
    if deposition == "linear_grid":
        source, _ = interpolate_source_grid(
            source_x,
            source_y,
            values,
            output_size,
            source_extent,
            smooth_radius=smooth_radius,
            contrast_percentile=99.0,
        )
        return source
    if deposition == "regularized_inversion":
        source, _, _ = regularized_source_inversion(
            source_x,
            source_y,
            values,
            output_size,
            source_extent,
            regularization_lambda=inversion_lambda,
            regularization_type=inversion_regularization_type,
            contrast_percentile=99.5,
        )
        return source

    sx = np.floor((source_x + source_extent) / (2 * source_extent) * output_size).astype(int)
    sy = np.floor((source_y + source_extent) / (2 * source_extent) * output_size).astype(int)
    valid = (sx >= 0) & (sx < output_size) & (sy >= 0) & (sy < output_size)

    source = np.zeros((output_size, output_size), dtype=float)
    counts = np.zeros_like(source)
    np.add.at(source, (sy[valid], sx[valid]), values[valid])
    np.add.at(counts, (sy[valid], sx[valid]), 1)

    filled = counts > 0
    source[filled] /= counts[filled]
    if np.any(filled):
        source[~filled] = float(np.percentile(source[filled], 5))
    source = box_blur(source, radius=smooth_radius)
    peak = float(source.max())
    return np.clip(source / peak, 0.0, 1.0) if peak > 0 else source
