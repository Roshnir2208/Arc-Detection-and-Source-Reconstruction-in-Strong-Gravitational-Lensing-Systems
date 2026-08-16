from __future__ import annotations

import numpy as np


# Scale image intensities robustly so detection thresholds behave consistently.
def normalize_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    lo = float(np.percentile(image, 1))
    hi = float(np.percentile(image, 99.5))
    if hi <= lo:
        return np.zeros_like(image, dtype=float)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


# Smooth small-scale noise using a simple box filter.
def box_blur(image: np.ndarray, radius: int = 2) -> np.ndarray:
    if radius <= 0:
        return image.astype(float)
    padded = np.pad(image.astype(float), radius, mode="edge")
    out = np.zeros_like(image, dtype=float)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out += padded[dy : dy + image.shape[0], dx : dx + image.shape[1]]
    return out / float(size * size)


# Apply a small 2D convolution kernel to an image.
def convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kernel = np.asarray(kernel, dtype=float)
    pad_y = kernel.shape[0] // 2
    pad_x = kernel.shape[1] // 2
    padded = np.pad(np.asarray(image, dtype=float), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
    out = np.zeros_like(image, dtype=float)
    for y in range(kernel.shape[0]):
        for x in range(kernel.shape[1]):
            out += kernel[y, x] * padded[y : y + image.shape[0], x : x + image.shape[1]]
    return out


# Build a Laplacian-of-Gaussian kernel for arc-edge response detection.
def log_kernel(sigma: float = 1.4, radius: int | None = None) -> np.ndarray:
    if radius is None:
        radius = max(2, int(round(3 * sigma)))
    y, x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    r2 = x**2 + y**2
    sigma2 = sigma**2
    kernel = ((r2 - 2 * sigma2) / (sigma2**2)) * np.exp(-r2 / (2 * sigma2))
    kernel -= kernel.mean()
    norm = np.sum(np.abs(kernel))
    return kernel / norm if norm > 0 else kernel


# Compute the positive LoG response used to highlight curved bright structures.
def laplacian_of_gaussian_response(image: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    response = -convolve2d(image, log_kernel(sigma=sigma))
    return normalize_image(np.maximum(response, 0.0))


# Expand a binary mask so nearby candidate pixels are connected.
def dilate_mask(mask: np.ndarray, radius: int = 2) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if radius <= 0:
        return mask
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    size = 2 * radius + 1
    for dy in range(size):
        for dx in range(size):
            out |= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


# Mark local image features using SIFT when available, with ORB as a fallback.
def sift_keypoint_mask(image: np.ndarray, dilate_radius: int = 3, max_features: int = 250) -> np.ndarray:
    """Return a keypoint support mask using OpenCV SIFT, or ORB if SIFT is unavailable."""
    try:
        import cv2
    except ImportError:
        return np.zeros_like(image, dtype=bool)

    norm = normalize_image(image)
    image_u8 = (norm * 255).astype(np.uint8)

    detector = None
    if hasattr(cv2, "SIFT_create"):
        detector = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.01, edgeThreshold=8)
    elif hasattr(cv2, "ORB_create"):
        detector = cv2.ORB_create(nfeatures=max_features, fastThreshold=8)

    if detector is None:
        return np.zeros_like(image, dtype=bool)

    keypoints = detector.detect(image_u8, None)
    mask = np.zeros_like(image, dtype=bool)
    for keypoint in keypoints:
        x, y = keypoint.pt
        ix = int(round(x))
        iy = int(round(y))
        if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
            mask[iy, ix] = True
    return dilate_mask(mask, radius=dilate_radius)


# Optionally subtract smooth central lens light before arc segmentation.
def suppress_central_lens_light(
    image: np.ndarray,
    radius_fraction: float = 0.22,
    strength: float = 0.75,
    blur_radius: int = 8,
) -> np.ndarray:
    """Subtract a smooth central-light estimate while preserving off-centre arc structure."""
    norm = normalize_image(image)
    if strength <= 0 or radius_fraction <= 0:
        return norm

    yy, xx = np.indices(norm.shape)
    cy = (norm.shape[0] - 1) / 2.0
    cx = (norm.shape[1] - 1) / 2.0
    radius = np.hypot(yy - cy, xx - cx)
    sigma = max(radius_fraction * min(norm.shape), 1.0)
    central_weight = np.exp(-0.5 * (radius / sigma) ** 2)

    smooth_lens = box_blur(norm, radius=blur_radius)
    cleaned = norm - strength * central_weight * smooth_lens
    return normalize_image(np.clip(cleaned, 0.0, None))


# Remove disconnected detections that are too small to be useful arcs.
def remove_small_components(mask: np.ndarray, min_size: int = 24) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    keep = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            component = []
            while stack:
                cy, cx = stack.pop()
                component.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            if len(component) >= min_size:
                yy, xx = zip(*component)
                keep[np.array(yy), np.array(xx)] = True
    return keep


# Split a binary mask into connected components for post-detection filtering.
def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    height, width = mask.shape

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                for ny in (cy - 1, cy, cy + 1):
                    for nx in (cx - 1, cx, cx + 1):
                        if ny == cy and nx == cx:
                            continue
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            component = np.zeros_like(mask, dtype=bool)
            yy, xx = zip(*pixels)
            component[np.array(yy), np.array(xx)] = True
            components.append(component)
    return components


# Remove likely false-positive components using brightness, radial, and shape consistency checks.
def filter_false_positive_components(
    image: np.ndarray,
    mask: np.ndarray,
    min_area: int = 16,
    max_area_fraction: float = 0.40,
    min_mean_brightness: float = 0.12,
    min_radius_fraction: float = 0.08,
    max_radius_fraction: float = 0.75,
    min_elongation: float = 1.08,
    max_axis_ratio: float = 0.92,
) -> np.ndarray:
    """Keep connected components that look more like arcs than compact noise or central light."""
    mask = np.asarray(mask, dtype=bool)
    norm = normalize_image(image)
    height, width = mask.shape
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    scale = float(min(mask.shape))
    max_area = max(min_area, int(max_area_fraction * mask.size))
    keep = np.zeros_like(mask, dtype=bool)

    for component in connected_components(mask):
        yy, xx = np.nonzero(component)
        area = len(xx)
        if area < min_area or area > max_area:
            continue

        mean_brightness = float(norm[yy, xx].mean())
        if area < 3 * min_area and mean_brightness < min_mean_brightness:
            continue

        radii = np.hypot(xx.astype(float) - cx, yy.astype(float) - cy)
        median_radius_fraction = float(np.median(radii) / scale)
        if median_radius_fraction < min_radius_fraction or median_radius_fraction > max_radius_fraction:
            continue

        coords = np.column_stack([xx.astype(float), yy.astype(float)])
        if len(coords) >= 3:
            centred = coords - coords.mean(axis=0)
            cov = np.cov(centred, rowvar=False)
            values = np.linalg.eigvalsh(cov)
            major = float(np.sqrt(max(values[-1], 0.0)))
            minor = float(np.sqrt(max(values[0], 0.0)))
            elongation = major / (minor + 1e-6)
            axis_ratio = minor / (major + 1e-6)
        else:
            elongation = 0.0
            axis_ratio = 1.0

        if area < 5 * min_area and elongation < min_elongation and axis_ratio > max_axis_ratio:
            continue
        keep |= component

    return keep


# Run the full classical arc detector and return a binary predicted arc mask.
def detect_arc_mask(
    image: np.ndarray,
    threshold_percentile: float = 96.0,
    blur_radius: int = 2,
    min_component_size: int = 24,
    suppress_centre_radius: float = 0.08,
    use_log: bool = True,
    log_sigma: float = 1.4,
    log_percentile: float = 98.0,
    use_sift: bool = True,
    sift_dilate_radius: int = 3,
    subtract_central_lens: bool = False,
    central_lens_radius: float = 0.22,
    central_lens_strength: float = 0.75,
    central_lens_blur_radius: int = 8,
    false_positive_filter: bool = False,
    fp_min_area: int = 16,
    fp_min_mean_brightness: float = 0.12,
    fp_min_radius_fraction: float = 0.08,
    fp_max_radius_fraction: float = 0.48,
    fp_min_elongation: float = 1.08,
    fp_max_axis_ratio: float = 0.92,
) -> np.ndarray:
    """Segment lensing arcs using brightness thresholding plus LoG and optional SIFT/ORB support."""
    if subtract_central_lens:
        norm = suppress_central_lens_light(
            image,
            radius_fraction=central_lens_radius,
            strength=central_lens_strength,
            blur_radius=central_lens_blur_radius,
        )
    else:
        norm = normalize_image(image)
    smooth = box_blur(norm, radius=blur_radius)
    threshold = float(np.percentile(smooth, threshold_percentile))
    brightness_mask = smooth >= threshold

    if use_log:
        log_response = laplacian_of_gaussian_response(smooth, sigma=log_sigma)
        log_threshold = float(np.percentile(log_response, log_percentile))
        log_mask = log_response >= log_threshold
    else:
        log_mask = np.zeros_like(brightness_mask, dtype=bool)

    if use_sift:
        keypoint_mask = sift_keypoint_mask(smooth, dilate_radius=sift_dilate_radius)
    else:
        keypoint_mask = np.zeros_like(brightness_mask, dtype=bool)

    # LoG recovers curved arc edges; SIFT/ORB keypoints add local feature support where available.
    mask = brightness_mask | (log_mask & dilate_mask(brightness_mask, radius=3)) | (keypoint_mask & log_mask)

    yy, xx = np.indices(mask.shape)
    cy = (mask.shape[0] - 1) / 2.0
    cx = (mask.shape[1] - 1) / 2.0
    radius = np.hypot(yy - cy, xx - cx)
    mask &= radius > suppress_centre_radius * min(mask.shape)

    mask = remove_small_components(mask, min_size=min_component_size)
    if false_positive_filter:
        mask = filter_false_positive_components(
            image,
            mask,
            min_area=max(fp_min_area, min_component_size),
            min_mean_brightness=fp_min_mean_brightness,
            min_radius_fraction=fp_min_radius_fraction,
            max_radius_fraction=fp_max_radius_fraction,
            min_elongation=fp_min_elongation,
            max_axis_ratio=fp_max_axis_ratio,
        )
    return mask
