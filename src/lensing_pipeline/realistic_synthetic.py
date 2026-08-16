from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from lensing_pipeline.reconstruction import approximate_sie_ray_shooting


@dataclass
class RealisticSyntheticCase:
    image: np.ndarray
    clean: np.ndarray
    lensed_source: np.ndarray
    mask: np.ndarray
    source: np.ndarray
    source_rgb: np.ndarray
    metadata: dict[str, float]


def _normalise(image: np.ndarray, percentile: float = 99.7) -> np.ndarray:
    image = np.clip(np.asarray(image, dtype=float), 0.0, None)
    positive = image[image > 0]
    if len(positive):
        scale = float(np.percentile(positive, percentile))
        if scale > 0:
            image = image / scale
    return np.clip(image, 0.0, 1.0)


def _ellipticity_from_axis_ratio_angle(axis_ratio: float, angle: float) -> tuple[float, float]:
    q = float(np.clip(axis_ratio, 0.2, 1.0))
    ellipticity = (1.0 - q) / (1.0 + q)
    return float(ellipticity * np.cos(2.0 * angle)), float(ellipticity * np.sin(2.0 * angle))


def make_procedural_galaxy_source(
    rng: np.random.Generator,
    size: int = 192,
    source_extent: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    grid = np.linspace(-source_extent, source_extent, size)
    x, y = np.meshgrid(grid, grid)

    cx = float(rng.uniform(-0.10, 0.10))
    cy = float(rng.uniform(-0.10, 0.10))
    angle = float(rng.uniform(0.0, np.pi))
    q = float(rng.uniform(0.45, 0.9))
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    dx = x - cx
    dy = y - cy
    major = dx * cos_a + dy * sin_a
    minor = -dx * sin_a + dy * cos_a
    radius = np.sqrt(major**2 + (minor / q) ** 2)
    theta = np.arctan2(minor / q, major)

    disk_scale = float(rng.uniform(0.13, 0.23))
    bulge_scale = float(rng.uniform(0.025, 0.055))
    disk = np.exp(-radius / disk_scale)
    bulge = np.exp(-0.5 * (radius / bulge_scale) ** 2)

    arm_count = int(rng.integers(2, 4))
    winding = float(rng.uniform(2.0, 4.5))
    arm_width = float(rng.uniform(0.16, 0.28))
    phase = float(rng.uniform(-np.pi, np.pi))
    arm_phase = np.angle(np.exp(1j * (arm_count * (theta + winding * np.log(radius / disk_scale + 0.08)) + phase)))
    arms = np.exp(-0.5 * (arm_phase / arm_width) ** 2) * np.exp(-radius / (disk_scale * 1.25))
    arms *= radius > bulge_scale * 1.8

    clumps = np.zeros_like(radius)
    for _ in range(int(rng.integers(10, 20))):
        arm_theta = float(rng.uniform(-np.pi, np.pi))
        arm_radius = float(rng.uniform(0.08, min(0.42, source_extent * 0.8)))
        px = cx + arm_radius * np.cos(arm_theta) * cos_a - q * arm_radius * np.sin(arm_theta) * sin_a
        py = cy + arm_radius * np.cos(arm_theta) * sin_a + q * arm_radius * np.sin(arm_theta) * cos_a
        sigma = float(rng.uniform(0.010, 0.026))
        amp = float(rng.uniform(0.35, 1.2))
        clumps += amp * np.exp(-0.5 * (((x - px) ** 2 + (y - py) ** 2) / sigma**2))

    dust_angle = angle + float(rng.uniform(-0.3, 0.3))
    dust_minor = -(x - cx) * np.sin(dust_angle) + (y - cy) * np.cos(dust_angle)
    dust = np.exp(-0.5 * (dust_minor / float(rng.uniform(0.018, 0.04))) ** 2) * np.exp(-radius / (disk_scale * 1.2))

    red = 0.78 * disk + 1.25 * bulge + 0.55 * arms + 0.75 * clumps
    green = 0.92 * disk + 1.05 * bulge + 0.82 * arms + 0.95 * clumps
    blue = 1.15 * disk + 0.72 * bulge + 1.25 * arms + 1.45 * clumps
    rgb = np.stack([red, green, blue], axis=-1)
    rgb *= (1.0 - 0.45 * dust[..., None])

    rgb = ndimage.gaussian_filter(rgb, sigma=(0.45, 0.45, 0.0))
    rgb = _normalise(rgb)
    gray = _normalise(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    e1, e2 = _ellipticity_from_axis_ratio_angle(q, angle)
    return gray, rgb, {
        "source_x": cx,
        "source_y": cy,
        "source_axis_ratio": q,
        "source_angle_degrees": float(np.degrees(angle)),
        "source_disk_scale": disk_scale,
        "source_bulge_scale": bulge_scale,
        "source_arm_count": float(arm_count),
        "source_e1": e1,
        "source_e2": e2,
    }


def _sample_source(source: np.ndarray, beta_x: np.ndarray, beta_y: np.ndarray, source_extent: float) -> np.ndarray:
    size = source.shape[0]
    sx = (beta_x + source_extent) / (2.0 * source_extent) * (size - 1)
    sy = (beta_y + source_extent) / (2.0 * source_extent) * (size - 1)
    if source.ndim == 2:
        return ndimage.map_coordinates(source, [sy, sx], order=1, mode="constant", cval=0.0)
    channels = [
        ndimage.map_coordinates(source[..., channel], [sy, sx], order=1, mode="constant", cval=0.0)
        for channel in range(source.shape[-1])
    ]
    return np.stack(channels, axis=-1)


def generate_realistic_lensing_case(
    seed: int,
    num_pix: int = 128,
    delta_pix: float = 0.05,
    source_size: int = 192,
    source_extent: float = 0.6,
    noise_sigma: float = 0.018,
) -> RealisticSyntheticCase:
    rng = np.random.default_rng(seed)
    source, source_rgb, source_meta = make_procedural_galaxy_source(rng, size=source_size, source_extent=source_extent)

    theta_e = float(rng.uniform(0.95, 1.35))
    lens_q = float(rng.uniform(0.55, 0.88))
    lens_angle = float(rng.uniform(0.0, np.pi))
    lens_e1, lens_e2 = _ellipticity_from_axis_ratio_angle(lens_q, lens_angle)
    metadata = {
        "seed": float(seed),
        "theta_E": theta_e,
        "lens_e1": lens_e1,
        "lens_e2": lens_e2,
        "lens_center_x": float(rng.uniform(-0.025, 0.025)),
        "lens_center_y": float(rng.uniform(-0.025, 0.025)),
        "source_extent": source_extent,
        **source_meta,
    }

    coords = (np.arange(num_pix, dtype=float) - (num_pix - 1) / 2.0) * delta_pix
    image_x, image_y = np.meshgrid(coords, coords)
    beta_x, beta_y = approximate_sie_ray_shooting(image_x, image_y, metadata)
    lensed_source = _sample_source(source, beta_x, beta_y, source_extent)

    lens_radius = np.hypot(image_x - metadata["lens_center_x"], image_y - metadata["lens_center_y"])
    lens_light = np.exp(-np.power(np.maximum(lens_radius, 1e-4) / float(rng.uniform(0.20, 0.34)), 0.35))
    lens_light *= float(rng.uniform(0.10, 0.22))
    clean = _normalise(lensed_source + lens_light)
    noisy = _normalise(clean + rng.normal(0.0, noise_sigma, clean.shape))
    lensed_source = _normalise(lensed_source)
    mask = lensed_source > max(float(np.percentile(lensed_source[lensed_source > 0], 45.0)), 0.015)

    return RealisticSyntheticCase(
        image=noisy,
        clean=clean,
        lensed_source=lensed_source,
        mask=mask,
        source=source,
        source_rgb=source_rgb,
        metadata=metadata,
    )
