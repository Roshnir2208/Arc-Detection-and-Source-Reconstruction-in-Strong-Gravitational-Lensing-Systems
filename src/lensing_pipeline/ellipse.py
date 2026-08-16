from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Stores the basic ellipse fitted to a detected arc/ring mask.
@dataclass
class EllipseFit:
    center_x: float
    center_y: float
    semi_major: float
    semi_minor: float
    angle_degrees: float
    pixel_count: int


# Stores the report-ready geometric parameters extracted from a detected mask.
@dataclass
class ArcParameters:
    centroid_x: float
    centroid_y: float
    semi_major: float
    semi_minor: float
    orientation_degrees: float
    axis_ratio: float
    area_pixels: int
    bounding_width: int
    bounding_height: int
    arc_length_estimate: float
    mean_radius_from_centre: float
    median_radius_from_centre: float
    einstein_radius_estimate: float
    radial_thickness_estimate: float
    angular_span_degrees: float


# Fit an approximate ellipse to the detected arc pixels using their covariance.
def fit_ellipse_from_mask(mask: np.ndarray) -> EllipseFit:
    yy, xx = np.nonzero(mask)
    if len(xx) < 5:
        return EllipseFit(np.nan, np.nan, np.nan, np.nan, np.nan, int(len(xx)))

    coords = np.column_stack([xx.astype(float), yy.astype(float)])
    center = coords.mean(axis=0)
    centred = coords - center
    cov = np.cov(centred, rowvar=False)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

    semi_major = 2.0 * float(np.sqrt(max(values[0], 0.0)))
    semi_minor = 2.0 * float(np.sqrt(max(values[1], 0.0)))
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))

    return EllipseFit(
        center_x=float(center[0]),
        center_y=float(center[1]),
        semi_major=semi_major,
        semi_minor=semi_minor,
        angle_degrees=angle,
        pixel_count=int(len(xx)),
    )


# Estimate the angular coverage of the detected arc around the lens centre.
def _angular_span_degrees(angles: np.ndarray) -> float:
    if len(angles) == 0:
        return np.nan
    unit = np.exp(1j * angles)
    mean_angle = np.angle(unit.mean())
    wrapped = np.angle(np.exp(1j * (angles - mean_angle)))
    return float(np.degrees(wrapped.max() - wrapped.min()))


# Convert a detected mask into quantitative arc/ring morphology parameters.
def extract_arc_parameters(mask: np.ndarray, image_shape: tuple[int, int] | None = None) -> ArcParameters:
    mask = np.asarray(mask, dtype=bool)
    yy, xx = np.nonzero(mask)
    ellipse = fit_ellipse_from_mask(mask)

    if len(xx) == 0:
        return ArcParameters(
            centroid_x=np.nan,
            centroid_y=np.nan,
            semi_major=np.nan,
            semi_minor=np.nan,
            orientation_degrees=np.nan,
            axis_ratio=np.nan,
            area_pixels=0,
            bounding_width=0,
            bounding_height=0,
            arc_length_estimate=np.nan,
            mean_radius_from_centre=np.nan,
            median_radius_from_centre=np.nan,
            einstein_radius_estimate=np.nan,
            radial_thickness_estimate=np.nan,
            angular_span_degrees=np.nan,
        )

    height, width = image_shape if image_shape is not None else mask.shape
    lens_cx = (width - 1) / 2.0
    lens_cy = (height - 1) / 2.0
    dx = xx.astype(float) - lens_cx
    dy = yy.astype(float) - lens_cy
    radii = np.hypot(dx, dy)
    angles = np.arctan2(dy, dx)

    x_min = int(xx.min())
    x_max = int(xx.max())
    y_min = int(yy.min())
    y_max = int(yy.max())
    angular_span = _angular_span_degrees(angles)
    median_radius = float(np.median(radii))

    return ArcParameters(
        centroid_x=ellipse.center_x,
        centroid_y=ellipse.center_y,
        semi_major=ellipse.semi_major,
        semi_minor=ellipse.semi_minor,
        orientation_degrees=ellipse.angle_degrees,
        axis_ratio=float(ellipse.semi_minor / ellipse.semi_major) if ellipse.semi_major > 0 else np.nan,
        area_pixels=int(len(xx)),
        bounding_width=int(x_max - x_min + 1),
        bounding_height=int(y_max - y_min + 1),
        arc_length_estimate=float(median_radius * np.radians(angular_span)) if np.isfinite(angular_span) else np.nan,
        mean_radius_from_centre=float(np.mean(radii)),
        median_radius_from_centre=median_radius,
        einstein_radius_estimate=median_radius,
        radial_thickness_estimate=float(np.percentile(radii, 90) - np.percentile(radii, 10)),
        angular_span_degrees=angular_span,
    )
