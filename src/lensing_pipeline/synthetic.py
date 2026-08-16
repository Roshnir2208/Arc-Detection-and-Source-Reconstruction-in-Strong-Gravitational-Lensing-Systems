from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Holds one lightweight toy arc case for quick tests without lenstronomy.
@dataclass
class ToyLensCase:
    image: np.ndarray
    clean: np.ndarray
    mask: np.ndarray
    source: np.ndarray
    metadata: dict[str, float]


# Generate a simple noisy arc image for fast pipeline sanity checks.
def generate_toy_lens_case(size: int = 128, seed: int | None = None) -> ToyLensCase:
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size))
    cx = size / 2 + rng.uniform(-5, 5)
    cy = size / 2 + rng.uniform(-5, 5)
    radius = rng.uniform(size * 0.25, size * 0.36)
    start = rng.uniform(-np.pi, np.pi)
    span = rng.uniform(0.9, 1.8)
    thickness = rng.uniform(2.0, 3.8)

    angle = np.arctan2(yy - cy, xx - cx)
    dist = np.hypot(xx - cx, yy - cy)
    delta = np.angle(np.exp(1j * (angle - start)))
    arc_region = (delta > 0) & (delta < span)
    clean = np.exp(-0.5 * ((dist - radius) / thickness) ** 2) * arc_region
    clean += 0.20 * np.exp(-0.5 * (dist / (size * 0.075)) ** 2)

    noise = rng.normal(0, 0.045, clean.shape)
    image = np.clip(clean + noise, 0.0, 1.0)
    mask = clean > 0.25

    sy, sx = np.indices((48, 48))
    source = np.exp(-(((sx - 24) / 8) ** 2 + ((sy - 24) / 5) ** 2) / 2)

    return ToyLensCase(
        image=image,
        clean=clean,
        mask=mask,
        source=source,
        metadata={"radius": float(radius), "arc_span": float(span), "thickness": float(thickness)},
    )
