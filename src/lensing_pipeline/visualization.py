from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lensing_pipeline.ellipse import EllipseFit


# Save a normalized image array as a grayscale PNG.
def save_grayscale_png(image: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    Image.fromarray((arr * 255).astype(np.uint8), mode="L").save(path)


# Save an overlay showing prediction, ground truth, and fitted ellipse.
def save_detection_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    true_mask: np.ndarray | None,
    ellipse: EllipseFit | None,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(np.asarray(image, dtype=float), 0.0, 1.0)
    rgb = np.repeat((arr * 255).astype(np.uint8)[..., None], 3, axis=2)

    if true_mask is not None:
        rgb[np.asarray(true_mask, dtype=bool), 1] = 255
    rgb[np.asarray(pred_mask, dtype=bool), 0] = 255

    im = Image.fromarray(rgb, mode="RGB")
    if ellipse is not None and np.isfinite(ellipse.semi_major):
        draw = ImageDraw.Draw(im)
        bbox = [
            ellipse.center_x - ellipse.semi_major,
            ellipse.center_y - ellipse.semi_major,
            ellipse.center_x + ellipse.semi_major,
            ellipse.center_y + ellipse.semi_major,
        ]
        draw.ellipse(bbox, outline=(255, 255, 0), width=2)
    im.save(path)
