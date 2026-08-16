from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Holds one simulated lens image plus its clean image, mask, and metadata.
@dataclass
class LenstronomyCase:
    image: np.ndarray
    clean: np.ndarray
    mask: np.ndarray
    source: np.ndarray
    metadata: dict[str, float]


# Import lenstronomy lazily so other scripts can still run without it installed.
def _import_lenstronomy():
    try:
        from lenstronomy.Data.imaging_data import ImageData
        from lenstronomy.Data.psf import PSF
        from lenstronomy.ImSim.image_model import ImageModel
        from lenstronomy.LensModel.lens_model import LensModel
        from lenstronomy.LightModel.light_model import LightModel
        from lenstronomy.Util import simulation_util
    except ImportError as exc:
        raise RuntimeError(
            "lenstronomy is not installed. Run `pip install -r requirements.txt` in your project venv."
        ) from exc
    return ImageData, PSF, ImageModel, LensModel, LightModel, simulation_util


# Create a visible-arc ground-truth mask from the brightest lensed source pixels.
def make_arc_mask(clean_lensed_source: np.ndarray, percentile: float = 92.0) -> np.ndarray:
    threshold = float(np.percentile(clean_lensed_source, percentile))
    return clean_lensed_source > max(threshold, 1e-6)


# Convert lenstronomy ellipticity components into an approximate axis ratio and angle.
def ellipticity_to_axis_ratio_angle(e1: float, e2: float) -> tuple[float, float]:
    ellipticity = min(float(np.hypot(e1, e2)), 0.8)
    axis_ratio = max((1.0 - ellipticity) / (1.0 + ellipticity), 0.15)
    angle = 0.5 * float(np.arctan2(e2, e1))
    return axis_ratio, angle


# Render the unlensed Sersic source on a fixed source-plane grid for reconstruction validation.
def render_source_plane(
    kwargs_source: dict[str, float],
    output_size: int = 64,
    source_extent: float = 0.6,
) -> np.ndarray:
    grid = np.linspace(-source_extent, source_extent, output_size)
    sx, sy = np.meshgrid(grid, grid)
    q, phi = ellipticity_to_axis_ratio_angle(kwargs_source["e1"], kwargs_source["e2"])

    dx = sx - kwargs_source["center_x"]
    dy = sy - kwargs_source["center_y"]
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    x_rot = cos_phi * dx + sin_phi * dy
    y_rot = -sin_phi * dx + cos_phi * dy
    elliptical_radius = np.sqrt(x_rot**2 + (y_rot / q) ** 2)

    n_sersic = max(kwargs_source["n_sersic"], 0.3)
    r_sersic = max(kwargs_source["R_sersic"], 1e-3)
    b_n = 2.0 * n_sersic - 1.0 / 3.0
    source = kwargs_source["amp"] * np.exp(-b_n * ((elliptical_radius / r_sersic) ** (1.0 / n_sersic) - 1.0))
    source = np.clip(source, 0.0, None)
    peak = float(source.max())
    return source / peak if peak > 0 else source


# Generate one randomized SIE + Sersic strong-lensing simulation.
def generate_lenstronomy_case(
    seed: int | None = None,
    num_pix: int = 96,
    delta_pix: float = 0.05,
    noise_sigma: float = 0.025,
    source_size: int = 128,
) -> LenstronomyCase:
    """Generate one SIE + Sersic source strong-lensing image with lenstronomy."""
    ImageData, PSF, ImageModel, LensModel, LightModel, simulation_util = _import_lenstronomy()
    rng = np.random.default_rng(seed)

    kwargs_data = simulation_util.data_configure_simple(
        numPix=num_pix,
        deltaPix=delta_pix,
        exposure_time=1000.0,
        background_rms=noise_sigma,
    )
    data_class = ImageData(**kwargs_data)

    kwargs_psf = {
        "psf_type": "GAUSSIAN",
        "fwhm": float(rng.uniform(0.05, 0.10)),
        "pixel_size": delta_pix,
        "truncation": 3,
    }
    psf_class = PSF(**kwargs_psf)

    lens_model = LensModel(lens_model_list=["SIE"])
    source_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    lens_light_model = LightModel(light_model_list=["SERSIC_ELLIPSE"])
    image_model = ImageModel(
        data_class,
        psf_class,
        lens_model_class=lens_model,
        source_model_class=source_model,
        lens_light_model_class=lens_light_model,
    )

    theta_e = float(rng.uniform(0.85, 1.25))
    lens_e1 = float(rng.uniform(-0.18, 0.18))
    lens_e2 = float(rng.uniform(-0.18, 0.18))
    kwargs_lens = [
        {
            "theta_E": theta_e,
            "e1": lens_e1,
            "e2": lens_e2,
            "center_x": float(rng.uniform(-0.04, 0.04)),
            "center_y": float(rng.uniform(-0.04, 0.04)),
        }
    ]

    kwargs_source = [
        {
            "amp": float(rng.uniform(45, 90)),
            "R_sersic": float(rng.uniform(0.05, 0.13)),
            "n_sersic": float(rng.uniform(1.0, 2.2)),
            "e1": float(rng.uniform(-0.25, 0.25)),
            "e2": float(rng.uniform(-0.25, 0.25)),
            "center_x": float(rng.uniform(-0.18, 0.18)),
            "center_y": float(rng.uniform(-0.18, 0.18)),
        }
    ]

    kwargs_lens_light = [
        {
            "amp": float(rng.uniform(8, 22)),
            "R_sersic": float(rng.uniform(0.22, 0.42)),
            "n_sersic": float(rng.uniform(2.0, 4.0)),
            "e1": float(rng.uniform(-0.12, 0.12)),
            "e2": float(rng.uniform(-0.12, 0.12)),
            "center_x": kwargs_lens[0]["center_x"],
            "center_y": kwargs_lens[0]["center_y"],
        }
    ]

    lensed_source = image_model.source_surface_brightness(kwargs_source, kwargs_lens)
    lens_light = image_model.lens_surface_brightness(kwargs_lens_light)
    clean = image_model.image(kwargs_lens, kwargs_source, kwargs_lens_light=kwargs_lens_light)
    source_plane = render_source_plane(kwargs_source[0], output_size=source_size)

    noisy = clean + rng.normal(0.0, noise_sigma, clean.shape)
    noisy = np.clip(noisy, 0.0, None)

    scale = float(np.percentile(noisy, 99.7))
    if scale > 0:
        noisy = np.clip(noisy / scale, 0.0, 1.0)
        clean = np.clip(clean / scale, 0.0, 1.0)
        lensed_source = np.clip(lensed_source / scale, 0.0, 1.0)
        lens_light = np.clip(lens_light / scale, 0.0, 1.0)

    mask = make_arc_mask(lensed_source)
    metadata = {
        "seed": float(seed if seed is not None else -1),
        "theta_E": theta_e,
        "lens_e1": lens_e1,
        "lens_e2": lens_e2,
        "lens_center_x": float(kwargs_lens[0]["center_x"]),
        "lens_center_y": float(kwargs_lens[0]["center_y"]),
        "source_amp": float(kwargs_source[0]["amp"]),
        "source_x": float(kwargs_source[0]["center_x"]),
        "source_y": float(kwargs_source[0]["center_y"]),
        "source_R_sersic": float(kwargs_source[0]["R_sersic"]),
        "source_n_sersic": float(kwargs_source[0]["n_sersic"]),
        "source_e1": float(kwargs_source[0]["e1"]),
        "source_e2": float(kwargs_source[0]["e2"]),
        "noise_sigma": float(noise_sigma),
        "psf_fwhm": float(kwargs_psf["fwhm"]),
        "lens_light_fraction": float(lens_light.sum() / (clean.sum() + 1e-12)),
    }
    return LenstronomyCase(image=noisy, clean=clean, mask=mask, source=source_plane, metadata=metadata)
