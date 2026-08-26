from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lensing_pipeline.robust_reconstruction import (  # noqa: E402
    RobustReconstructionConfig,
    SourceGrid,
    bilinear_weighted_accumulate,
    image_pixels_to_angles,
    reconstruction_quality_metrics,
    robust_source_reconstruction,
    select_valid_pixels,
    aggregate_beta_samples,
    scattered_interpolate_source,
    tikhonov_regularize_populated_source,
)
from lensing_pipeline.metrics import ssim_simple  # noqa: E402
from lensing_pipeline.reconstruction import approximate_sie_ray_shooting  # noqa: E402


class RobustReconstructionTests(unittest.TestCase):
    def test_center_pixel_converts_to_zero_angle_for_odd_image(self) -> None:
        theta_x, theta_y = image_pixels_to_angles(np.array([2]), np.array([2]), (5, 5), 0.1)
        self.assertAlmostEqual(float(theta_x[0]), 0.0)
        self.assertAlmostEqual(float(theta_y[0]), 0.0)

    def test_bilinear_weights_conserve_flux_for_single_sample(self) -> None:
        grid = SourceGrid(center_x=0.0, center_y=0.0, extent=1.0, output_size=5)
        source, coverage, valid = bilinear_weighted_accumulate(
            np.array([0.0]),
            np.array([0.0]),
            np.array([2.5]),
            grid,
        )
        self.assertTrue(bool(valid[0]))
        self.assertAlmostEqual(float(coverage.sum()), 1.0, places=6)
        self.assertAlmostEqual(float((source * coverage).sum()), 2.5, places=6)

    def test_mask_restriction_keeps_only_requested_pixels(self) -> None:
        image = np.array([[0.0, 0.2], [0.8, 0.1]])
        mask = np.array([[True, False], [True, True]])
        valid = select_valid_pixels(image, mask, mode="arc_and_flux", threshold=0.15)
        self.assertEqual(int(valid.sum()), 1)
        self.assertTrue(bool(valid[1, 0]))

    def test_zero_einstein_radius_keeps_compact_source_finite(self) -> None:
        image = np.zeros((7, 7), dtype=float)
        image[3, 3] = 1.0
        mask = image > 0
        result = robust_source_reconstruction(
            image,
            mask,
            lens_metadata={"theta_E": 0.0, "lens_e1": 0.0, "lens_e2": 0.0, "lens_center_x": 0.0, "lens_center_y": 0.0},
            delta_pix=0.1,
            valid_pixel_mode="arc",
            config=RobustReconstructionConfig(reconstruction_method="weighted"),
        )
        self.assertTrue(np.all(np.isfinite(result.source)))
        self.assertGreater(float(result.source.max()), 0.0)
        self.assertEqual(int(result.stats["valid_image_pixels"]), 1)

    def test_no_lensing_identity_on_same_grid(self) -> None:
        image = np.zeros((9, 9), dtype=float)
        image[3:6, 4:6] = np.array([[0.2, 0.4], [0.7, 1.0], [0.3, 0.5]])
        mask = image > 0
        result = robust_source_reconstruction(
            image,
            mask,
            lens_metadata={"theta_E": 0.0, "lens_e1": 0.0, "lens_e2": 0.0, "lens_center_x": 0.0, "lens_center_y": 0.0},
            delta_pix=1.0,
            valid_pixel_mode="arc",
            config=RobustReconstructionConfig(output_size=9, source_extent=4.0, auto_extent=False, hole_fill="none", output_normalization="none"),
        )
        expected = image / float(image.max())
        np.testing.assert_allclose(result.source, expected, atol=1e-8)

    def test_simple_centered_sis_reconstruction_is_compact(self) -> None:
        image = np.zeros((31, 31), dtype=float)
        yy, xx = np.mgrid[:31, :31]
        radius = np.hypot(xx - 15, yy - 15)
        ring = np.abs(radius - 8.0) < 0.7
        image[ring] = 1.0
        result = robust_source_reconstruction(
            image,
            ring,
            lens_metadata={"theta_E": 8.0, "lens_e1": 0.0, "lens_e2": 0.0, "lens_center_x": 0.0, "lens_center_y": 0.0},
            delta_pix=1.0,
            valid_pixel_mode="arc",
            config=RobustReconstructionConfig(output_size=31, source_extent=3.0, auto_extent=True, hole_fill="nearest_local"),
        )
        stats = reconstruction_quality_metrics(np.eye(31), result.source)
        self.assertGreater(float(result.source.max()), 0.0)
        self.assertLess(float(result.stats["source_grid_extent"]), 3.5)
        self.assertTrue(np.isfinite(stats["ncc_aligned"]))

    def test_perfect_noiseless_sis_ring_recovers_source_above_095_ssim(self) -> None:
        size = 101
        extent = 0.55
        delta_pix = 0.05
        yy, xx = np.mgrid[:size, :size]
        beta_x_grid = (xx / (size - 1)) * 2.0 * extent - extent
        beta_y_grid = (yy / (size - 1)) * 2.0 * extent - extent
        source = np.exp(-0.5 * (((beta_x_grid - 0.08) / 0.055) ** 2 + ((beta_y_grid + 0.03) / 0.035) ** 2))
        source += 0.35 * np.exp(-0.5 * (((beta_x_grid + 0.02) / 0.09) ** 2 + ((beta_y_grid - 0.05) / 0.055) ** 2))
        source /= float(source.max())

        coords = (np.arange(size, dtype=float) - (size - 1) / 2.0) * delta_pix
        theta_x, theta_y = np.meshgrid(coords, coords)
        lens_metadata = {"theta_E": 1.05, "lens_e1": 0.0, "lens_e2": 0.0, "lens_center_x": 0.0, "lens_center_y": 0.0}
        beta_x, beta_y = approximate_sie_ray_shooting(theta_x, theta_y, lens_metadata)
        sample_x = (beta_x + extent) / (2.0 * extent) * (size - 1)
        sample_y = (beta_y + extent) / (2.0 * extent) * (size - 1)
        image = np.clip(ndimage.map_coordinates(source, [sample_y, sample_x], order=1, mode="constant", cval=0.0), 0.0, 1.0)
        mask = image > max(float(np.percentile(image[image > 0], 35.0)), 0.01)

        result = robust_source_reconstruction(
            image,
            mask,
            lens_metadata=lens_metadata,
            delta_pix=delta_pix,
            valid_pixel_mode="arc",
            config=RobustReconstructionConfig(output_size=size, source_extent=extent, auto_extent=False, hole_fill="none", output_normalization="none"),
        )

        self.assertGreater(ssim_simple(source, result.source), 0.95)

    def test_tikhonov_regularization_does_not_populate_empty_pixels(self) -> None:
        source = np.zeros((7, 7), dtype=float)
        source[2:5, 2:5] = np.array([[0.2, 0.6, 0.3], [0.8, 1.0, 0.4], [0.1, 0.5, 0.2]])
        coverage = np.zeros_like(source)
        coverage[2:5, 2:5] = 1.0

        regularized = tikhonov_regularize_populated_source(source, coverage, regularization_lambda=0.05)

        self.assertEqual(int(np.count_nonzero(regularized[coverage == 0])), 0)
        self.assertGreater(float(regularized[coverage > 0].max()), 0.0)

    def test_tikhonov_regularization_preserves_constant_populated_region(self) -> None:
        source = np.zeros((6, 6), dtype=float)
        coverage = np.zeros_like(source)
        source[1:5, 1:5] = 0.7
        coverage[1:5, 1:5] = 2.0

        regularized = tikhonov_regularize_populated_source(source, coverage, regularization_lambda=0.2)

        np.testing.assert_allclose(regularized[coverage > 0], 0.7, atol=1e-8)
        self.assertEqual(float(regularized[coverage == 0].sum()), 0.0)

    def test_scattered_interpolation_recovers_simple_plane_inside_support(self) -> None:
        grid = SourceGrid(center_x=0.0, center_y=0.0, extent=1.0, output_size=11)
        beta_x = np.array([-0.5, 0.5, -0.5, 0.5, 0.0])
        beta_y = np.array([-0.5, -0.5, 0.5, 0.5, 0.0])
        values = beta_x + 2.0 * beta_y + 1.0

        source, stats = scattered_interpolate_source(beta_x, beta_y, values, grid, method="griddata_linear", max_grid_distance=10.0)

        self.assertAlmostEqual(float(source[5, 5]), 1.0, places=6)
        self.assertEqual(float(source[0, 0]), 0.0)
        self.assertEqual(stats["scattered_method_used"], "griddata_linear")

    def test_beta_sample_aggregation_reduces_duplicate_samples(self) -> None:
        grid = SourceGrid(center_x=0.0, center_y=0.0, extent=1.0, output_size=9)
        beta_x = np.array([0.0, 0.01, 0.02, 0.5])
        beta_y = np.array([0.0, 0.01, 0.02, 0.5])
        values = np.array([1.0, 1.05, 20.0, 0.4])

        agg_x, agg_y, agg_values, stats = aggregate_beta_samples(beta_x, beta_y, values, grid, sigma_clip=1.5, min_bin_samples=2)

        self.assertLess(len(agg_values), len(values))
        self.assertGreaterEqual(float(stats["rejected_outlier_count"]), 1.0)
        self.assertTrue(np.all(np.isfinite(agg_x)))
        self.assertTrue(np.all(np.isfinite(agg_y)))

    def test_reconstruction_method_records_scattered_backend(self) -> None:
        image = np.zeros((9, 9), dtype=float)
        image[3:6, 3:6] = 1.0
        mask = image > 0
        result = robust_source_reconstruction(
            image,
            mask,
            lens_metadata={"theta_E": 0.0, "lens_e1": 0.0, "lens_e2": 0.0, "lens_center_x": 0.0, "lens_center_y": 0.0},
            delta_pix=1.0,
            valid_pixel_mode="arc",
            config=RobustReconstructionConfig(
                output_size=9,
                source_extent=4.0,
                auto_extent=False,
                reconstruction_method="griddata_linear",
                output_normalization="none",
            ),
        )

        self.assertEqual(result.stats["reconstruction_method"], "griddata_linear")
        self.assertGreater(float(result.source.max()), 0.0)

    def test_each_rbf_kernel_is_passed_to_distinct_interpolator(self) -> None:
        grid = SourceGrid(center_x=0.0, center_y=0.0, extent=1.0, output_size=7)
        beta_x = np.array([-0.7, -0.2, 0.4, 0.7, -0.4, 0.1, 0.55, -0.55, 0.2])
        beta_y = np.array([-0.5, 0.65, -0.3, 0.55, 0.2, -0.75, 0.05, 0.75, 0.35])
        values = np.sin(2.1 * beta_x) + np.cos(1.7 * beta_y) + beta_x * beta_y
        kernels = ["linear", "thin_plate_spline", "cubic", "multiquadric", "inverse_multiquadric", "gaussian"]
        called_kernels: list[str] = []

        class FakeRBFInterpolator:
            def __init__(self, points: np.ndarray, vals: np.ndarray, **kwargs: object) -> None:
                self.kernel = str(kwargs["kernel"])
                called_kernels.append(self.kernel)

            def __call__(self, target_points: np.ndarray) -> np.ndarray:
                kernel_index = kernels.index(self.kernel) + 1
                return np.full(len(target_points), kernel_index / 10.0, dtype=float)

        outputs = []
        with patch("lensing_pipeline.robust_reconstruction.interpolate.RBFInterpolator", FakeRBFInterpolator):
            for kernel in kernels:
                source, stats = scattered_interpolate_source(
                    beta_x,
                    beta_y,
                    values,
                    grid,
                    method=f"rbf_{kernel}_aggregated",
                    max_grid_distance=0.0,
                    rbf_neighbors=12,
                    rbf_epsilon=0.8,
                )
                self.assertEqual(stats["rbf_kernel"], kernel)
                self.assertNotEqual(stats["scattered_method_used"], "griddata_linear_fallback")
                outputs.append(source.copy())

        self.assertEqual(called_kernels, kernels)
        for left, right in zip(outputs, outputs[1:], strict=False):
            self.assertGreater(float(np.max(np.abs(left - right))), 1e-6)


if __name__ == "__main__":
    unittest.main()
