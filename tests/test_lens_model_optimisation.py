from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lensing_pipeline.lens_model_optimisation import (  # noqa: E402
    LensOptimisationConfig,
    LensSourceParameters,
    enforce_valid_ellipticity,
    forward_model_image,
    loss_vector,
    multistart_vectors,
    parameter_bounds,
    params_to_vector,
)


class LensModelOptimisationTests(unittest.TestCase):
    def test_enforce_valid_ellipticity_clips_large_values(self) -> None:
        e1, e2 = enforce_valid_ellipticity(2.0, 0.0, max_ellipticity=0.55)

        self.assertLessEqual(np.hypot(e1, e2), 0.55 + 1e-12)

    def test_parameter_bounds_contain_initial_guess(self) -> None:
        config = LensOptimisationConfig()
        initial = LensSourceParameters(1.0, 0.02, -0.03, 0.1, -0.05, 0.0, 0.0, 0.08, 0.7)

        lower, upper = parameter_bounds(initial, config)
        vector = params_to_vector(initial)

        self.assertTrue(np.all(vector >= lower))
        self.assertTrue(np.all(vector <= upper))
        self.assertGreater(lower[0], 0.0)
        self.assertGreater(lower[7], 0.0)
        self.assertGreater(lower[8], 0.0)

    def test_multistart_vectors_are_reproducible(self) -> None:
        config = LensOptimisationConfig(n_starts=4, random_seed=123)
        initial = LensSourceParameters(1.0, 0.02, -0.03, 0.1, -0.05, 0.0, 0.0, 0.08, 0.7)
        lower, upper = parameter_bounds(initial, config)

        starts_a = multistart_vectors(initial, lower, upper, config)
        starts_b = multistart_vectors(initial, lower, upper, config)

        self.assertEqual(len(starts_a), 4)
        for left, right in zip(starts_a, starts_b):
            np.testing.assert_allclose(left, right)

    def test_true_parameters_have_lower_loss_than_perturbed_parameters(self) -> None:
        config = LensOptimisationConfig(delta_pix=0.05, noise_sigma=0.02, fit_stride=1, ray_tracer="approximate")
        true_params = LensSourceParameters(0.65, 0.0, 0.0, 0.06, -0.03, 0.03, -0.02, 0.05, 1.0)
        observed = forward_model_image(true_params, (33, 33), config)
        mask = observed > np.percentile(observed, 70.0)
        perturbed = LensSourceParameters(0.95, 0.12, -0.08, -0.12, 0.09, -0.12, 0.1, 0.09, 0.65)

        true_loss = np.mean(loss_vector(params_to_vector(true_params), observed, mask, true_params, config) ** 2)
        perturbed_loss = np.mean(loss_vector(params_to_vector(perturbed), observed, mask, true_params, config) ** 2)

        self.assertLess(true_loss, perturbed_loss)


if __name__ == "__main__":
    unittest.main()
