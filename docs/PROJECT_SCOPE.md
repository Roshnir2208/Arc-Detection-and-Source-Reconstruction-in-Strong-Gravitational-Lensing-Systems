# Project Scope

Core implemented pipeline:

1. Synthetic gravitational lens generation with Lenstronomy.
2. Classical gravitational arc detection.
3. Ellipse fitting and geometric parameter extraction.
4. Source reconstruction using the inverse lens equation.
5. Quantitative validation on synthetic data.
6. Morphology feature extraction.
7. SVM and CNN morphology classification on labelled Galaxy Zoo data.
8. Confidence-aware ensemble.
9. Qualitative real-data demonstration on selected SLACS/HST images.

Out of scope for the final pipeline:

- Bayesian lens modelling,
- PyAutoLens-style optimisation as the main method,
- GAN/diffusion image generation,
- claiming quantitative real-source validation without ground truth,
- uploading full generated or downloaded datasets.
