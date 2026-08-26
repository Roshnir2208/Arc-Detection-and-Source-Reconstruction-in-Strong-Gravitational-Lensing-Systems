# Arc Detection and Source Reconstruction in Strong Gravitational Lensing Systems

A master's project implementing an automated pipeline for gravitational arc detection, lens geometry estimation, source-plane reconstruction, and galaxy morphology classification.

## Overview

This project investigates an interpretable end-to-end framework for analysing strong gravitational lensing systems. The pipeline combines classical computer vision, physics-guided source reconstruction, and machine learning within a modular workflow.

Synthetic gravitational lens systems generated using Lenstronomy are used for quantitative evaluation because the true source images, lens parameters, and arc masks are known. Selected real SLACS/HST lens images are also processed as qualitative demonstrations of the pipeline on observational data, where equivalent source-plane ground truth is unavailable.

## Pipeline

```text
Synthetic or real lens image
        |
        v
Gravitational arc detection
        |
        v
Lens geometry estimation
        |
        v
Source reconstruction
        |
        +--------------------------+
        |                          |
        v                          v
Forward validation          Morphology analysis
(synthetic)                       |
                                  v
                             SVM + CNN
                                  |
                                  v
                       Confidence-aware ensemble
                                  |
                                  v
                        Evaluation and reporting
```

The main stages are:

1. **Arc Detection** — identifies gravitational arcs and Einstein-ring structures using classical computer vision.
2. **Lens Geometry Estimation** — extracts approximate geometric information from the detected arc mask.
3. **Source Reconstruction** — maps observed image-plane information back into the source plane.
4. **Forward Validation** — provides an additional consistency check for synthetic reconstruction.
5. **Morphology Classification** — analyses reconstructed sources using SVM and CNN classifiers together with a confidence-aware ensemble.

---

## Synthetic Dataset

Synthetic strong gravitational lens systems are generated using Lenstronomy to provide controlled data for quantitative evaluation.

Each simulated sample provides:

- a gravitationally lensed image,
- the corresponding unlensed source image,
- a ground-truth arc mask, and
- associated lens and source parameters.

The foreground lens is represented using a Singular Isothermal Ellipsoid (SIE) mass model, while the background galaxy is represented using a Sérsic light profile. PSF effects and observational noise are incorporated to approximate astronomical observations.

The availability of known source images, masks, and simulation parameters enables quantitative evaluation of the pipeline.

---

## Arc Detection

Gravitational arcs are detected using a classical computer vision pipeline.

```text
Input image
-> Normalisation
-> Gaussian smoothing
-> Laplacian of Gaussian filtering
-> Thresholding
-> Morphological cleanup
-> Connected-component filtering
-> Binary arc mask
```

This approach provides an interpretable detection stage without requiring a large labelled training dataset.

### Detection Performance

Final performance on the 1,000-image synthetic dataset:

| Metric | Value |
|---|---:|
| Precision | 0.9455 |
| Recall | 0.9176 |
| F1-score | 0.9313 |
| IoU | 0.8715 |
| Accuracy | 0.9854 |

---

## Lens Geometry Estimation

Geometric information is estimated directly from the detected binary arc mask.

The implementation uses the spatial distribution of detected arc pixels to compute a covariance-based ellipse approximation. Eigenvalue and eigenvector decomposition of the pixel-coordinate covariance matrix provides estimates of the principal axes and orientation.

Extracted geometric descriptors include:

- centroid,
- semi-major axis,
- semi-minor axis,
- axis ratio,
- orientation,
- approximate Einstein radius,
- radial thickness,
- angular span, and
- mask area.

These quantities provide an approximate geometric description of the detected lens structure. They are not a complete physical SIE lens-model optimisation.

---

## Source Reconstruction

Source reconstruction estimates the intrinsic source-plane appearance of the background galaxy from the observed gravitationally lensed image.

Two reconstruction settings are considered: **oracle reconstruction** and **automatic reconstruction**.

### Oracle Reconstruction

Oracle reconstruction uses the known lens parameters available from the synthetic simulations. This provides a controlled benchmark for evaluating reconstruction quality independently of errors introduced by automatic lens-geometry estimation.

The final oracle reconstruction uses:

- exact Lenstronomy SIE ray-shooting,
- Richardson-Lucy deconvolution,
- linear Radial Basis Function (RBF) interpolation, and
- a native 64 x 64 source grid.

The 64 x 64 output is a source-plane reconstruction rather than a direct resize of the 128 x 128 lensed input. Image-plane information is mapped into source-plane coordinates and reconstructed on the source grid.

### Automatic Reconstruction

The automatic reconstruction branch uses geometric parameters estimated from the detected arc mask rather than the known synthetic lens parameters.

This represents the automated reconstruction pathway. However, the estimated lens geometry is an approximation and should not be interpreted as a complete physical lens-model fit.

The distinction between the two modes is important:

- **Oracle mode** evaluates reconstruction using known synthetic lens information.
- **Automatic mode** includes uncertainty introduced by arc detection and geometric lens estimation.

### Oracle Reconstruction Performance

The final quantitative oracle reconstruction benchmark achieved:

| Metric | Value |
|---|---:|
| Mean SSIM | 0.796 |
| Mean PSNR | 34.34 dB |

These values represent **oracle reconstruction performance using known synthetic lens parameters** and should not be interpreted as the performance of the fully automatic geometry-estimation branch.

---

## Forward Validation

Forward validation is included as an additional image-plane consistency check for synthetic source reconstruction.

The reconstructed source is projected back into the image plane to produce a predicted lensed image, which can then be compared with the corresponding synthetic lensed observation.

Forward-validation outputs are kept separate from the primary source-plane reconstruction metrics. The primary reconstruction benchmark compares the reconstructed source directly with the known synthetic source, whereas forward validation assesses whether a reconstructed source can reproduce the observed lens structure when projected back into the image plane.

Forward-validation results are included in:

```text
results_summary/forward_validation_summary.csv
```

This validation should not be interpreted as quantitative validation of real SLACS/HST reconstruction, where equivalent source-plane ground truth is unavailable.

---

## Morphology Classification

The reconstructed source images are subsequently analysed for galaxy morphology.

Two complementary classification approaches are implemented:

- **Support Vector Machine (SVM)** — operates on extracted morphological features.
- **Convolutional Neural Network (CNN)** — learns image representations directly from galaxy images.

A confidence-aware ensemble combines the outputs of the two classifiers. Agreement and confidence information are used to distinguish more reliable predictions from uncertain or conflicting cases.

### Classification Performance

Final results on the 90-image Galaxy Zoo evaluation set:

| Model / Measure | Value |
|---|---:|
| SVM accuracy | 0.8000 |
| CNN accuracy | 0.6444 |
| Confidence-aware ensemble accuracy on consensus-labelled samples | 0.9000 |
| Strong-agreement coverage | 0.3667 |
| Weak-agreement coverage | 0.1889 |
| Disagreement fraction | 0.4444 |

The reported **90% ensemble accuracy applies specifically to consensus-labelled samples** and should not be interpreted as 90% accuracy across all evaluation cases without this qualification.

---

## Real SLACS/HST Demonstration

Selected real strong gravitational lens images from SLACS/HST are processed through the pipeline to demonstrate its applicability to observational data.

The real-data experiments are treated as **qualitative demonstrations rather than quantitative validation** because equivalent true unlensed source-plane images and verified morphology labels are not available as direct ground truth.

Example real-data inputs and outputs are included in the `examples/` directory.

---

## Installation

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Quick Demo

Run the synthetic demonstration:

```powershell
python demo/run_live_demo.py --sample synthetic
```

Run the real-data qualitative demonstration:

```powershell
python demo/run_live_demo.py --sample real
```

Run fallback mode using precomputed release assets:

```powershell
python demo/run_live_demo.py --fallback
```

On Windows, the provided demo launchers include:

```text
.\demo\RUN_END_TO_END_PIPELINE_DEMO.bat
.\demo\RUN_RELENSING_FORWARD_VALIDATION_DEMO.bat
.\demo\RUN_RECONSTRUCTION_COMPARISON_DEMO.bat
.\demo\RUN_MORPHOLOGY_CLASSIFICATION_DEMO.bat
.\demo\START_REAL_DEMO.bat
```

---

## Core Commands

Generate a small synthetic dataset:

```powershell
python scripts/generate_synthetic_dataset.py --count 10 --out-dir data/lenstronomy_demo
```

Run arc detection:

```powershell
python scripts/run_arc_detection.py
```

Run a source reconstruction example:

```powershell
python scripts/run_reconstruction.py
```

Extract morphology features:

```powershell
python scripts/run_classification.py
```

Display the SVM, CNN, and confidence-aware ensemble summary:

```powershell
python scripts/run_cnn_ensemble.py
```

Run the integrated pipeline:

```powershell
python scripts/run_full_pipeline.py
```

---

## Repository Structure

```text
Arc-Detection-and-Source-Reconstruction-in-Strong-Gravitational-Lensing-Systems/
├── configs/
│   ├── final_classification.json
│   ├── final_detection.json
│   ├── final_paths.example.json
│   └── final_reconstruction.json
│
├── demo/
│   ├── check_demo_environment.py
│   ├── DEMO_READINESS_REPORT.md
│   ├── run_live_demo.py
│   ├── START_DEMO.bat
│   ├── START_CLASSIFICATION_DEMO.bat
│   ├── START_COMPARE_TWO_LENSES.bat
│   └── START_REAL_DEMO.bat
│
├── examples/
│   ├── synthetic lens examples
│   ├── reconstructed source examples
│   ├── ground-truth masks and sources
│   └── real SLACS/HST demonstration images
│
├── experiments/
│   ├── audit_and_correct_lenstronomy_mapping.py
│   ├── run_cosmos_lensing_benchmark.py
│   ├── run_reconstruction_improvement_ablation.py
│   ├── run_source_grid_resolution_ablation.py
│   ├── run_sparse_coverage_targeted_fixes.py
│   └── run_support_tv_reconstruction_validation.py
│
├── outputs/
│   ├── arc_detection_demo/
│   ├── classification_demo/
│   └── reconstruction_demo/
│
├── results_summary/
│   ├── final_detection_summary.csv
│   ├── final_reconstruction_summary.csv
│   ├── final_reconstruction_per_image_metrics.csv
│   ├── final_classification_summary.csv
│   ├── final_pipeline_results_summary.json
│   ├── forward_validation_summary.csv
│   └── result figures
│
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── run_arc_detection.py
│   ├── run_reconstruction.py
│   ├── run_classification.py
│   ├── run_cnn_ensemble.py
│   └── run_full_pipeline.py
│
├── src/
│   └── lensing_pipeline/
│       ├── detection.py
│       ├── ellipse.py
│       ├── lenstronomy_sim.py
│       ├── metrics.py
│       ├── morphology_ensemble.py
│       ├── morphology_models.py
│       ├── reconstruction.py
│       ├── robust_reconstruction.py
│       ├── synthetic.py
│       └── visualization.py
│
├── .gitignore
├── DATA.md
├── LICENSE
├── README.md
├── requirements.txt
└── requirements-dev.txt
```

Generated caches such as `__pycache__/` and individual output images are omitted from the structure above for clarity.

---

## Data

Only small demonstration images are included in this repository.

The complete experimental datasets and larger output directories are excluded from the release, including:

- full synthetic datasets,
- Galaxy Zoo data,
- raw HST/FITS files, and
- complete experimental output trees.

See `DATA.md` for additional information.

---

## Limitations

- Quantitative source-reconstruction validation is primarily based on synthetic data.
- The reported SSIM and PSNR reconstruction results correspond to oracle reconstruction using known synthetic lens parameters.
- Automatic lens geometry estimation is approximate and does not constitute a full physical lens-model fit.
- Forward validation is separate from the primary source-plane reconstruction benchmark.
- Real SLACS/HST experiments are qualitative demonstrations.
- Equivalent direct source-plane ground truth is unavailable for the real observations used in this project.
- Reconstruction quality can affect downstream morphology classification.
- The demonstration assets included in this repository are smaller than the complete datasets used for the dissertation experiments.

---

## Future Work

Potential extensions include more advanced physical lens modelling, improved automatic lens-parameter estimation, physics-informed or deep learning-based source reconstruction, validation using larger collections of real strong-lens systems, and adaptation of the pipeline for large-scale astronomical surveys such as LSST, Euclid, and the Nancy Grace Roman Space Telescope.

---

## License

See `LICENSE` for the repository licence.
