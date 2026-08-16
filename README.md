# Automated Strong Gravitational Lensing Analysis Pipeline

Minimal release repository for an MSc project on detecting strong gravitational lensing features, estimating lens geometry, reconstructing source-plane galaxies, and validating downstream morphology classification.

## Overview

This project implements an interpretable end-to-end strong-lensing analysis pipeline. Synthetic gravitational lenses generated with Lenstronomy are used for quantitative validation because the true source image, lens parameters, and arc mask are known. Selected real SLACS/HST images are processed as qualitative demonstrations because source-plane ground truth is unavailable.

## Pipeline

```text
Synthetic or real lens image
-> Classical arc detection
-> Lens geometry estimation / ellipse fitting
-> Source reconstruction
-> Morphology feature extraction
-> SVM + CNN classification
-> Confidence-aware ensemble
-> Evaluation and reporting
```

## Final Reconstruction Method

The final validated synthetic reconstruction uses:

- exact Lenstronomy SIE ray-shooting in known-parameter/oracle mode,
- Richardson-Lucy preprocessing with 15 iterations,
- linear radial basis function interpolation,
- native 64 x 64 source grid.

The automatic reconstruction branch uses ellipse-derived geometric parameters as practical approximations. These are not presented as exact physical SIE lens-model fits.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Demo

Run the safest synthetic demonstration:

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

On Windows, double-click:

- `demo/START_DEMO.bat`
- `demo/START_REAL_DEMO.bat`

## Core Commands

Generate a small synthetic dataset:

```powershell
python scripts/generate_synthetic_dataset.py --count 10 --out-dir data/lenstronomy_demo
```

Run arc detection on the included example:

```powershell
python scripts/run_arc_detection.py
```

Run one source reconstruction example:

```powershell
python scripts/run_reconstruction.py
```

Extract morphology features and locate final classifier summaries:

```powershell
python scripts/run_classification.py
```

Show final SVM/CNN/ensemble summary:

```powershell
python scripts/run_cnn_ensemble.py
```

## Headline Results

Final detection summary on the 1,000-image synthetic dataset:

- overall precision: 0.9455
- overall recall: 0.9176
- overall F1-score: 0.9313
- overall IoU: 0.8715
- overall accuracy: 0.9854

Final reconstruction summary on the 1,000-image synthetic benchmark:

- mean SSIM: 0.8250
- median SSIM: 0.8328
- mean NCC: 0.8491
- mean PSNR: 29.9167 dB
- mean MSE: 0.001138

Final morphology classification summary on the 90-image Galaxy Zoo evaluation set:

- SVM accuracy: 0.8000
- CNN accuracy: 0.6444
- confidence-aware ensemble accuracy on consensus-labelled samples: 0.9000
- strong-agreement coverage: 0.3667
- weak-agreement coverage: 0.1889
- disagreement fraction: 0.4444

## Repository Structure

```text
github_release/
  README.md
  DATA.md
  requirements.txt
  configs/
  demo/
  docs/
  examples/
  experiments/
  results_summary/
  scripts/
  src/lensing_pipeline/
```

## Data

The repository includes only small demonstration images. Full synthetic datasets, Galaxy Zoo data, raw HST/FITS files, and full output trees are excluded. See `DATA.md`.

## Limitations

- Quantitative source-reconstruction validation is synthetic only.
- Real SLACS/HST results are qualitative demonstrations.
- Automatic lens geometry is ellipse-derived and approximate.
- Real source-plane ground truth is unavailable.
- The released examples are intentionally small and do not replace the full dissertation experiments.
