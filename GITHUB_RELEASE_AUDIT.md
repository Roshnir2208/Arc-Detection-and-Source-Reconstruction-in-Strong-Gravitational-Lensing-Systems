# GitHub Release Audit

## Release Folder

`github_release/`

## Files Included

Included:

- final reusable source modules under `src/lensing_pipeline/`,
- clean final task entry points under `scripts/`,
- final configuration files under `configs/`,
- safe live-demo wrapper under `demo/`,
- small representative example images under `examples/`,
- compact final result summaries and selected figures under `results_summary/`,
- documentation under `README.md`, `DATA.md`, and `docs/`.

## Files Excluded

Excluded from the release:

- full development `outputs/` tree,
- full 1,000 / 10,000 / 50,000 synthetic datasets,
- Galaxy Zoo datasets,
- raw SLACS/HST FITS downloads,
- large NPY/NPZ arrays,
- temporary diagnostic folders,
- old generated figures,
- local virtual environments,
- Python caches,
- large model checkpoints.

These exclusions are intentional. GitHub is used as a clean research-code release, not as a backup of the full development folder.

## External Datasets Required for Full Reproduction

- Synthetic Lenstronomy datasets can be regenerated using `scripts/generate_synthetic_dataset.py`.
- Galaxy Zoo/SDSS source images are required to fully reproduce morphology-classifier training and validation.
- SLACS/HST data are required to reproduce the real-data qualitative demonstration from raw observations.

## Model Files

No large trained model checkpoints are included.

The release includes final classification summary outputs. Full classifier retraining requires the labelled Galaxy Zoo source dataset and the scripts provided in the repository.

## Repository Size

Clean release size excluding ignored smoke outputs and Python caches: approximately 1.65 MB.

Largest included files:

| File | Approx. Size |
| --- | ---: |
| `results_summary/real_lens_demonstration.png` | 0.545 MB |
| `results_summary/reconstruction_gallery_best_median_worst.png` | 0.365 MB |
| `results_summary/morphology_classification_results.png` | 0.351 MB |
| `src/lensing_pipeline/reconstruction.py` | 0.049 MB |
| `experiments/run_reconstruction_improvement_ablation.py` | 0.043 MB |

No included file is larger than 20 MB.

## Security and Privacy Audit

Searched for:

- local Windows absolute paths,
- email-style private identifiers,
- API key patterns,
- tokens,
- credential keywords,
- private access strings.

Status: no release-blocking matches found after sanitising copied summary paths.

## Reproducibility Status

Passed smoke tests:

- environment check,
- arc detection on included example,
- reconstruction smoke test using final validated example,
- morphology feature extraction,
- classifier summary loading,
- synthetic live demo,
- real-data live demo.

See `RELEASE_VALIDATION.md`.

## Recommended Repository Name

`strong-lensing-analysis-pipeline`

Alternative:

`msc-strong-lensing-pipeline`

## Publishing Commands

From inside `github_release/`:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Final MSc project release"
```

Then create an empty repository on GitHub under:

`https://github.com/Roshnir2208`

For example, if the repository is named `strong-lensing-analysis-pipeline`:

```powershell
git remote add origin https://github.com/Roshnir2208/strong-lensing-analysis-pipeline.git
git push -u origin main
```

Do not run these commands in the development folder. Run them only inside `github_release/`.
