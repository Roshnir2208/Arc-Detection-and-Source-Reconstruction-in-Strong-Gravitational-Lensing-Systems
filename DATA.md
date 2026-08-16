# Data Policy

This repository intentionally does not include the full datasets used during development.

## Included

Small demonstration assets are included in `examples/`:

- one representative synthetic lens image,
- its true arc mask,
- its true source image,
- its final 64 x 64 reconstructed source,
- one real SLACS/HST demonstration image and associated final outputs.

These files are included only to support a quick smoke test and live demonstration.

## Not Included

The following are excluded because they are large, generated, or externally sourced:

- full 1,000-image synthetic Lenstronomy evaluation dataset,
- scaled 10,000 and 50,000 synthetic datasets,
- Galaxy Zoo source dataset,
- SLACS/HST FITS downloads,
- full reconstruction output folders,
- intermediate diagnostic figures,
- local caches and model checkpoints.

## Synthetic Dataset Regeneration

Generate a small synthetic dataset:

```powershell
python scripts/generate_synthetic_dataset.py --count 10 --out-dir data/lenstronomy_demo
```

Generate a larger dataset by increasing `--count`. Large generated datasets should remain outside Git or under an ignored `data/` directory.

## Galaxy Zoo Data

Galaxy Zoo labelled source images were used for the quantitative morphology-classification experiment. The full dataset is not redistributed here. Use the original Galaxy Zoo/SDSS data sources and respect their licence and attribution requirements.

## Real Lens Data

Real lens demonstrations used selected SLACS/HST observations. Full raw FITS downloads are not redistributed here. The dissertation treats these as qualitative demonstrations because source-plane ground truth is unavailable.
