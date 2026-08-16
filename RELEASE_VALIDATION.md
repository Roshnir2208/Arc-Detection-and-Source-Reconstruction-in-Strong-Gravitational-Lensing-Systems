# Release Validation

Validation date: 2026-08-15

## Environment

Validated using Python 3.11.4 from the existing project virtual environment.

Required packages checked:

- NumPy
- SciPy
- scikit-learn
- Pillow
- Matplotlib
- Lenstronomy
- PyTorch
- Torchvision

## Commands Tested

From `github_release/`:

```powershell
python demo/check_demo_environment.py
python scripts/run_arc_detection.py
python scripts/run_reconstruction.py
python scripts/run_classification.py
python scripts/run_cnn_ensemble.py
python demo/run_live_demo.py --sample synthetic
python demo/run_live_demo.py --sample real
python scripts/run_full_pipeline.py
```

## Results

| Test | Status | Notes |
| --- | ---: | --- |
| Import/compile check | Passed | All release Python files compiled successfully. |
| Environment check | Passed | Required packages and release assets were found. |
| Arc detection smoke test | Passed | Saved mask, overlay, and metrics for the included synthetic example. |
| Reconstruction smoke test | Passed | Default release mode loads the included final validated reconstruction example. |
| Classification smoke test | Passed | Extracted morphology features and located final classifier summary. |
| SVM/CNN/ensemble summary | Passed | Printed final classification rows from `results_summary/`. |
| Synthetic live demo | Passed | Generated `demo_outputs/synthetic/final_demo_summary.png`. |
| Real live demo | Passed | Generated `demo_outputs/real/final_demo_summary.png`. |
| Full pipeline wrapper | Passed | Calls the safe synthetic live demo. |

## Important Notes

- The full scientific benchmark requires external/full datasets that are not included in GitHub.
- `scripts/run_reconstruction.py` defaults to the included final validated reconstruction asset for a reliable smoke test. Recomputing full results requires the original NPY synthetic dataset.
- Smoke-test outputs are written under `outputs/` and ignored by `.gitignore`.
- Real-data results are qualitative only because true source-plane ground truth is unavailable.
