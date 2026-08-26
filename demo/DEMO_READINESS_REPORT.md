# Live Demo Readiness Report

## Demo Package

Demo folder:

`D:\OneDrive Data\Masters Study\Summer\MSc_Lensing_Live_Demo`

Main entry point:

`demo/run_live_demo.py`

Safest launcher:

`demo/START_DEMO.bat`

One-time local environment setup:

`demo/SETUP_LOCAL_DEMO_ENV.bat`

Classification launcher:

`demo/START_CLASSIFICATION_DEMO.bat`

## Verified Demo Modes

### Synthetic Demo

Status: PASS

Command:

```powershell
& "D:\OneDrive Data\Masters Study\Summer\Masters Project\Codex\Project\outputs\lensing_pipeline_starter\.venv\Scripts\python.exe" "demo\run_live_demo.py" --sample synthetic --fallback
```

Sample ID: `lens_00695`

Purpose: Main presentation demo. Shows the complete synthetic validation path from lens image to detected arc, fitted ellipse, reconstructed source, true source comparison, residual, and final classifier summary.

Selection reason: `lens_00695` is the best-ranked reconstruction example in the final validated benchmark by aligned SSIM, with aligned SSIM approximately 0.973 and aligned NCC approximately 0.983.

Runtime observed during preparation: approximately 3 seconds.

Output:

`demo_outputs/fallback_synthetic/final_demo_summary.png`

### Real-Data Demo

Status: PASS

Command:

```powershell
& "D:\OneDrive Data\Masters Study\Summer\Masters Project\Codex\Project\outputs\lensing_pipeline_starter\.venv\Scripts\python.exe" "demo\run_live_demo.py" --sample real
```

Sample ID: `J1023p4230_00`

Purpose: Optional qualitative demonstration on a real SLACS/HST image. This should not be used as the main result because no true source-plane ground truth is available.

Runtime observed during preparation: approximately 1 second.

Output:

`demo_outputs/real/final_demo_summary.png`

### Classification Demo

Status: PASS

Commands:

```powershell
& "D:\OneDrive Data\Masters Study\Summer\Masters Project\Codex\Project\outputs\lensing_pipeline_starter\.venv\Scripts\python.exe" "scripts\run_classification.py"
& "D:\OneDrive Data\Masters Study\Summer\Masters Project\Codex\Project\outputs\lensing_pipeline_starter\.venv\Scripts\python.exe" "scripts\run_cnn_ensemble.py"
```

Purpose: Demonstrates the morphology branch by extracting morphology features from the reconstructed demo source and printing the final SVM, CNN, and confidence-aware ensemble results.

Outputs:

`outputs/classification_demo/morphology_features.csv`

`results_summary/final_classification_summary.csv`

`results_summary/morphology_classification_results.png`

## Final Presentation Sequence

1. Open PowerPoint and explain the pipeline architecture.
2. Double-click `demo/START_DEMO.bat`, or run the synthetic fallback command manually.
3. Open `demo_outputs/fallback_synthetic/final_demo_summary.png`.
4. Run the classification demo commands if asked to show the classifier branch.
5. Only show the real-data demo if asked, and describe it as qualitative.

## Remaining Risks

- Do not run the full 50,000 image generation during the presentation.
- Do not train SVM/CNN models live.
- Do not run full reconstruction benchmarks live.
- Do not rely on plain `python`; use the known working virtual environment path.
- For the final presentation, use the local `.venv` copied inside the demo folder so the command is independent of the original Codex working path.
- Real-data reconstruction should be described as qualitative only.

## Safest Live Command

```powershell
cd "D:\OneDrive Data\Masters Study\Summer\MSc_Lensing_Live_Demo"
.\.venv\Scripts\python.exe "demo\run_live_demo.py" --sample synthetic --fallback
Invoke-Item "demo_outputs\fallback_synthetic\final_demo_summary.png"
```

## Plain-English Instruction

For the presentation, do this:

1. Go to `D:\OneDrive Data\Masters Study\Summer\MSc_Lensing_Live_Demo`.
2. Before the presentation, run `demo/SETUP_LOCAL_DEMO_ENV.bat` once.
3. During the presentation, run `demo/START_DEMO.bat`.
4. Show `demo_outputs/fallback_synthetic/final_demo_summary.png`.
5. Run `demo/START_CLASSIFICATION_DEMO.bat` if the panel wants to see the morphology branch.
