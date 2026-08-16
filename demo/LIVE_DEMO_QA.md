# Likely Examiner Questions

## Why use synthetic data?

Synthetic data provides known ground truth: the true source image, lens parameters, clean lensed image, and true arc mask. This makes quantitative validation possible.

## Why also use real data?

Real SLACS/HST images show whether the pipeline can process observational data. These results are qualitative because true source-plane images are not available.

## Why 64 x 64 instead of 128 x 128?

The 64 x 64 grid better matches the available inverse-mapped source-plane sampling density. The 128 x 128 grid oversampled sparse beta-plane samples and produced stronger interpolation artefacts.

## Why classical computer vision rather than deep learning for detection?

The project scope was an interpretable MSc pipeline aligned with the interim report. Classical detection made the stages transparent and allowed pixel-level validation using precision, recall, F1-score, IoU, and accuracy.

## What is oracle reconstruction?

Oracle reconstruction uses the known synthetic simulation lens parameters and exact Lenstronomy ray-shooting. It validates the reconstruction machinery under the best available conditions.

## Why is automatic reconstruction less accurate?

Automatic reconstruction estimates geometry from the detected arc mask. These ellipse-derived values are geometric approximations, not exact physical SIE lens parameters.

## Why use exact Lenstronomy ray-shooting for oracle validation?

The synthetic dataset was generated using Lenstronomy. Using the same ray-shooting implementation avoids errors from a custom approximate deflection model and validates the inverse mapping consistently.

## Why use RL15?

Richardson-Lucy deconvolution with 15 iterations was the selected restoration setting from the final reconstruction experiments. It improved reconstruction fidelity without adding aggressive denoising.

## Why linear RBF?

Linear RBF interpolation was selected after comparing interpolation variants. It reduced visible artefacts relative to earlier thin-plate spline behaviour while preserving reconstruction metrics.

## Why use both SVM and CNN?

The SVM uses explicit morphology features, while the CNN uses image appearance. Combining them provides two independent views of morphology.

## What happens when SVM and CNN disagree?

The confidence-aware ensemble outputs `uncertain` rather than forcing a class. Strong agreement is only reported when both classifiers agree with sufficient confidence.

## Can real reconstructions be quantitatively validated?

Not with the current data, because real source-plane ground truth is unavailable. Real results are therefore presented as qualitative demonstrations.

## What is the biggest limitation?

The largest limitation is source reconstruction accuracy for real systems, where the true lens model and true source image are not known.

## What would you improve next?

The next priority would be physically richer real-lens modelling, PSF-aware fitting, and validation against published lens models or survey-provided reference reconstructions where available.
