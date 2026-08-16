# Live Demo Script

Target length: 3-5 minutes.

## Opening

Say:

`This demonstration shows the final strong gravitational lensing pipeline. The safest quantitative example is synthetic, because the true source, mask, and lens parameters are known. I will also show a real SLACS/HST example as a qualitative demonstration.`

## Input

Say:

`This is the observed gravitational lens image provided to the pipeline. In the synthetic case, the corresponding source and arc mask are known, so we can calculate objective metrics.`

## Arc Detection

Say:

`The first stage identifies candidate gravitational arc pixels using classical image processing: normalisation, smoothing, LoG filtering, thresholding, morphology, and connected-component filtering.`

## Lens Geometry Estimation

Say:

`The detected mask is fitted with an ellipse. These are geometric estimates: centre, axis ratio, orientation, approximate Einstein radius, and arc span. They are not full physical lens-model optimisation.`

## Source Reconstruction

Say:

`For validated synthetic reconstruction, the pipeline uses the known simulation lens parameters and exact Lenstronomy ray-shooting. The lensed light is mapped back into the source plane, restored using RL15 preprocessing, and interpolated onto a 64 by 64 source grid using linear RBF interpolation.`

## Real-Data Warning

Say:

`For real SLACS/HST images, the output is a qualitative reconstructed source candidate. There is no true source-plane image, so I do not report synthetic-style SSIM or PSNR for real systems.`

## Morphology Classification

Say:

`The morphology classifier was validated separately using labelled Galaxy Zoo sources. A feature-based SVM and an image-based CNN are combined using a confidence-aware ensemble. If they disagree, the system does not force a prediction.`

## Closing

Say:

`The main result is that the pipeline is quantitatively validated on synthetic data and can be run as a qualitative demonstration on real lens observations. The major limitation is that real source reconstruction cannot be quantitatively validated without ground-truth source-plane images.`
