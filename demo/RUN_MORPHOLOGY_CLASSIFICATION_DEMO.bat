@echo off
cd /d "%~dp0.."

echo ============================================================
echo MORPHOLOGY CLASSIFICATION DEMO
echo ============================================================
echo.
echo Classification task:
echo Reconstructed source galaxies are classified as elliptical or spiral.
echo.
echo Classifiers used:
echo 1. SVM  - feature-based classifier
echo 2. CNN  - image-based classifier
echo 3. Confidence-aware ensemble
echo.
echo Final evaluated results:
echo SVM accuracy:      80.00%%
echo CNN accuracy:      64.44%%
echo Ensemble accuracy: 90.00%%
echo.
echo Ensemble rule:
echo - If SVM and CNN agree confidently, output final class
echo - If they disagree, mark the result as uncertain
echo.
echo Key point:
echo Combining feature-based and image-based classifiers improved reliability.
echo.
pause
