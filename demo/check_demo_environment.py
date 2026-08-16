from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_IMPORTS = [
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("sklearn", "scikit-learn"),
    ("PIL", "Pillow"),
    ("matplotlib", "Matplotlib"),
    ("lenstronomy", "Lenstronomy"),
    ("torch", "PyTorch"),
    ("torchvision", "Torchvision"),
]

REQUIRED_PATHS = [
    ROOT / "src" / "lensing_pipeline" / "detection.py",
    ROOT / "src" / "lensing_pipeline" / "robust_reconstruction.py",
    ROOT / "examples" / "lens_00932_observed.png",
    ROOT / "examples" / "lens_00932_true_mask.png",
    ROOT / "examples" / "lens_00932_true_source.png",
    ROOT / "examples" / "lens_00932_reconstructed_source64.png",
    ROOT / "examples" / "J1023p4230_00_real_input.png",
    ROOT / "results_summary" / "final_classification_summary.csv",
    ROOT / "results_summary" / "final_reconstruction_summary.csv",
]


def main() -> int:
    print("Checking release demo environment...\n")
    print(f"Python: {sys.version.split()[0]}")
    ok = True
    for module, label in REQUIRED_IMPORTS:
        try:
            imported = importlib.import_module(module)
            print(f"[OK] {label}: {getattr(imported, '__version__', 'installed')}")
        except Exception as exc:
            ok = False
            print(f"[MISSING] {label}: {exc}")
    print("\nRequired release files:")
    for path in REQUIRED_PATHS:
        if path.exists():
            print(f"[OK] {path.relative_to(ROOT)}")
        else:
            ok = False
            print(f"[MISSING] {path.relative_to(ROOT)}")
    print("\nDEMO ENVIRONMENT READY" if ok else "\nDEMO ENVIRONMENT NOT READY")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
