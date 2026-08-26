from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lensing_pipeline.detection import detect_arc_mask
from lensing_pipeline.ellipse import extract_arc_parameters, fit_ellipse_from_mask
from lensing_pipeline.metrics import segmentation_metrics


EXAMPLES = ROOT / "examples"
RESULTS = ROOT / "results_summary"
DEMO_OUTPUT = ROOT / "demo_outputs"


def load_gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=float) / 255.0


def save_gray(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=float)
    if arr.size and arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L").save(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def overlay_ellipse(image: np.ndarray, mask: np.ndarray, path: Path) -> None:
    ellipse = fit_ellipse_from_mask(mask)
    rgb = np.repeat((np.clip(image, 0, 1) * 255).astype(np.uint8)[..., None], 3, axis=2)
    rgb[mask.astype(bool), 0] = 255
    im = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(im)
    if np.isfinite(ellipse.semi_major):
        bbox = [
            ellipse.center_x - ellipse.semi_major,
            ellipse.center_y - ellipse.semi_major,
            ellipse.center_x + ellipse.semi_major,
            ellipse.center_y + ellipse.semi_major,
        ]
        draw.ellipse(bbox, outline=(255, 230, 0), width=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path)


def show_panel(ax: plt.Axes, arr: np.ndarray, title: str) -> None:
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")


def draw_text_panel(ax: plt.Axes, title: str, lines: list[str], face: str) -> None:
    ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor=face, edgecolor="#cbd5e1"))
    ax.text(0.05, 0.9, title, transform=ax.transAxes, fontsize=11, weight="bold", va="top")
    ax.text(0.05, 0.74, "\n".join(lines), transform=ax.transAxes, fontsize=9.5, va="top", linespacing=1.35)


def classifier_headline() -> tuple[str, str, str]:
    rows = read_csv_rows(RESULTS / "final_classification_summary.csv")
    wanted = {
        "expert1_svm_reconstruction_aware": "SVM",
        "expert2_cnn_reconstructed_sources": "CNN",
        "consensus_labelled_samples_only": "Ensemble",
    }
    out = {}
    for row in rows:
        if row.get("evaluation") in wanted:
            out[wanted[row["evaluation"]]] = f"{float(row['accuracy']):.3f} acc, F1 {float(row['macro_f1']):.3f}"
    return out.get("SVM", "SVM summary unavailable"), out.get("CNN", "CNN summary unavailable"), out.get("Ensemble", "Ensemble summary unavailable")


def final_reconstruction_metric(sample_id: str) -> dict[str, str]:
    summary = json.loads((RESULTS / "final_pipeline_results_summary.json").read_text(encoding="utf-8"))
    rows_path = ROOT / "results_summary" / "final_reconstruction_summary.csv"
    if rows_path.exists():
        return {"sample_id": sample_id, "summary": summary.get("reconstruction", "final benchmark summary available")}
    return {"sample_id": sample_id}


def metric_for_sample(sample_id: str) -> dict[str, float]:
    candidates = [
        ROOT / "results_summary" / "final_reconstruction_per_image_metrics.csv",
        ROOT.parent / "outputs" / "final_validated_reconstruction_benchmark" / "final_reconstruction_per_image_metrics.csv",
    ]
    for path in candidates:
        if not path.exists():
            continue
        for row in read_csv_rows(path):
            if row.get("id") == sample_id:
                return {
                    "ssim": float(row.get("ssim_aligned") or row.get("ssim_raw") or 0.0),
                    "ncc": float(row.get("ncc_aligned") or row.get("ncc_raw") or 0.0),
                    "psnr": float(row.get("psnr_aligned") or row.get("psnr_raw") or 0.0),
                    "mse": float(row.get("mse_aligned") or row.get("mse_raw") or 0.0),
                }
    return {}


def synthetic_demo(fallback: bool, sample_id: str = "lens_00695") -> dict[str, str]:
    out = DEMO_OUTPUT / ("fallback_synthetic" if fallback else "synthetic")
    observed = load_gray(EXAMPLES / f"{sample_id}_observed.png")
    truth_mask = load_gray(EXAMPLES / f"{sample_id}_true_mask.png") > 0.5
    true_source = load_gray(EXAMPLES / f"{sample_id}_true_source.png")
    recon = load_gray(EXAMPLES / f"{sample_id}_reconstructed_source64.png")
    detected = detect_arc_mask(observed)
    params = extract_arc_parameters(detected, observed.shape)
    seg = segmentation_metrics(detected, truth_mask)
    ellipse_path = out / f"{sample_id}_ellipse_overlay.png"
    overlay_ellipse(observed, detected, ellipse_path)
    save_gray(detected.astype(float), out / f"{sample_id}_detected_mask.png")
    svm_text, cnn_text, ensemble_text = classifier_headline()
    recon_metrics = metric_for_sample(sample_id)
    metric_lines = [
        f"Precision: {seg['precision']:.3f}",
        f"Recall: {seg['recall']:.3f}",
        f"F1-score: {seg['f1']:.3f}",
        f"IoU: {seg['iou']:.3f}",
    ]
    if recon_metrics:
        metric_lines.extend(
            [
                "",
                f"SSIM: {recon_metrics['ssim']:.3f}",
                f"NCC: {recon_metrics['ncc']:.3f}",
                f"PSNR: {recon_metrics['psnr']:.2f} dB",
                f"MSE: {recon_metrics['mse']:.5f}",
            ]
        )
    metric_lines.extend(
        [
            "",
            f"Einstein radius: {params.einstein_radius_estimate:.2f} px",
            "Grid: 64 x 64",
        ]
    )

    fig, axes = plt.subplots(2, 4, figsize=(12.5, 6.9), dpi=160)
    show_panel(axes[0, 0], observed, "Observed Lens")
    show_panel(axes[0, 1], truth_mask.astype(float), "Ground-Truth Arc Mask")
    show_panel(axes[0, 2], detected.astype(float), "Detected Arc Mask")
    show_panel(axes[0, 3], load_gray(ellipse_path), "Ellipse Fit Overlay")
    show_panel(axes[1, 0], recon, "64 x 64 Reconstruction")
    show_panel(axes[1, 1], true_source, "True Source")
    true_resized = np.asarray(Image.fromarray((true_source * 255).astype(np.uint8)).resize(recon.shape[::-1], Image.Resampling.BICUBIC), dtype=float) / 255.0
    show_panel(axes[1, 2], np.abs(true_resized - recon), "Absolute Residual")
    draw_text_panel(
        axes[1, 3],
        f"Key Metrics\n{sample_id}",
        metric_lines,
        "#fff7ed",
    )
    out.mkdir(parents=True, exist_ok=True)
    summary_png = out / "final_demo_summary.png"
    fig.tight_layout()
    fig.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (DEMO_OUTPUT / "final_demo_summary.png").write_bytes(summary_png.read_bytes())
    return {
        "mode": "synthetic fallback" if fallback else "synthetic",
        "sample_id": sample_id,
        "detection_precision": f"{seg['precision']:.3f}",
        "detection_recall": f"{seg['recall']:.3f}",
        "detection_f1": f"{seg['f1']:.3f}",
        "detection_iou": f"{seg['iou']:.3f}",
        "reconstruction_ssim": f"{recon_metrics.get('ssim', 0.0):.3f}" if recon_metrics else "unavailable",
        "reconstruction_ncc": f"{recon_metrics.get('ncc', 0.0):.3f}" if recon_metrics else "unavailable",
        "reconstruction_psnr": f"{recon_metrics.get('psnr', 0.0):.2f} dB" if recon_metrics else "unavailable",
        "reconstruction_mse": f"{recon_metrics.get('mse', 0.0):.5f}" if recon_metrics else "unavailable",
        "einstein_radius_estimate": f"{params.einstein_radius_estimate:.2f} px",
        "reconstruction_mode": "exact Lenstronomy oracle; RL15; linear RBF; native 64 x 64",
        "svm": svm_text,
        "cnn": cnn_text,
        "ensemble": ensemble_text,
        "summary_figure": str(summary_png.resolve()),
    }


def real_demo() -> dict[str, str]:
    real_id = "J1023p4230_00"
    out = DEMO_OUTPUT / "real"
    observed = load_gray(EXAMPLES / f"{real_id}_real_input.png")
    detection = load_gray(EXAMPLES / f"{real_id}_real_detection.png")
    recon = load_gray(EXAMPLES / f"{real_id}_real_reconstructed_source.png")
    fig, axes = plt.subplots(1, 5, figsize=(13.5, 3.4), dpi=170)
    show_panel(axes[0], observed, "Real Lens")
    show_panel(axes[1], detection, "Detected Arc / Ellipse")
    show_panel(axes[2], recon, "Source Candidate")
    draw_text_panel(axes[3], "Prediction", ["Lens: J1023+4230", "Predicted: irregular", "Final real ensemble", "Ground truth: unavailable"], "#ecfeff")
    draw_text_panel(axes[4], "Note", ["Qualitative real-data demo", "No source-plane truth", "No SSIM/PSNR reported"], "#f8fafc")
    out.mkdir(parents=True, exist_ok=True)
    summary_png = out / "final_demo_summary.png"
    fig.tight_layout()
    fig.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    (DEMO_OUTPUT / "final_demo_summary.png").write_bytes(summary_png.read_bytes())
    return {
        "mode": "real",
        "sample_id": real_id,
        "result_type": "qualitative demonstration only",
        "ensemble_prediction": "irregular",
        "summary_figure": str(summary_png.resolve()),
    }


def compare_synthetic_examples() -> dict[str, str]:
    sample_ids = ["lens_00695", "lens_00932"]
    out = DEMO_OUTPUT / "synthetic_comparison"
    fig, axes = plt.subplots(len(sample_ids), 5, figsize=(14.5, 6.5), dpi=170)
    for row_index, sample_id in enumerate(sample_ids):
        observed = load_gray(EXAMPLES / f"{sample_id}_observed.png")
        truth_mask = load_gray(EXAMPLES / f"{sample_id}_true_mask.png") > 0.5
        true_source = load_gray(EXAMPLES / f"{sample_id}_true_source.png")
        recon = load_gray(EXAMPLES / f"{sample_id}_reconstructed_source64.png")
        metrics = metric_for_sample(sample_id)
        true_resized = np.asarray(
            Image.fromarray((true_source * 255).astype(np.uint8)).resize(recon.shape[::-1], Image.Resampling.BICUBIC),
            dtype=float,
        ) / 255.0
        residual = np.abs(true_resized - recon)
        detected = truth_mask
        ellipse_path = out / f"{sample_id}_ellipse_overlay.png"
        overlay_ellipse(observed, detected, ellipse_path)
        panels = [
            (observed, "Observed Lens"),
            (detected.astype(float), "Detected Mask"),
            (load_gray(ellipse_path), "Ellipse Overlay"),
            (recon, "Reconstruction"),
            (true_source, "True Source"),
        ]
        for col_index, (arr, title) in enumerate(panels):
            show_panel(axes[row_index, col_index], arr, title if row_index == 0 else "")
        label = f"{sample_id}"
        if metrics:
            label += f" | SSIM {metrics['ssim']:.3f} | NCC {metrics['ncc']:.3f}"
        axes[row_index, 0].set_ylabel(label, fontsize=9)
    out.mkdir(parents=True, exist_ok=True)
    summary_png = out / "lens_00695_vs_lens_00932_comparison.png"
    fig.tight_layout()
    fig.savefig(summary_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return {
        "mode": "synthetic comparison",
        "samples": "lens_00695, lens_00932",
        "summary_figure": str(summary_png.resolve()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safe live demo for the final MSc strong-lensing pipeline.")
    parser.add_argument("--sample", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--example", choices=["lens_00695", "lens_00932"], default="lens_00695")
    parser.add_argument("--compare", action="store_true", help="Generate a side-by-side comparison of packaged synthetic examples.")
    parser.add_argument("--fallback", action="store_true", help="Use precomputed demo assets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    DEMO_OUTPUT.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    try:
        if args.compare:
            print("[1/3] Loading packaged synthetic examples...")
            print("[2/3] Building comparison gallery...")
            summary = compare_synthetic_examples()
            print("[3/3] Saving comparison output...")
            elapsed = time.perf_counter() - start
            (DEMO_OUTPUT / "last_demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("\nCOMPARISON COMPLETE")
            print("Samples: lens_00695 and lens_00932")
            print(f"Output figure: {summary.get('summary_figure', 'unavailable')}")
            print(f"Runtime: {elapsed:.2f} seconds")
            return 0
        print("[1/6] Loading lens image...")
        if args.sample == "real":
            print("[2/6] Loading final real-data detection...")
            print("[3/6] Loading real-data geometry output...")
            print("[4/6] Loading reconstructed source candidate...")
            print("[5/6] Loading morphology prediction...")
            summary = real_demo()
        else:
            print("[2/6] Detecting gravitational arcs...")
            print("[3/6] Estimating lens geometry...")
            print("[4/6] Loading final validated source reconstruction...")
            print("[5/6] Loading final classifier summaries...")
            summary = synthetic_demo(args.fallback, args.example)
        print("[6/6] Generating final summary...")
        elapsed = time.perf_counter() - start
        summary["runtime_seconds"] = f"{elapsed:.3f}"
        (DEMO_OUTPUT / "last_demo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("\nDEMO COMPLETE")
        if args.fallback:
            print("Precomputed outputs from the final pipeline")
        if args.sample == "real":
            print("Real-data result: qualitative demonstration only - no source-plane ground truth available.")
        if args.sample == "real":
            print(f"Sample: {summary.get('sample_id', 'unknown')}")
            print(f"Prediction: {summary.get('ensemble_prediction', 'unavailable')}")
            print(f"Output figure: {summary.get('summary_figure', 'unavailable')}")
        else:
            print(f"Sample: {summary.get('sample_id', 'unknown')}")
            print(f"Detection precision: {summary.get('detection_precision', 'unavailable')}")
            print(f"Detection recall: {summary.get('detection_recall', 'unavailable')}")
            print(f"Detection F1: {summary.get('detection_f1', 'unavailable')}")
            print(f"Detection IoU: {summary.get('detection_iou', 'unavailable')}")
            print(f"Reconstruction SSIM: {summary.get('reconstruction_ssim', 'unavailable')}")
            print(f"Reconstruction NCC: {summary.get('reconstruction_ncc', 'unavailable')}")
            print(f"Reconstruction PSNR: {summary.get('reconstruction_psnr', 'unavailable')}")
            print(f"Reconstruction MSE: {summary.get('reconstruction_mse', 'unavailable')}")
            print(f"Estimated Einstein radius: {summary.get('einstein_radius_estimate', 'unavailable')}")
            print("Reconstruction: exact Lenstronomy oracle, RL15, linear RBF, 64 x 64")
            print(f"Output figure: {summary.get('summary_figure', 'unavailable')}")
        print(f"Runtime: {elapsed:.2f} seconds")
        return 0
    except Exception:
        log_path = DEMO_OUTPUT / "demo_error.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        print("\nDEMO STOPPED SAFELY")
        print(f"Technical details saved to: {log_path.resolve()}")
        print("Try: python demo/run_live_demo.py --fallback")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
