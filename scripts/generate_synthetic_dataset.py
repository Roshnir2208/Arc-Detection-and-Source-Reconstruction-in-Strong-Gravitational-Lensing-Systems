from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from lensing_pipeline.lenstronomy_sim import generate_lenstronomy_case
from lensing_pipeline.visualization import save_grayscale_png


# Define command-line options for dataset size, image resolution, and output location.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic strong-lensing images with lenstronomy.")
    parser.add_argument("--count", type=int, default=10, help="Number of images to generate.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting numeric id for batch-safe generation.")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "lenstronomy")
    parser.add_argument("--seed", type=int, default=25015168)
    parser.add_argument("--num-pix", type=int, default=96)
    parser.add_argument("--source-size", type=int, default=128)
    parser.add_argument("--delta-pix", type=float, default=0.05)
    parser.add_argument("--noise-sigma", type=float, default=0.025)
    return parser.parse_args()


# Generate the requested lenstronomy dataset and save images, masks, and metadata.
def main() -> None:
    args = parse_args()
    images_png = args.out_dir / "images_png"
    images_npy = args.out_dir / "images_npy"
    clean_npy = args.out_dir / "clean_npy"
    masks_npy = args.out_dir / "masks_npy"
    sources_png = args.out_dir / "sources_png"
    sources_npy = args.out_dir / "sources_npy"
    for folder in (images_png, images_npy, clean_npy, masks_npy, sources_png, sources_npy):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(args.count):
        # Use a deterministic seed per image so results can be reproduced.
        global_index = args.start_index + index
        seed = args.seed + global_index
        case = generate_lenstronomy_case(
            seed=seed,
            num_pix=args.num_pix,
            delta_pix=args.delta_pix,
            noise_sigma=args.noise_sigma,
            source_size=args.source_size,
        )
        stem = f"lens_{global_index:05d}"
        # Save each simulated case as PNG for viewing and NPY for analysis.
        save_grayscale_png(case.image, images_png / f"{stem}.png")
        np.save(images_npy / f"{stem}.npy", case.image)
        np.save(clean_npy / f"{stem}.npy", case.clean)
        np.save(masks_npy / f"{stem}.npy", case.mask.astype(np.uint8))
        save_grayscale_png(case.source, sources_png / f"{stem}_source.png")
        np.save(sources_npy / f"{stem}.npy", case.source)
        rows.append({"id": stem, **case.metadata})

        if (index + 1) % 50 == 0 or index + 1 == args.count:
            print(f"Generated {index + 1}/{args.count} in batch starting at {args.start_index}")

    metadata_path = args.out_dir / "metadata.csv"
    # Store simulation parameters so each generated lens can be traced later.
    fieldnames = list(rows[0].keys()) if rows else ["id"]
    append_metadata = args.start_index > 0 and metadata_path.exists()
    with metadata_path.open("a" if append_metadata else "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not append_metadata:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Saved dataset to {args.out_dir}")


if __name__ == "__main__":
    main()
