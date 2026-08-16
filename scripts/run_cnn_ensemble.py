from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "results_summary" / "final_classification_summary.csv"
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = {
        "expert1_svm_reconstruction_aware",
        "expert2_cnn_reconstructed_sources",
        "consensus_labelled_samples_only",
        "consensus_strong_agreement_only",
        "consensus_coverage",
    }
    print("Final SVM, CNN, and confidence-aware ensemble results:")
    for row in rows:
        if row.get("evaluation") in keep:
            print(row)


if __name__ == "__main__":
    main()
