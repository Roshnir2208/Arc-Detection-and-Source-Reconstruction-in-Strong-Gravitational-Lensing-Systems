from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np

from lensing_pipeline.morphology_models import CANONICAL_CLASSES, MorphologyPrediction


@dataclass(frozen=True)
class ConsensusConfig:
    minimum_model_confidence: float = 0.60
    strong_agreement_threshold: float = 0.75
    minimum_valid_experts: int = 2


@dataclass
class MorphologyConsensus:
    consensus_label: str
    agreement: str
    valid_expert_count: int
    agreeing_expert_count: int
    mean_agreeing_confidence: float
    probability_entropy: float
    vote_margin: float
    disagreement_flag: bool
    model_status_summary: str

    def to_dict(self, prefix: str = "") -> dict[str, float | str | int | bool]:
        values = asdict(self)
        return {f"{prefix}{key}": value for key, value in values.items()}


def probability_entropy(probabilities: dict[str, float]) -> float:
    values = np.array([max(float(probabilities.get(label, 0.0)), 0.0) for label in CANONICAL_CLASSES], dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    values = values / total
    return float(-np.sum([p * math.log(p, 2) for p in values if p > 0]))


def combine_probabilities(predictions: list[MorphologyPrediction]) -> dict[str, float]:
    combined = {label: 0.0 for label in CANONICAL_CLASSES}
    if not predictions:
        return combined
    for prediction in predictions:
        if prediction.probabilities:
            for label in CANONICAL_CLASSES:
                combined[label] += float(prediction.probabilities.get(label, 0.0))
        else:
            combined[prediction.canonical_label] += float(prediction.confidence)
    for label in combined:
        combined[label] /= max(len(predictions), 1)
    return combined


def consensus_from_predictions(
    predictions: list[MorphologyPrediction],
    config: ConsensusConfig | None = None,
) -> MorphologyConsensus:
    config = config or ConsensusConfig()
    runnable = [p for p in predictions if p.status == "ran" and p.canonical_label in CANONICAL_CLASSES]
    valid = [p for p in runnable if p.confidence >= config.minimum_model_confidence and p.canonical_label != "uncertain"]
    status_counts = Counter(p.status for p in predictions)
    status_summary = ";".join(f"{key}:{value}" for key, value in sorted(status_counts.items()))

    if len(valid) < config.minimum_valid_experts:
        combined = combine_probabilities(runnable)
        return MorphologyConsensus(
            consensus_label="uncertain",
            agreement="weak",
            valid_expert_count=len(valid),
            agreeing_expert_count=0,
            mean_agreeing_confidence=0.0,
            probability_entropy=probability_entropy(combined),
            vote_margin=0.0,
            disagreement_flag=True,
            model_status_summary=status_summary,
        )

    votes = Counter(p.canonical_label for p in valid)
    ordered = votes.most_common()
    top_label, top_count = ordered[0]
    second_count = ordered[1][1] if len(ordered) > 1 else 0
    agreeing = [p for p in valid if p.canonical_label == top_label]
    mean_confidence = float(np.mean([p.confidence for p in agreeing])) if agreeing else 0.0
    all_agree = top_count == len(valid)
    strong = all_agree and all(p.confidence >= config.strong_agreement_threshold for p in valid)
    majority = top_count >= 2 and top_count > second_count
    combined = combine_probabilities(valid)

    if strong:
        label = top_label
        agreement = "strong"
    elif majority:
        label = top_label
        agreement = "majority"
    else:
        label = "uncertain"
        agreement = "weak"

    return MorphologyConsensus(
        consensus_label=label,
        agreement=agreement,
        valid_expert_count=len(valid),
        agreeing_expert_count=int(top_count if label != "uncertain" else 0),
        mean_agreeing_confidence=mean_confidence if label != "uncertain" else 0.0,
        probability_entropy=probability_entropy(combined),
        vote_margin=float((top_count - second_count) / max(len(valid), 1)),
        disagreement_flag=not all_agree,
        model_status_summary=status_summary,
    )
