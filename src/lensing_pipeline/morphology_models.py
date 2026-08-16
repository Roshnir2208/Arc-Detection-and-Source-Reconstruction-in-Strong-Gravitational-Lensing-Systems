from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage

from lensing_pipeline.detection import convolve2d, normalize_image


CANONICAL_CLASSES = ["spiral", "elliptical", "peculiar", "irregular", "uncertain"]


@dataclass(frozen=True)
class ModelRegistryEntry:
    model_name: str
    framework: str
    source: str
    expected_input_dimensions: tuple[int, int]
    expected_channels: int
    normalisation: str
    native_labels: list[str]
    label_mapping: dict[str, str]
    confidence_output: str
    licence: str
    status: str
    notes: str


@dataclass
class MorphologyPrediction:
    model_name: str
    native_label: str
    canonical_label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    preprocessing_used: str = "raw_source_crop"
    inference_time: float = 0.0
    status: str = "ran"
    error_message: str = ""


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    channels: int = 3
    padding_fraction: float = 0.18
    mode: str = "raw_source_crop"


def canonicalise_label(native_label: str, mapping: dict[str, str] | None = None) -> str:
    text = str(native_label).strip().lower().replace("-", "_").replace(" ", "_")
    if mapping and text in mapping:
        return mapping[text]
    aliases = {
        "spiral": "spiral",
        "spiral_galaxy": "spiral",
        "barred_spiral": "spiral",
        "unbarred_spiral": "spiral",
        "disk": "spiral",
        "disc": "spiral",
        "elliptical": "elliptical",
        "elliptical_galaxy": "elliptical",
        "smooth": "elliptical",
        "round": "elliptical",
        "peculiar": "peculiar",
        "merger": "peculiar",
        "disturbed": "peculiar",
        "irregular": "irregular",
        "unknown": "uncertain",
        "uncertain": "uncertain",
    }
    return aliases.get(text, "uncertain")


def default_model_registry() -> list[ModelRegistryEntry]:
    return [
        ModelRegistryEntry(
            model_name="feature_structure_expert",
            framework="classical_features",
            source="local feature extractor; no external checkpoint",
            expected_input_dimensions=(128, 128),
            expected_channels=1,
            normalisation="NaN removal, non-negative scaling, centroid crop, aspect-preserving resize",
            native_labels=["spiral", "elliptical", "peculiar", "irregular", "uncertain"],
            label_mapping={label: label for label in CANONICAL_CLASSES},
            confidence_output="heuristic confidence from morphology feature margins",
            licence="project local code",
            status="compatible",
            notes="Independent non-neural expert using concentration, axis ratio, elongation, and effective radius. It is not supervised unless calibrated with labelled data.",
        ),
        ModelRegistryEntry(
            model_name="feature_texture_expert",
            framework="classical_features",
            source="local feature extractor; no external checkpoint",
            expected_input_dimensions=(128, 128),
            expected_channels=1,
            normalisation="NaN removal, non-negative scaling, centroid crop, aspect-preserving resize",
            native_labels=["spiral", "elliptical", "peculiar", "irregular", "uncertain"],
            label_mapping={label: label for label in CANONICAL_CLASSES},
            confidence_output="heuristic confidence from texture, asymmetry, clumpiness, Gini, and M20 feature margins",
            licence="project local code",
            status="compatible",
            notes="Second independent non-neural expert with different feature emphasis from the structure expert.",
        ),
        ModelRegistryEntry(
            model_name="hf_image_classification_optional",
            framework="transformers",
            source="user-supplied Hugging Face image-classification model id",
            expected_input_dimensions=(224, 224),
            expected_channels=3,
            normalisation="model pipeline default after morphology-preserving crop",
            native_labels=[],
            label_mapping={
                "spiral": "spiral",
                "spiral_galaxy": "spiral",
                "barred_spiral": "spiral",
                "elliptical": "elliptical",
                "elliptical_galaxy": "elliptical",
                "smooth": "elliptical",
                "peculiar": "peculiar",
                "merger": "peculiar",
                "irregular": "irregular",
                "unknown": "uncertain",
            },
            confidence_output="model confidence score returned by transformers pipeline; uncalibrated unless separately calibrated",
            licence="see selected Hugging Face model card",
            status="uncertain",
            notes="Only compatible when the selected model card documents morphology labels and preprocessing.",
        ),
    ]


def write_model_registry(path: Path, entries: list[ModelRegistryEntry] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = entries or default_model_registry()
    path.write_text(json.dumps([asdict(entry) for entry in entries], indent=2), encoding="utf-8")


def load_image_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=float)
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=float) / 255.0


def supported_source_mask(image: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if float(np.max(arr)) <= 0:
        return np.zeros(arr.shape, dtype=bool)
    norm = normalize_image(arr)
    positive = norm[norm > 0]
    threshold = max(float(np.percentile(positive, 45.0)), 0.025) if len(positive) else 0.025
    mask = norm >= threshold
    labels, count = ndimage.label(mask)
    if count <= 1:
        return mask
    best_label = 0
    best_flux = -1.0
    for label in range(1, count + 1):
        flux = float(norm[labels == label].sum())
        if flux > best_flux:
            best_flux = flux
            best_label = label
    return ndimage.binary_dilation(labels == best_label, iterations=2)


def crop_around_centroid(image: np.ndarray, padding_fraction: float = 0.18) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    mask = supported_source_mask(arr)
    if not np.any(mask):
        return normalize_image(arr)
    yy, xx = np.nonzero(mask)
    pad = max(3, int(round(padding_fraction * max(int(yy.max() - yy.min() + 1), int(xx.max() - xx.min() + 1)))))
    y0 = max(0, int(yy.min()) - pad)
    y1 = min(arr.shape[0], int(yy.max()) + pad + 1)
    x0 = max(0, int(xx.min()) - pad)
    x1 = min(arr.shape[1], int(xx.max()) + pad + 1)
    return normalize_image(arr[y0:y1, x0:x1])


def resize_preserve_aspect(image: np.ndarray, size: int) -> np.ndarray:
    arr = np.asarray(image, dtype=float)
    if arr.size == 0:
        return np.zeros((size, size), dtype=float)
    height, width = arr.shape
    side = max(height, width, 1)
    square = np.zeros((side, side), dtype=float)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    square[y0 : y0 + height, x0 : x0 + width] = arr
    zoom = size / side
    resized = ndimage.zoom(square, zoom=zoom, order=1)
    out = np.zeros((size, size), dtype=float)
    h = min(size, resized.shape[0])
    w = min(size, resized.shape[1])
    out[:h, :w] = resized[:h, :w]
    return normalize_image(out)


def preprocess_for_classifier(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(image, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.clip(arr, 0.0, None)
    if config.mode == "background_subtracted":
        arr = np.clip(arr - np.percentile(arr, 10.0), 0.0, None)
    elif config.mode == "log_intensity":
        arr = np.log1p(12.0 * normalize_image(arr))
    elif config.mode == "asinh_intensity":
        arr = np.arcsinh(8.0 * normalize_image(arr))
    elif config.mode != "raw_source_crop":
        raise ValueError(f"Unknown preprocessing mode: {config.mode}")
    cropped = crop_around_centroid(arr, config.padding_fraction)
    resized = resize_preserve_aspect(cropped, config.image_size)
    if config.channels == 3:
        return np.repeat(resized[..., None], 3, axis=2)
    return resized


def save_classifier_ready_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(image, dtype=float)
    if arr.ndim == 3:
        out = np.clip(arr, 0.0, 1.0)
        Image.fromarray((out * 255).astype(np.uint8), mode="RGB").save(path)
    else:
        Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="L").save(path)


def gini_coefficient(values: np.ndarray) -> float:
    data = np.sort(np.clip(np.asarray(values, dtype=float).ravel(), 0.0, None))
    if len(data) == 0 or float(np.sum(data)) <= 0:
        return 0.0
    index = np.arange(1, len(data) + 1, dtype=float)
    return float(np.sum((2.0 * index - len(data) - 1.0) * data) / (len(data) * np.sum(data)))


def m20_statistic(image: np.ndarray) -> float:
    arr = normalize_image(np.clip(np.asarray(image, dtype=float), 0.0, None))
    total_flux = float(arr.sum())
    if total_flux <= 0:
        return math.nan
    yy, xx = np.indices(arr.shape)
    cx = float((xx * arr).sum() / total_flux)
    cy = float((yy * arr).sum() / total_flux)
    moment = arr * ((xx - cx) ** 2 + (yy - cy) ** 2)
    total_moment = float(moment.sum())
    if total_moment <= 0:
        return math.nan
    flat_flux = arr.ravel()
    flat_moment = moment.ravel()
    order = np.argsort(flat_flux)[::-1]
    cumulative_flux = np.cumsum(flat_flux[order])
    keep = cumulative_flux <= 0.2 * total_flux
    if not np.any(keep):
        keep[0] = True
    return float(np.log10(max(float(flat_moment[order][keep].sum()), 1e-12) / total_moment))


def hu_moments(image: np.ndarray) -> list[float]:
    arr = normalize_image(np.clip(np.asarray(image, dtype=float), 0.0, None))
    yy, xx = np.indices(arr.shape)
    m00 = float(arr.sum())
    if m00 <= 0:
        return [0.0] * 7
    cx = float((xx * arr).sum() / m00)
    cy = float((yy * arr).sum() / m00)

    def mu(p: int, q: int) -> float:
        return float((((xx - cx) ** p) * ((yy - cy) ** q) * arr).sum())

    def eta(p: int, q: int) -> float:
        return mu(p, q) / max(m00 ** (1.0 + 0.5 * (p + q)), 1e-12)

    n20, n02, n11 = eta(2, 0), eta(0, 2), eta(1, 1)
    n30, n12, n21, n03 = eta(3, 0), eta(1, 2), eta(2, 1), eta(0, 3)
    return [
        n20 + n02,
        (n20 - n02) ** 2 + 4 * n11**2,
        (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2,
        (n30 + n12) ** 2 + (n21 + n03) ** 2,
        (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        + (3 * n21 - n03) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
        (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03),
        (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2)
        - (n30 - 3 * n12) * (n21 + n03) * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2),
    ]


def morphology_features(image: np.ndarray) -> dict[str, float]:
    arr = normalize_image(np.clip(np.asarray(image, dtype=float), 0.0, None))
    yy, xx = np.indices(arr.shape)
    total = float(arr.sum()) + 1e-12
    cx = float((xx * arr).sum() / total)
    cy = float((yy * arr).sum() / total)
    radius = np.hypot(xx - cx, yy - cy)
    max_radius = float(np.hypot(arr.shape[1] / 2.0, arr.shape[0] / 2.0))
    inner = float(arr[radius <= 0.25 * max_radius].sum())
    outer = float(arr[radius <= 0.75 * max_radius].sum()) + 1e-12
    concentration = inner / outer
    rotated = np.rot90(arr, 2)
    asymmetry = float(np.abs(arr - rotated).sum() / (float(arr.sum()) + 1e-12))
    smoothed = ndimage.gaussian_filter(arr, sigma=1.2)
    clumpiness = float(np.clip(np.sum(np.clip(arr - smoothed, 0.0, None)) / (float(arr.sum()) + 1e-12), 0.0, 1.0))
    grad_x = convolve2d(arr, np.array([[-1, 0, 1]], dtype=float))
    grad_y = convolve2d(arr, np.array([[-1], [0], [1]], dtype=float))
    edge_strength = float(np.mean(np.hypot(grad_x, grad_y)))
    active = arr > max(float(np.percentile(arr, 75.0)), 0.04)
    if int(active.sum()) >= 5:
        coords = np.column_stack([xx[active].astype(float), yy[active].astype(float)])
        centred = coords - coords.mean(axis=0)
        eig = np.maximum(np.linalg.eigvalsh(np.cov(centred, rowvar=False)), 1e-8)
        axis_ratio = float(np.sqrt(eig[0] / eig[1]))
        elongation = float(1.0 / max(axis_ratio, 1e-6))
        effective_radius = float(np.sqrt(eig.sum()))
    else:
        axis_ratio = 1.0
        elongation = 1.0
        effective_radius = 0.0
    hu = hu_moments(arr)
    features = {
        "concentration": float(concentration),
        "asymmetry": asymmetry,
        "clumpiness": clumpiness,
        "edge_strength": edge_strength,
        "gini": gini_coefficient(arr),
        "m20": m20_statistic(arr),
        "axis_ratio": axis_ratio,
        "ellipticity": float(1.0 - axis_ratio),
        "elongation": elongation,
        "effective_radius": effective_radius,
        "central_intensity_ratio": float(inner / (float(arr.sum()) + 1e-12)),
        "active_fraction": float(np.mean(active)),
    }
    for index, value in enumerate(hu, start=1):
        features[f"hu_{index}"] = float(value)
    return features


def structure_feature_expert_predict(image: np.ndarray, preprocessing_used: str = "raw_source_crop") -> MorphologyPrediction:
    start = time.perf_counter()
    try:
        features = morphology_features(image)
        if features["concentration"] > 0.44 and features["axis_ratio"] > 0.55 and features["clumpiness"] < 0.26:
            native = "elliptical"
            confidence = min(0.90, 0.55 + 0.45 * features["concentration"] + 0.15 * features["axis_ratio"])
        elif features["elongation"] > 3.2 or features["axis_ratio"] < 0.30:
            native = "peculiar"
            confidence = min(0.82, 0.53 + 0.07 * features["elongation"])
        elif features["active_fraction"] > 0.018 and features["effective_radius"] > 6:
            native = "spiral"
            confidence = min(0.80, 0.52 + 0.25 * features["active_fraction"] + 0.04 * features["effective_radius"])
        elif features["asymmetry"] > 1.10:
            native = "irregular"
            confidence = min(0.80, 0.52 + 0.18 * features["asymmetry"])
        else:
            native = "uncertain"
            confidence = 0.45
        probabilities = {label: 0.0 for label in CANONICAL_CLASSES}
        probabilities[canonicalise_label(native)] = float(confidence)
        probabilities["uncertain"] = max(probabilities["uncertain"], float(1.0 - confidence))
        return MorphologyPrediction(
            model_name="feature_structure_expert",
            native_label=native,
            canonical_label=canonicalise_label(native),
            confidence=float(confidence),
            probabilities=probabilities,
            preprocessing_used=preprocessing_used,
            inference_time=time.perf_counter() - start,
        )
    except Exception as exc:
        return MorphologyPrediction(
            model_name="feature_structure_expert",
            native_label="",
            canonical_label="uncertain",
            confidence=0.0,
            preprocessing_used=preprocessing_used,
            inference_time=time.perf_counter() - start,
            status="error",
            error_message=str(exc),
        )


def texture_feature_expert_predict(image: np.ndarray, preprocessing_used: str = "raw_source_crop") -> MorphologyPrediction:
    start = time.perf_counter()
    try:
        features = morphology_features(image)
        if features["asymmetry"] > 1.05 or features["clumpiness"] > 0.38:
            native = "irregular"
            confidence = min(0.88, 0.50 + 0.22 * features["asymmetry"] + 0.35 * features["clumpiness"])
        elif features["edge_strength"] > 0.055 and features["gini"] > 0.45 and features["active_fraction"] > 0.018:
            native = "spiral"
            confidence = min(0.84, 0.52 + 2.8 * features["edge_strength"] + 0.12 * features["gini"])
        elif features["m20"] > -0.55 or (features["clumpiness"] > 0.25 and features["asymmetry"] > 0.72):
            native = "peculiar"
            confidence = min(0.82, 0.52 + 0.25 * features["clumpiness"] + 0.12 * features["asymmetry"])
        elif features["concentration"] > 0.38 and features["edge_strength"] < 0.05:
            native = "elliptical"
            confidence = min(0.82, 0.54 + 0.35 * features["concentration"])
        else:
            native = "uncertain"
            confidence = 0.44
        probabilities = {label: 0.0 for label in CANONICAL_CLASSES}
        probabilities[canonicalise_label(native)] = float(confidence)
        probabilities["uncertain"] = max(probabilities["uncertain"], float(1.0 - confidence))
        return MorphologyPrediction(
            model_name="feature_texture_expert",
            native_label=native,
            canonical_label=canonicalise_label(native),
            confidence=float(confidence),
            probabilities=probabilities,
            preprocessing_used=preprocessing_used,
            inference_time=time.perf_counter() - start,
        )
    except Exception as exc:
        return MorphologyPrediction(
            model_name="feature_texture_expert",
            native_label="",
            canonical_label="uncertain",
            confidence=0.0,
            preprocessing_used=preprocessing_used,
            inference_time=time.perf_counter() - start,
            status="error",
            error_message=str(exc),
        )


def feature_expert_predict(image: np.ndarray, preprocessing_used: str = "raw_source_crop") -> MorphologyPrediction:
    return structure_feature_expert_predict(image, preprocessing_used)


class HuggingFaceMorphologyExpert:
    def __init__(self, model_id: str, label_mapping: dict[str, str] | None = None, device: int = -1):
        self.model_id = model_id
        self.label_mapping = label_mapping or default_model_registry()[-1].label_mapping
        self.device = device
        self.pipeline: Any | None = None
        self.load_error = ""
        try:
            from transformers import pipeline

            self.pipeline = pipeline("image-classification", model=model_id, device=device)
        except Exception as exc:
            self.load_error = str(exc)

    def predict(self, image: np.ndarray, preprocessing_used: str = "raw_source_crop") -> MorphologyPrediction:
        start = time.perf_counter()
        if self.pipeline is None:
            return MorphologyPrediction(
                model_name=f"huggingface:{self.model_id}",
                native_label="",
                canonical_label="uncertain",
                confidence=0.0,
                preprocessing_used=preprocessing_used,
                inference_time=time.perf_counter() - start,
                status="not_run_model_load_failed",
                error_message=self.load_error,
            )
        try:
            arr = np.asarray(image, dtype=float)
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
            pil = Image.fromarray((np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8), mode="RGB")
            result = self.pipeline(pil)
            result = result if isinstance(result, list) else [result]
            probabilities: dict[str, float] = {label: 0.0 for label in CANONICAL_CLASSES}
            best_native = ""
            best_score = 0.0
            for item in result:
                native = str(item.get("label", ""))
                score = float(item.get("score", 0.0))
                canonical = canonicalise_label(native, self.label_mapping)
                probabilities[canonical] = max(probabilities.get(canonical, 0.0), score)
                if score > best_score:
                    best_native = native
                    best_score = score
            return MorphologyPrediction(
                model_name=f"huggingface:{self.model_id}",
                native_label=best_native,
                canonical_label=canonicalise_label(best_native, self.label_mapping),
                confidence=float(best_score),
                probabilities=probabilities,
                preprocessing_used=preprocessing_used,
                inference_time=time.perf_counter() - start,
            )
        except Exception as exc:
            return MorphologyPrediction(
                model_name=f"huggingface:{self.model_id}",
                native_label="",
                canonical_label="uncertain",
                confidence=0.0,
                preprocessing_used=preprocessing_used,
                inference_time=time.perf_counter() - start,
                status="error",
                error_message=str(exc),
            )
