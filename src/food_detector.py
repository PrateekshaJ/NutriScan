"""
Food image recognition using TensorFlow/Keras.

Loads a trained model from models/ when available; otherwise uses a
deterministic demo classifier for development and UI testing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from src.preprocessing import DEFAULT_IMAGE_SIZE, preprocess_for_model

# Project root (NutriScan/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "food_classifier.keras"
DEFAULT_LABELS_PATH = PROJECT_ROOT / "models" / "class_labels.json"


@dataclass(frozen=True)
class DetectionResult:
    """Food classification output."""

    food_name: str
    confidence: float
    top_predictions: List[Dict[str, float]]

    @property
    def confidence_percent(self) -> float:
        return round(self.confidence * 100, 2)


class FoodDetector:
    """
    Predict food category from an uploaded image.

    Usage:
        detector = FoodDetector()
        result = detector.predict("path/to/image.jpg")
    """

    # Demo labels when no trained model is present (per 100g reference foods)
    DEMO_CLASSES: List[str] = [
        "apple",
        "banana",
        "pizza",
        "burger",
        "salad",
        "pasta",
        "rice_bowl",
        "grilled_chicken",
        "fries",
        "ice_cream",
        "sushi",
        "sandwich",
        "steak",
        "soup",
        "yogurt",
    ]

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        labels_path: Optional[Union[str, Path]] = None,
        image_size: tuple = DEFAULT_IMAGE_SIZE,
    ) -> None:
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self.labels_path = Path(labels_path) if labels_path else DEFAULT_LABELS_PATH
        self.image_size = image_size
        self._model = None
        self._class_names: List[str] = []
        self._load_resources()

    def _load_resources(self) -> None:
        """Load Keras model and label map, or enable demo mode."""
        if self.labels_path.exists():
            with open(self.labels_path, encoding="utf-8") as f:
                data = json.load(f)
            self._class_names = data.get("classes", data) if isinstance(data, dict) else list(data)

        if self.model_path.exists():
            import tensorflow as tf

            self._model = tf.keras.models.load_model(self.model_path)
            if not self._class_names:
                output_dim = int(self._model.output_shape[-1])
                self._class_names = [f"class_{i}" for i in range(output_dim)]
        else:
            self._class_names = self._class_names or self.DEMO_CLASSES.copy()

    @property
    def is_demo_mode(self) -> bool:
        return self._model is None

    @property
    def class_names(self) -> List[str]:
        return list(self._class_names)

    def predict(self, image_source: Union[str, Path, bytes, np.ndarray]) -> DetectionResult:
        """
        Classify food in the given image.

        Args:
            image_source: Path, bytes, or BGR numpy array.

        Returns:
            DetectionResult with primary label, confidence, and top-5 list.
        """
        tensor = preprocess_for_model(image_source, self.image_size, add_batch_dim=True)

        if self._model is not None:
            probs = self._model.predict(tensor, verbose=0)[0]
        else:
            probs = self._demo_predict(tensor)

        top_indices = np.argsort(probs)[::-1][:5]
        top_predictions = [
            {
                "food": self._format_label(self._class_names[i]),
                "confidence": float(probs[i]),
            }
            for i in top_indices
            if i < len(self._class_names)
        ]

        best = top_predictions[0]
        return DetectionResult(
            food_name=best["food"],
            confidence=best["confidence"],
            top_predictions=top_predictions,
        )

    def _demo_predict(self, tensor: np.ndarray) -> np.ndarray:
        """
        Deterministic pseudo-probabilities from image statistics for demo UI.
        Replace by training with train_model.py and placing model in models/.
        """
        flat = tensor.flatten()
        seed = int(np.abs(flat.mean() * 1e4 + flat.std() * 1e3)) % (2**31)
        rng = np.random.default_rng(seed)
        logits = rng.standard_normal(len(self._class_names))
        # Bias toward healthier demo labels for varied images
        for i, name in enumerate(self._class_names):
            if name in ("salad", "apple", "banana", "grilled_chicken", "yogurt"):
                logits[i] += 0.4
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    @staticmethod
    def _format_label(raw: str) -> str:
        return raw.replace("_", " ").title()
