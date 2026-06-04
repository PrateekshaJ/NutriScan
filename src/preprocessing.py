"""
Image preprocessing utilities for NutriScan.

Handles loading, resizing, normalization, and batch preparation
for TensorFlow/Keras inference and training pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import cv2
import numpy as np

# Default input size for MobileNet-style classifiers
DEFAULT_IMAGE_SIZE: Tuple[int, int] = (224, 224)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_bgr(source: Union[str, Path, bytes, np.ndarray]) -> np.ndarray:
    """
    Load an image as a BGR numpy array.

    Args:
        source: File path, raw bytes, or existing BGR array.

    Returns:
        BGR image array (H, W, 3).

    Raises:
        ValueError: If the image cannot be decoded.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2BGR)
        return source.copy()

    if isinstance(source, (str, Path)):
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"Could not read image from path: {source}")
        return image

    if isinstance(source, bytes):
        buffer = np.frombuffer(source, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image from bytes.")
        return image

    raise TypeError(f"Unsupported image source type: {type(source)}")


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """Resize image to (width, height) while preserving aspect ratio via center crop."""
    h, w = image.shape[:2]
    target_w, target_h = target_size

    scale = max(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    start_x = (new_w - target_w) // 2
    start_y = (new_h - target_h) // 2
    cropped = resized[start_y : start_y + target_h, start_x : start_x + target_w]
    return cropped


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert BGR OpenCV image to RGB."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def normalize_imagenet(rgb_image: np.ndarray) -> np.ndarray:
    """
    Scale uint8 RGB [0, 255] to float32 and apply ImageNet normalization.

    Returns:
        Normalized array with shape (H, W, 3), dtype float32.
    """
    scaled = rgb_image.astype(np.float32) / 255.0
    return (scaled - IMAGENET_MEAN) / IMAGENET_STD


def preprocess_for_model(
    source: Union[str, Path, bytes, np.ndarray],
    target_size: Tuple[int, int] = DEFAULT_IMAGE_SIZE,
    add_batch_dim: bool = True,
) -> np.ndarray:
    """
    Full pipeline: load → resize → RGB → ImageNet normalize → optional batch dim.

    Args:
        source: Image input (path, bytes, or array).
        target_size: Model input (width, height).
        add_batch_dim: If True, returns shape (1, H, W, 3).

    Returns:
        Model-ready float32 tensor.
    """
    bgr = load_image_bgr(source)
    resized = resize_image(bgr, target_size)
    rgb = bgr_to_rgb(resized)
    normalized = normalize_imagenet(rgb)
    if add_batch_dim:
        return np.expand_dims(normalized, axis=0)
    return normalized


def augment_training_sample(
    image: np.ndarray,
    flip_horizontal: bool = True,
    brightness_delta: float = 0.1,
) -> np.ndarray:
    """
    Lightweight augmentation for training notebooks and train_model.py.

    Args:
        image: BGR uint8 image.
        flip_horizontal: Random horizontal flip when True.
        brightness_delta: Max absolute brightness shift in [-delta, +delta].

    Returns:
        Augmented BGR image.
    """
    out = image.copy()
    if flip_horizontal and np.random.rand() > 0.5:
        out = cv2.flip(out, 1)
    if brightness_delta > 0:
        delta = np.random.uniform(-brightness_delta, brightness_delta)
        out = np.clip(out.astype(np.float32) * (1.0 + delta), 0, 255).astype(np.uint8)
    return out
