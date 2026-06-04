"""
NutriScan food classifier training script.

Expected dataset layout (add your data under data/):
    data/train/<class_name>/*.jpg
    data/val/<class_name>/*.jpg

Run:
    python train_model.py --data-dir data --epochs 10

Outputs:
    models/food_classifier.keras
    models/class_labels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf

from src.preprocessing import DEFAULT_IMAGE_SIZE

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
MODEL_OUT = PROJECT_ROOT / "models" / "food_classifier.keras"
LABELS_OUT = PROJECT_ROOT / "models" / "class_labels.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NutriScan food classifier")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Root folder containing train/ and val/ subdirectories",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_OUT,
        help="Path to save trained Keras model",
    )
    return parser.parse_args()


def build_datasets(
    data_dir: Path,
    image_size: int,
    batch_size: int,
) -> tuple:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"

    if not train_dir.exists():
        raise FileNotFoundError(
            f"Training folder not found: {train_dir}\n"
            "Create data/train/<class_name>/ with images per class."
        )

    img_size = (image_size, image_size)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="categorical",
    )
    class_names = train_ds.class_names

    if val_dir.exists():
        val_ds = tf.keras.utils.image_dataset_from_directory(
            val_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            class_names=class_names,
        )
    else:
        # Split 20% from train if no val folder
        full = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=img_size,
            batch_size=batch_size,
            label_mode="categorical",
            validation_split=0.2,
            subset="both",
            seed=42,
        )
        train_ds, val_ds = full

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(256).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)
    return train_ds, val_ds, class_names


def build_model(num_classes: int, image_size: int, learning_rate: float) -> tf.keras.Model:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_labels(class_names: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "classes": class_names,
        "version": "1.0.0",
        "description": "Auto-generated from train_model.py",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    args = parse_args()
    print(f"Loading datasets from {args.data_dir}...")
    train_ds, val_ds, class_names = build_datasets(
        args.data_dir, args.image_size, args.batch_size
    )
    print(f"Classes ({len(class_names)}): {class_names}")

    model = build_model(len(class_names), args.image_size, args.learning_rate)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    save_labels(class_names, LABELS_OUT)

    final_acc = history.history.get("val_accuracy", history.history["accuracy"])[-1]
    print(f"Model saved to {args.output}")
    print(f"Labels saved to {LABELS_OUT}")
    print(f"Final validation accuracy: {final_acc:.4f}")


if __name__ == "__main__":
    main()
