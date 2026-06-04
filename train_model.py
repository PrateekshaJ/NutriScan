import tensorflow as tf
import json
import os


DATA_DIR = "food_images"

IMG_SIZE = (224,224)
BATCH_SIZE = 32


train_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + "/train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


val_data = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR + "/val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)


classes = train_data.class_names


base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights="imagenet"
)


base_model.trainable = False


model = tf.keras.Sequential(
    [
        tf.keras.layers.Rescaling(1./255),

        base_model,

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            len(classes),
            activation="softmax"
        )
    ]
)


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)


os.makedirs(
    "models",
    exist_ok=True
)


model.save(
    "models/food_classifier.keras"
)


with open(
    "models/class_labels.json",
    "w"
) as f:

    json.dump(
        classes,
        f
    )


print("Food AI model trained 🥗🔥")