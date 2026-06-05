import tensorflow as tf
import numpy as np
import json


class FoodDetector:

    def __init__(self):

        self.model = tf.keras.models.load_model(
    "models/food_classifier.keras",
    compile=False
)

        with open(
            "models/class_labels.json"
        ) as f:
            self.labels = json.load(f)


    def predict(self, image):

        # convert every image to RGB
        img = image.convert("RGB")

        img = img.resize(
            (224, 224)
        )

        arr = tf.keras.utils.img_to_array(
            img
        )

        # same scaling as training
        arr = arr / 255.0

        arr = np.expand_dims(
            arr,
            axis=0
        )

        prediction = self.model.predict(
            arr
        )

        index = np.argmax(
            prediction
        )

        return (
            self.labels[str(index)] if isinstance(self.labels, dict)
            else self.labels[index],

            round(
                float(prediction[0][index]) * 100,
                2
            )
        )