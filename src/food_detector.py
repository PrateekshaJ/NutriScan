import tensorflow as tf
import numpy as np
import json


class FoodDetector:

    def __init__(self):

        self.model = tf.keras.layers.TFSMLayer(
            "models/food_classifier_fixed",
            call_endpoint="serve"
        )

        with open("models/class_labels.json") as f:
            self.labels = json.load(f)


    def predict(self, image):

        # convert image to RGB
        img = image.convert("RGB")

        # resize for model
        img = img.resize((224, 224))

        # image to array
        arr = tf.keras.utils.img_to_array(img)

        # normalize
        arr = arr / 255.0

        # add batch dimension
        arr = np.expand_dims(arr, axis=0)

        # run model
        prediction = self.model(arr)

        # tensor -> numpy
        prediction = prediction.numpy()

        index = np.argmax(prediction)

        confidence = round(
            float(prediction[0][index]) * 100,
            2
        )

        if isinstance(self.labels, dict):
            food = self.labels[str(index)]
        else:
            food = self.labels[index]

        return food, confidence