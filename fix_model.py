import tensorflow as tf

model = tf.keras.models.load_model(
    "models/food_classifier.keras",
    compile=False
)

model.export("models/food_classifier_fixed")

print("Fixed model saved")