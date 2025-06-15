import tensorflow as tf

Load the Keras model from .h5 file
model = tf.keras.models.load_model("C:/Users/asus/Desktop/Sourabh/College/IndicLanguageDetection/models/ann_model.h5")

Convert the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

Save the TFLite model
with open("models/ann_model.tflite", "wb") as f:
f.write(tflite_model)

print("Successfully converted ann_model.h5 to ann_model.tflite")
