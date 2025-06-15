import tensorflow as tf

try:
    interpreter = tf.lite.Interpreter(model_path="models/ann_model.tflite")
    interpreter.allocate_tensors()
    print("TFLite model loaded successfully")
except Exception as e:
    print(f"Error loading TFLite model: {e}")