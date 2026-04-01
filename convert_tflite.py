import tensorflow as tf
model = tf.keras.models.load_model("models/best_model.h5")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
open("models/best_model_quant.tflite", "wb").write(tflite_model)
print("Saved models/best_model_quant.tflite")
