#!/usr/bin/python3

import tensorflow as tf
from tensorflow import python as tf_python

######## Conversion #########
tf_model = tf.keras.models.load_model('model_files/direct_output.h5')

# Setting batch size into 1 to prevent this error while inferring the model=> ERROR: Attempting to use a delegate that only supports static-sized tensors
for i, _ in enumerate(tf_model.inputs):
    tf_model.inputs[i].shape._dims[0] = tf_python.framework.tensor_shape.Dimension(1)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)

tflite_model_quantized = converter.convert()
tflite_model_quantized_file = 'fp32_direct_output.tflite'

with open(tflite_model_quantized_file, 'wb') as f:
    f.write(tflite_model_quantized)
