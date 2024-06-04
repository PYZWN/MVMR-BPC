import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
    for device in tf.config.list_physical_devices('GPU'):
        print("Device name:", device.name)
else:
    print("GPU is not available, using CPU instead.")
