import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("CUDA is available")
else:
    print("CUDA is not available")

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

def check_cudnn_availability():
    try:
        # Create a simple convolutional layer to test cuDNN availability
        layer = tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 2))
        input_data = tf.random.normal([1, 4, 4, 1])
        _ = layer(input_data)
        print("cuDNN is available")
    except RuntimeError as e:
        if 'could not create cudnn handle' in str(e).lower():
            print("cuDNN is not available")
        else:
            raise e

check_cudnn_availability()


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
