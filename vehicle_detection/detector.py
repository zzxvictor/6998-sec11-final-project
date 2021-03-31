import tensorflow.keras as tfk
import tensorflow.keras.layers as layers

def get_detector(input_shape):
    model = tfk.Sequential()
    # model.add(layers.Conv2D(kernel_size=5, str))