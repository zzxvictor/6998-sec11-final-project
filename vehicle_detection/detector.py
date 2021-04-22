import tensorflow.keras as tfk
import tensorflow.keras.layers as layers


def get_detector():
    model = tfk.Sequential()
    model.add(layers.Conv2D(filters=16, kernel_size=7, strides=1, padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same'))
    model.add(layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same'))
    model.add(layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.Conv2D(filters=256, kernel_size=3, strides=1, padding='same'))
    model.add(layers.MaxPool2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(20, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model
