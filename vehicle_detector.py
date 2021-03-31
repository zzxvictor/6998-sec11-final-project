import constants
from utils.DataLoader import DataLoader
from tensorflow.keras.applications import mobilenet
import tensorflow.keras as tfk


if __name__ == '__main__':
    data_loader = DataLoader(constants.DATA_ROOT,
                             constants.ANNOTATION_FILES)
    train, val = data_loader.load()
    # model = mobilenet.MobileNet(input_shape=(150, 150, 3),
    #                             weights=None,
    #                             pooling='avg',
    #                             include_top=False)
    # out_layer = tfk.layers.Dense(1, activation='softmax')(model.output)
    # model = tfk.Model(inputs=model.input, outputs=out_layer)
    # model.compile(loss=tfk.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tfk.optimizers.Adam(1e-4),
    #               metrics=['accuracy'])
    # model.fit(train,
    #           validation_data=val,
    #           epochs=20,
    #           steps_per_epoch=500)