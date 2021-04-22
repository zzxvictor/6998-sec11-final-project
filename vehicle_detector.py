from configs import config
from utils.data_loader import DataLoader4Detector
from utils.callbacks import Logger, LrStepDecay
from vehicle_detection import detector
import tensorflow.keras as tfk

MODEL_NAME = 'simple_cnn'
if __name__ == '__main__':
    data_loader = DataLoader4Detector(config.DATA_ROOT,
                                      config.ANNOTATION_FILES)
    train, val = data_loader.load(batch_size=32, repeat=True)
    # for imgs, labels in train:
        # print(np.max(imgs), np.min(imgs))
        # plt.imshow(imgs[1].numpy())
        # plt.title(labels[1].numpy())
        # plt.show()
    model = detector.get_detector()

    model.compile(loss=tfk.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tfk.optimizers.Adam(5e-5),
                  metrics=['accuracy'])

    model_ckpt = tfk.callbacks.ModelCheckpoint(
        filepath='logs/vehicle_detection/{}/ckpt'.format(MODEL_NAME),
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    model.fit(train,
              validation_data=val,
              epochs=50,
              steps_per_epoch=500,
              validation_steps=50,
              callbacks=[Logger(log_dir='logs/vehicle_detection', name=MODEL_NAME),
                         LrStepDecay(decay_rate=0.5, decay_at=[5, 15, 30]),
                         model_ckpt])
    model.summary()