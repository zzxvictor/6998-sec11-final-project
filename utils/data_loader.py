import tensorflow as tf
from tensorflow.keras.applications.resnet import preprocess_input
import numpy as np


class DataLoader:

    def __init__(self,
                 data_root,
                 annotation_file):
        self.data_root = data_root
        self.annotation_file = annotation_file
        self.annotation = self._read_annotation()

    def load(self,
             batch_size=64,
             repeat=False,
             ratio=0.2,
             seed=123,
             brightness_delta=0.05,
             contrast_factor=0.8,
             img_size=150):
        self.brightness_delta = brightness_delta
        self.contrast_factor = contrast_factor
        self.img_size = img_size

        np.random.seed(seed)
        np.random.shuffle(self.annotation)
        split_idx = int(len(self.annotation) * (1 - ratio))
        train, val = self.annotation[: split_idx], self.annotation[split_idx: ]
        train_data = self._load(train).batch(batch_size).prefetch(batch_size * 2)
        val_data = self._load(val, train=False).batch(batch_size)
        if repeat:
            train_data = train_data.repeat()
        train_data.shuffle(buffer_size=128)
        return train_data, val_data

    def _load(self, records, train=True):
        data = tf.data.Dataset.from_tensor_slices(records)
        data = data.map(lambda x: self._parse_record(x, train))
        return data

    def _parse_record(self, line, train):
        outputs = tf.strings.split(line, sep=' ')
        img_path, label = outputs[0], outputs[1]
        img_path = tf.strings.join(inputs=[self.data_root, 'PATCHES/', img_path])
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3) / 255 - 0.5
        img = tf.image.resize_with_crop_or_pad(img,
                                               target_height=self.img_size,
                                               target_width=self.img_size)
        # add noise
        if train:
            img = tf.image.random_brightness(img, max_delta=self.brightness_delta)
            img = tf.image.random_contrast(img, lower=self.contrast_factor,
                                           upper=1 / self.contrast_factor)

        return img, tf.strings.to_number(label, tf.float32)

    def _read_annotation(self):
        records = []
        with open(self.annotation_file, 'r') as fp:
            for line in fp.readlines():
                records.append(line.strip('\n'))
        return records


