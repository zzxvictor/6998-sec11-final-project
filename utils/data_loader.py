import tensorflow as tf
import numpy as np


class DataLoader:
    def __init__(self,
                 data_root,
                 annotation_file):
        self.data_root = data_root
        self.annotation_file = annotation_file
        self.annotation = self._read_annotation(annotation_file)

    @classmethod
    def _read_annotation(cls, path):
        records = []
        with open(path, 'r') as fp:
            for line in fp.readlines():
                records.append(line.strip('\n'))
        return records

    @classmethod
    def _load_img(cls, img_path, img_h, img_w):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3) / 255 - 0.5
        img = tf.image.resize(img,
                              antialias=True,
                              size=[img_h, img_w])
        return img


class DataLoader4Signature(DataLoader):
    def __init__(self,
                 data_root,
                 annotation_file):
        super(DataLoader4Signature, self).__init__(data_root, annotation_file)
        self.delimiter = '|'

    def load(self,
             batch_size=64,
             repeat=False,
             ratio=0.2,
             seed=123,
             brightness_delta=0.1,
             contrast_factor=0.7,
             img_size=150):

        self.brightness_delta = brightness_delta
        self.contrast_factor = contrast_factor
        self.img_size = img_size
        np.random.seed(seed)
        np.random.shuffle(self.annotation)
        split_idx = int(len(self.annotation) * (1 - ratio))
        train, val = self.annotation[: split_idx], self.annotation[split_idx: ]
        train_data = self._load(train).batch(batch_size).prefetch(batch_size * 2)
        val_data = self._load(val).batch(batch_size)
        return train_data, val_data

    def _load(self, records):
        positives = [record[:-2] for record in records if record.endswith('1')]
        negatives = [record[:-2] for record in records if record.endswith('0')]
        records = self._generate_pairs(positives, negatives)
        data = tf.data.Dataset.from_tensor_slices(records)
        data = data.map(self._parse_record)
        return data

    def _parse_record(self, line):
        data = tf.strings.split(line, sep=self.delimiter)
        img_1 = self._processing(data[0])
        img_2 = self._processing(data[1])
        return img_1, img_2, tf.strings.to_number(data[-1], tf.float32)

    def _processing(self, path):
        img_path = tf.strings.join(inputs=[self.data_root, 'PATCHES/', path])
        img = self._load_img(img_path, self.img_size + 20, self.img_size + 20)
        img = tf.image.random_crop(img, size=[self.img_size, self.img_size, 3])
        img = tf.image.random_brightness(img, max_delta=self.brightness_delta)
        img = tf.image.random_contrast(img, lower=self.contrast_factor,
                                       upper=1 / self.contrast_factor)
        return img

    def _generate_pairs(self, positives, negatives):
        pairs = []
        for idx, img in enumerate(positives):
            pairs.append(self.delimiter.join([img, img, '1']))

            other = np.random.randint(low=0, high=len(positives))
            if other != idx:
                pairs.append(self.delimiter.join([img, positives[other], '0']))

            other = np.random.randint(low=0, high=len(negatives))
            pairs.append(self.delimiter.join([img, negatives[other], '0']))
        np.random.shuffle(pairs)
        return pairs


class DataLoader4Detector(DataLoader):

    def __init__(self,
                 data_root,
                 annotation_file):
        super(DataLoader4Detector, self).__init__(data_root, annotation_file)

    def load(self,
             batch_size=64,
             repeat=False,
             ratio=0.2,
             seed=123,
             brightness_delta=0.0,
             contrast_factor=0.1,
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
        tf.print(line)
        img_path, label = outputs[0], outputs[1]
        img_path = tf.strings.join(inputs=[self.data_root, 'PATCHES/', img_path])
        img = self._load_img(img_path, self.img_size, self.img_size)
        # add noise
        if train:
            pass
            img = tf.image.random_brightness(img, max_delta=self.brightness_delta)
            img = tf.image.random_contrast(img, lower=self.contrast_factor,
                                           upper=1 / self.contrast_factor)

        return img, tf.strings.to_number(label, tf.float32)




