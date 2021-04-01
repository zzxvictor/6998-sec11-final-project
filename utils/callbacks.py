import tensorflow.keras as tfk
import tensorflow as tf
import tensorflow.keras.layers as layers
import json
import collections
from datetime import datetime
import os


class LrStepDecay(tfk.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 decay_at):
        super(LrStepDecay, self).__init__()
        self.decay_rate = decay_rate
        self.decay_at = decay_at
        self.counter = 0

    def on_epoch_end(self, epoch, logs=None):
        if self.counter >= len(self.decay_at):
            return

        if epoch >= self.decay_at[self.counter]:
            self.counter += 1
            new_lr = float(tfk.backend.get_value(self.model.optimizer.learning_rate)) * self.decay_rate
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            print("\nEpoch %05d: Learning rate is %3.6f." % (epoch, new_lr))


class Logger(tfk.callbacks.Callback):

    def __init__(self,
                 name,
                 log_dir):
        super(Logger, self).__init__()
        self.name = name
        self.log_dir = log_dir
        self.log = collections.defaultdict(list)
        self.start_time = datetime.now()
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

    def on_epoch_begin(self, epoch, logs=None):
        self.start_time = datetime.now()

    def on_epoch_end(self, epoch, logs=None):
        file = open('{}/{}.json'.format(self.log_dir, self.name), 'w')
        for key in logs:
            self.log[key].append(logs[key])
        self.log['epoch'].append(epoch)
        self.log['walltime'].append((datetime.now() - self.start_time).seconds)
        json.dump(self.log, file)
        file.close()