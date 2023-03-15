import math
import numpy as np
import tensorflow as tf
from data.madry_cifar10.cifar10_input import CIFAR10Data
from data.madry_cifar10.model import Model as cifar10_model

from victim_model import ModelNP

CIFAR10_ROOT = 'data/madry_cifar10/cifar10_data'
# CIFAR10_MODEL_PATH = 'data/madry_cifar10/models/naturally_trained'
CIFAR10_MODEL_PATH = 'data/madry_cifar10/models/model_0'

CLASSES = 10

def load_data(n_ex):
    cifar = CIFAR10Data(CIFAR10_ROOT)
    x_test, y_test = cifar.eval_data.xs.astype(np.float32), cifar.eval_data.ys
    x_test = np.transpose(x_test, axes=[0, 3, 1, 2])
    return x_test[:n_ex], y_test[:n_ex]


class Model(ModelNP):
    def __init__(self, config):
        super(Model, self).__init__(config)
        
        model_file = tf.train.latest_checkpoint(CIFAR10_MODEL_PATH)
        
        self.model = cifar10_model()
        if 'logits' not in self.model.__dict__:
            self.model.logits = self.model.pre_softmax

        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, model_file)

    def predict(self, x):
        x = np.transpose(x, axes=[0, 2, 3, 1])

        bs = self.batch_size
        bn = math.ceil(x.shape[0] / bs)
        
        logits_all = []
        for i in range(bn):
            x_batch = x[i * bs: (i + 1) * bs]
            logits = self.sess.run(
                self.model.logits, feed_dict={self.model.x_input: x_batch})
            logits_all.append(logits)
        logits_all = np.vstack(logits_all)
        return logits_all
