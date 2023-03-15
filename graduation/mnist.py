import math
import numpy as np
import tensorflow as tf
from keras.datasets import mnist as mnist_keras
from data.madry_mnist.model import Model as mnist_model

from victim_model import ModelNP

# MNIST_MODEL_PATH = 'data/madry_mnist/models/natural'
MNIST_MODEL_PATH = 'data/madry_mnist/models/secret'

CLASSES = 10

def load_data(n_ex):
    x_test, y_test = mnist_keras.load_data()[1]
    x_test = x_test.astype(np.float64) / 255.0
    x_test = x_test[:, None, :, :]

    return x_test[:n_ex], y_test[:n_ex]


class Model(ModelNP):
    def __init__(self, config):
        super(Model, self).__init__(config)
        
        model_file = tf.train.latest_checkpoint(MNIST_MODEL_PATH)
        
        self.model = mnist_model()
        if 'logits' not in self.model.__dict__:
            self.model.logits = self.model.pre_softmax

        self.sess = tf.Session()
        tf.train.Saver().restore(self.sess, model_file)

    def predict(self, x):
        shape = self.model.x_input.shape[1:].as_list()
        x = np.reshape(x, [-1, *shape])

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
