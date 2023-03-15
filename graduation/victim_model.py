import numpy as np


class ModelNP(object):
    def __init__(self, config):
        self.l_type     = config['loss']
        self.targeted   = config['targeted']
        self.batch_size = config['batch_size']

    def predict(self, x):
        raise NotImplementedError('Use sub-class of ModelNP')

    def loss(self, logits, label):
        one_hot_label = np.zeros_like(logits, dtype=np.bool)
        one_hot_label[np.arange(logits.shape[0]), label] = True
        
        if self.l_type == 'cw':    # negative CW loss
            diff = logits - logits[one_hot_label][:, np.newaxis]
            diff[one_hot_label] = -np.inf
            loss = -diff.max(axis=1) if self.targeted else diff.max(axis=1)
        elif self.l_type == 'ce':  # Cross-Entropy Loss
            # avoid overflow of 'exp' operation
            logits = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs  = logits / logits.sum(axis=1, keepdims=True)
            loss = -np.log(probs[one_hot_label])
            loss = -loss if self.targeted else loss
        else:
            raise NotImplementedError('Unknown loss: ' + self.l_type)
        return loss
    
    def correct(self, logits, label):
        return logits.argmax(axis=-1) == label
    
    def done(self, logits, label):
        if self.targeted:
            return logits.argmax(axis=-1) == label
        else:
            return logits.argmax(axis=-1) != label
