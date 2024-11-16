import timm
import torch.nn as nn
from collections import Iterable

from utils import MODEL_PATH


class PENet(nn.Module):
    '''
    Parameter Estimation Network.
    '''
    def __init__(self, out_dim, backbone='resnet50d'):
        super(PENet, self).__init__()
        self.network = timm.create_model(
            backbone, pretrained=True,
            pretrained_cfg_overlay=dict(file=MODEL_PATH[backbone]),
            num_classes=out_dim
        )

    def forward(self, x):
        return self.network(x)


class PENetWrapper(nn.Module):
    '''
    Wrapper for Parameter Estimation Network. 
    '''
    def __init__(self, network, preprocess=None, postprocess=None):
        super(PENetWrapper, self).__init__()

        self.network = network
        self.preprocess = preprocess
        self.postprocess = postprocess

    def forward(self, x):
        if isinstance(self.preprocess, Iterable):
            for preprocess in self.preprocess:
                if callable(preprocess):
                    x = preprocess(x)
        elif callable(self.preprocess):
            x = self.preprocess(x)

        out = self.network(x)

        if isinstance(self.postprocess, Iterable):
            for postprocess in self.postprocess:
                if callable(postprocess):
                    out = postprocess(out)
        elif callable(self.postprocess):
            out = self.postprocess(out)
        return out
