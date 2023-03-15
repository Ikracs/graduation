import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import imagenet
from imagenet_labels import label_to_name

droot = 'adv/resnet50/adv'
cfg = {
    'model': 'resnet50',
    'batch_size': 32,
    'targeted': False,
    'loss': None
}
if __name__ == '__main__':
    vmodel = imagenet.Model(cfg)
    count = 0
    for i in os.listdir(droot):
        with open(os.path.join(droot, i), 'rb') as f:
            img = Image.open(f).convert('RGB')
            img = np.array(transforms.ToTensor()(img))[np.newaxis, ...]
            label = label_to_name(vmodel.predict(img).argmax().item())
            count += 1 if label == i.split('#')[0] else 0
    print(count / len(os.listdir(droot)))

