import os
import sys
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import vgg16_bn
from torchvision.models import inception_v3
from torchvision.models import densenet161
from torchvision.models import resnet152

import utils
import imagenet

from attacks import accelerated_sign_hunter as ash
from imagenet_labels import label_to_name

ADV_ROOT = 'adv'
MODEL_LIST = ['resnet50', 'vgg16_bn', 'inception_v3', 'densenet161', 'resnet152']


class AdvData(Dataset):
    def __init__(self, droot):
        self.trans = transforms.ToTensor()
        self.imgs = [os.path.join(droot, i) for i in os.listdir(droot)]
        self.labels = [i.split('&')[0] for i in os.listdir(droot)]
    
    def _load_img(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def __getitem__(self, index):
        img = self._load_img(self.imgs[index])
        
        if self.trans is not None:
            img = self.trans(img)
        return img, self.labels[index]

    def __len__(self):
        return len(self.imgs)

def _normalize(img):
    IMAGENET_MEAN = torch.tensor(imagenet.IMAGENET_MEAN)
    IMAGENET_STD = torch.tensor(imagenet.IMAGENET_STD)
    return ((img - IMAGENET_MEAN) / IMAGENET_STD).float()

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for smodel_type in MODEL_LIST:
        print('Surrogate Model: {:s}'.format(smodel_type))
        print('Attack Success Rate on Victim Model ', end='')
        droot = os.path.join(ADV_ROOT, smodel_type, 'adv')
        loader = DataLoader(AdvData(droot), batch_size=32, num_workers=3)
        for vmodel_type in MODEL_LIST:
            vmodel = eval(vmodel_type)(pretrained=True).to(device)
            count, total_num = 0, len(loader.dataset)
            for img, label in loader:
                img = _normalize(img).to(device)
                pred = [label_to_name(p) for p in vmodel(img).argmax(dim=-1)]
                count += torch.tensor([1 if p != l else 0 for p, l in zip(pred, label)]).sum()
            print('{:s}: {:.4f}, '.format(vmodel_type, count.item() / total_num), end='')
        print('')
