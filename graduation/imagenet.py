import os
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models import vgg16_bn
from torchvision.models import inception_v3
from torchvision.models import densenet161
from torchvision.models import resnet152

from victim_model import ModelNP

IMAGENET_ROOT = '/mnt/datadev_2/std/lisiyuan/imagenet'

IMAGENET_TRANS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

CLASSES = 1000

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

class ImageNet(Dataset):
    def __init__(self, trans=None):
        self.trans = trans
        droot = os.path.join(IMAGENET_ROOT, 'val')
        self.imgs = sorted([os.path.join(droot, i) for i in os.listdir(droot)])

        label_path = os.path.join(IMAGENET_ROOT, 'val.txt')
        with open(label_path) as label_file:
            labels = [i.split(' ') for i in label_file.read().strip().split('\n')]
            self.labels = {os.path.basename(i[0]): int(i[1]) for i in labels}

    def _load_img(self, path):
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')

    def __getitem__(self, index):
        img = self._load_img(self.imgs[index])
        label = self.labels[os.path.basename(self.imgs[index])]
        
        if self.trans is not None:
            img = self.trans(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

def load_data(n_ex):
    loader = DataLoader(
        ImageNet(IMAGENET_TRANS),
        batch_size=n_ex, shuffle=False,
        num_workers=3
    )
    x_test, y_test = next(iter(loader))
    return np.array(x_test, dtype=np.float32), np.array(y_test)


class Model(ModelNP):
    def __init__(self, config):
        super(Model, self).__init__(config)
        model = eval(config['model'])(pretrained=True)
        self.model = DataParallel(model.cuda()).eval()

    def predict(self, x):
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        x = x.astype(np.float32)

        bs = self.batch_size
        bn = math.ceil(x.shape[0] / bs)
        
        logits_all = []
        with torch.no_grad():
            for i in range(bn):
                x_batch = x[i * bs: (i + 1) * bs]
                x_batch = torch.from_numpy(x_batch).cuda()
                logits = self.model(x_batch).cpu().numpy()
                logits_all.append(logits)
        logits_all = np.vstack(logits_all)
        return logits_all
