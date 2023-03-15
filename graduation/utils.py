import os
import numpy as np
from PIL import Image
from imagenet_labels import label_to_name

EXT = '.jpg'

class Logger(object):
    def __init__(self):
        self.info = []

    def log_info(self, *args):
        self.info.append(list(args))

    def save(self, logging_pth):
        np.save(logging_pth, self.info)

def save_imgs(imgs, save_names, save_root):
    count = {}
    for img, save_name in zip(imgs, save_names):
        if save_name not in count.keys():
            count[save_name] = 0
            save_name += EXT
        else:
            count[save_name] += 1
            save_name += '#{}'.format(count[save_name]) + EXT
        save_img(img, os.path.join(save_root, save_name))

def save_img(array, save_pth):
    img = array.transpose(1, 2, 0)
    img = np.uint8(255 * img)
    img = Image.fromarray(img)
    img.save(save_pth)

def random_pseudo_label(label, n_cls=1000):
    target = np.zeros_like(label)
    for i in range(label.shape[0]):
        classes = list(range(n_cls))
        classes.remove(label[i])
        target[i] = np.random.choice(classes)
    return target

def down_sample(img, src_res, dst_res):
    pass

def up_sample(img, src_res, dst_res):
    pass
