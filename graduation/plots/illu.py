import os
import sys
import cv2
import pdb
import numpy as np
from PIL import Image
from scipy import stats
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

IMAGENET_ROOT = '/mnt/datadev_2/std/lisiyuan/imagenet/val'

pw, cw = 0.4, 0.6
n_seg = 64
BLACK = (0, 0, 0)
RED = (255, 0, 0)

path = os.path.join(IMAGENET_ROOT, 'ILSVRC2012_val_00000002.JPEG')
with open(path, 'rb') as f:
    img = np.array(Image.open(f).convert('RGB'))
    img = cv2.resize(img, (256, 256))
    h, w, c = img.shape
    patch_21 = img[0: (h // 2), :, :]
    patch_22 = img[(h // 2):, :, :]
    patch_41 = img[0: (h // 2), 0: (w // 2), :]
    patch_42 = img[0: (h // 2), (w // 2):, :]
    patch_43 = img[(h // 2):, 0: (w // 2), :]
    patch_44 = img[(h // 2):, (w // 2):, :]
    img_1 = np.zeros_like(img)
    img_3 = np.zeros_like(img)
    img_1[(h // 2):, :, :] = patch_21
    img_1[0: (h // 2), :, :] = patch_22
    cv2.line(img_1, (0, h // 2), (w, h // 2), BLACK, 1, 8)
    img_2 = np.copy(img_1)
    cv2.line(img_2, (w // 2, 0), (w // 2, h), BLACK, 1, 8)
    cmap_1 = cv2.applyColorMap(20 * np.ones_like(patch_41, dtype=np.uint8), cv2.COLORMAP_JET)
    cmap_2 = cv2.applyColorMap(40 * np.ones_like(patch_41, dtype=np.uint8), cv2.COLORMAP_JET)
    cmap_3 = cv2.applyColorMap(60 * np.ones_like(patch_41, dtype=np.uint8), cv2.COLORMAP_JET)
    cmap_4 = cv2.applyColorMap(80 * np.ones_like(patch_41, dtype=np.uint8), cv2.COLORMAP_JET)
    img_3[0: (h // 2), 0: (w // 2), :] = cv2.addWeighted(patch_41, pw, cmap_1, cw, 0)
    img_3[0: (h // 2), (w // 2):, :] = cv2.addWeighted(patch_43, pw, cmap_2, cw, 0)
    img_3[(h // 2):, 0: (w // 2), :] = cv2.addWeighted(patch_44, pw, cmap_3, cw, 0)
    img_3[(h // 2):, (w // 2):, :] = cv2.addWeighted(patch_42, pw, cmap_4, cw, 0)
    cv2.line(img_3, (0, h // 2), (w, h // 2), BLACK, 1, 8)
    cv2.line(img_3, (w // 2, 0), (w // 2, h), BLACK, 1, 8)
    img_4 = np.copy(img_1)
    cv2.line(img_4, (0, h // 2), (w, h), RED, 2, 8)
    cv2.line(img_4, (w, h // 2), (0, h), RED, 2, 8)
    sign_bits = np.ones(c * h * w)
    seg_len = (c * h * w) // n_seg
    for i in range(n_seg):
        factor = np.random.choice([-1, 1], p=[0.5, 0.5])
        sign_bits[i * seg_len: (i + 1) * seg_len] *= factor
    delta = 6 * sign_bits.reshape(c, h, w).transpose(1, 2, 0)
    img_5 = np.clip(img + delta, 0, 255)
    img_5 = img_5.astype(np.uint8)
    Image.fromarray(img).save('figs/illu_0.jpg')
    Image.fromarray(img_1).save('figs/illu_1.jpg')
    Image.fromarray(img_2).save('figs/illu_2.jpg')
    Image.fromarray(img_3).save('figs/illu_3.jpg')
    Image.fromarray(img_4).save('figs/illu_4.jpg')
    Image.fromarray(img_5).save('figs/illu_5.jpg')

