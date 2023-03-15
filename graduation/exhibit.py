
import os
import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import utils
import imagenet
from attacks import _ash_single_img
from imagenet_labels import label_to_name

cfg = {
    'loss': 'cw',
    'model': 'resnet50',
    'targeted': False,
    'budget': 10000,
    'epsilon': 0.05,
    'n_ex': 5,
    'batch_size': 2,
    'save_root': 'figs/exhibit'
}

def plot_bar(logits, save_pth):
    probs = torch.tensor(logits).softmax(dim=-1)
    score, label = probs.topk(k=5, dim=-1)
    label = [label_to_name(l) for l in label]
    pdf = PdfPages(save_pth)
    plt.barh(label[:: -1], score.flip(-1))
    f = plt.gca()
    f.axes.get_xaxis().set_visible(False)
    plt.ylabel(fonesize=20)
    plt.tight_layout(); pdf.savefig()
    plt.close(); pdf.close()

if __name__ == '__main__':
    model = imagenet.Model(cfg)
    imgs, labels = imagenet.load_data(cfg['n_ex'])
    correct = model.correct(model.predict(imgs), labels)
    imgs, labels = imgs[correct], labels[correct]
    for count, (img, label) in enumerate(zip(imgs, labels)):
        img, label = img[np.newaxis, ...], label[np.newaxis, ...]
        done, adv_img, queries = _ash_single_img(model, img, label, cfg)
        logits, adv_logits = model.predict(img), model.predict(adv_img)
        adv_label = adv_logits.argmax(axis=-1).item()
        if done:
            bar_name = 'bar#{}.pdf'.format(count)
            plot_bar(logits.squeeze(0), os.path.join(cfg['save_root'], bar_name))
            bar_name = 'adv_bar#{}.pdf'.format(count)
            plot_bar(adv_logits.squeeze(0), os.path.join(cfg['save_root'], bar_name))
            
            save_name = '{}.jpg'.format(count)
            utils.save_img(img.squeeze(0), os.path.join(cfg['save_root'], save_name))
            save_name = 'adv_{}.jpg'.format(count)
            utils.save_img(adv_img.squeeze(0), os.path.join(cfg['save_root'], save_name))
