
import os
import torch
import numpy as np
from tqdm import tqdm

import utils
import imagenet
from attacks import _ash_single_img

MODEL_LIST = ['resnet50', 'vgg16_bn', 'inception_v3', 'densenet161', 'resnet152']

cfg = {
    'loss': 'cw',
    'targeted': False,
    'budget': 1000,
    'epsilon': 0.05,
    'n_ex': 500,
    'batch_size': 10,
}

def accelerated_sign_hunter(model, imgs, labels, config):
    adv_imgs, adv_labels = [], []
    for count, (img, label) in enumerate(zip(imgs, labels)):
        img, label = img[np.newaxis, ...], label[np.newaxis, ...]
        adv_img = _ash_single_img(model, img, label, config)
        adv_label = model.predict(adv_img).argmax(axis=-1).item()
        adv_imgs.append(adv_img); adv_labels.append(adv_label)
    adv_imgs, adv_labels = np.concatenate(adv_imgs), np.array(adv_labels)
    return adv_imgs, adv_labels

def _ash_single_img(model, img, label, config):
    budget    = config['budget']
    epsilon   = config['epsilon']
    
    min_v, max_v = 0, 1 if img.max() <= 1 else 255

    dim = np.prod(img.shape)
    sign_bits = np.ones(dim)

    delta = epsilon * sign_bits.reshape(img.shape)
    perturbed = np.clip(img + delta, min_v, max_v)
    
    logits = model.predict(perturbed)
    loss = model.loss(logits, label)

    node_i, tree_h = 0, 0
    regions = [[0.0, [0, dim]]]

    def _divide(regions):
        regions_new = []
        for region in regions:
            start, end = region[1]
            mid = start + (end - start) // 2
            regions_new.append([region[0], [start, mid]])
            regions_new.append([region[0], [mid, end]])
        return regions_new
    
    for _ in range(budget):
        need_query = True
        if node_i % 2 == 1:
            regions[node_i][0] -= regions[node_i - 1][0]
            need_query = regions[node_i][0] < 0
        
        if need_query:
            sign_bits_new = sign_bits.copy()
            start, end = regions[node_i][1]
            sign_bits_new[start: end] *= -1

            if start != end:
                delta = epsilon * sign_bits_new.reshape(img.shape)
                perturbed = np.clip(img + delta, min_v, max_v)

                logits = model.predict(perturbed)
                loss_new = model.loss(logits, label)

                regions[node_i][0] = (loss - loss_new).item()

                if loss_new > loss:
                    loss, sign_bits = loss_new, sign_bits_new
            else:
                regions[node_i][0] = float('inf')

        node_i += 1
        if node_i == 2 ** tree_h:
            node_i = 0; tree_h += 1
            
            regions = [[abs(r[0]), r[1]] for r in _divide(regions)]
            regions = sorted(regions, key=lambda r: r[0], reverse=False)
    
    delta = epsilon * sign_bits.reshape(img.shape)
    adv_img = np.clip(img + delta, min_v, max_v)
    return adv_img

if __name__ == '__main__':
    img, label = imagenet.load_data(cfg['n_ex'])
    for smodel_type in MODEL_LIST:
        cfg['model'] = smodel_type
        print('Surrogate Model: {:s}'.format(smodel_type))
        print('Attack Success Rate on Victim Model ', end='')
        smodel = imagenet.Model(cfg)
        adv_img, _ = accelerated_sign_hunter(smodel, img, label, cfg)
        for vmodel_type in MODEL_LIST:
            cfg['model'] = vmodel_type
            vmodel = imagenet.Model(cfg)
            logits = vmodel.predict(adv_img)
            asr = 1.0 - vmodel.correct(logits, label).sum() / img.shape[0]
            print('{:s}: {:.4f}, '.format(vmodel_type, asr), end='')
        print('')
