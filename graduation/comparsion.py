import os
import argparse
from datetime import datetime

import torch
import numpy as np

import utils
import mnist
import cifar10
import imagenet

from attacks import zeroth_order_opt as zoo
from attacks import natural_evolution_strategy as nes
from attacks import sign_hunter as sh
from attacks import accelerated_sign_hunter as ash

SUPPORT_ATTACKS = ['zoo', 'nes', 'sh', 'ash']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Black-Box attack against image classification')
    parser.add_argument('--attack', type=str, default='ash', choices=SUPPORT_ATTACKS)
    parser.add_argument('--loss', type=str, default='cw', choices=['cw', 'ce'])
    parser.add_argument('--dataset', type=str, default='imagenet', choices=['mnist', 'cifar10', 'imagenet'])
    parser.add_argument('--model', type=str, default='resnet50', help='victim model')
    parser.add_argument('--targeted', action='store_true', help='targeted attack or not')
    parser.add_argument('--budget', type=int, default=10000, help='query budget for black-box attack')
    parser.add_argument('--epsilon', type=float, default=0.05, help='maximum linf norm of perturbation')
    parser.add_argument('--n_ex', type=int, default=1000, help='total num of imgs to attack')
    parser.add_argument('--batch_size', type=int, default=32, help='num of imgs attacked in an iter')
    parser.add_argument('--gpu', type=str, default='0', help='Available GPU id')
    parser.add_argument('--log_freq', type=int, default=50, help='log frequency')
    parser.add_argument('--log_root', type=str, default=None, help='log root of attacking')
    parser.add_argument('--save_root', type=str, default=None, help='save root of adv imgs')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')

    timestamp = str(datetime.now())[:-7]
    print('Running start at ' + timestamp)
    
    cfg = vars(parser.parse_args())
    for key in cfg.keys():
        print(key + ' ' + str(cfg[key]))

    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg['gpu']

    vmodel = eval(cfg['dataset']).Model(cfg)
    
    img, label = eval(cfg['dataset']).load_data(cfg['n_ex'])
    
    logits = vmodel.predict(img)
    correct = vmodel.correct(logits, label)
    img, label = img[correct], label[correct]
    
    accuracy = correct.sum() / cfg['n_ex']
    print('Clean accuracy: {:.4f}'.format(accuracy))
    
    if cfg['targeted']:
        n_cls = eval(cfg['dataset']).CLASSES
        label = utils.random_pseudo_label(label, n_cls)

    eval(cfg['attack'])(vmodel, img, label, cfg, timestamp)
