#! /bin/bash

python comparsion.py --attack nes --dataset imagenet --model vgg16_bn --log_freq 100 --sigma 1e-3 --n_samples 25
python comparsion.py --attack nes --dataset imagenet --model resnet50 --log_freq 100 --sigma 1e-3 --n_samples 25
python comparsion.py --attack nes --dataset imagenet --model inception_v3 --log_freq 100 --sigma 1e-3 --n_samples 25
