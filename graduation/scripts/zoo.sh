#! /bin/bash

python comparsion.py --attack zoo --dataset imagenet --model vgg16_bn --log_freq 100 --n_samples 50
python comparsion.py --attack zoo --dataset imagenet --model resnet50 --log_freq 100 --n_samples 50
python comparsion.py --attack zoo --dataset imagenet --model inception_v3 --log_freq 100 --n_samples 50

