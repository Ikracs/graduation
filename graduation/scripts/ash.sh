#! /bin/bash

python comparsion.py --attack ash --dataset imagenet --model resnet50 --log_freq 50 --log_root logs
python comparsion.py --attack ash --dataset imagenet --model vgg16_bn --log_freq 50 --log_root logs
python comparsion.py --attack ash --dataset imagenet --model inception_v3 --log_freq 50 --log_root logs