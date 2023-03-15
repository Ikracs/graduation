#! /bin/bash

python comparsion.py --attack ash --dataset cifar10 --epsilon 0.4 --log_freq 5 --log_root logs --n_ex 100
python comparsion.py --attack sh --dataset cifar10 --epsilon 0.4 --log_freq 5 --log_root logs --n_ex 100
# python comparsion.py --attack ash --targeted --dataset cifar10 --budget 1000  --epsilon 0.4 --log_freq 5 --log_root logs
# python comparsion.py --attack sh --targeted --dataset cifar10 --budget 10000 --epsilon 0.4 --log_freq 5 --log_root logs