#! /bin/bash

python comparsion.py --attack sh --dataset mnist --epsilon 0.3 --log_freq 1 --log_root logs --n_ex 100
python comparsion.py --attack ash --dataset mnist --epsilon 0.3 --log_freq 1 --log_root logs --n_ex 100
# python comparsion.py --attack sh  --targeted --dataset mnist --epsilon 0.3 --log_freq 1 --log_root logs
# python comparsion.py --attack ash --targeted --dataset mnist --epsilon 0.3 --log_freq 1 --log_root logs