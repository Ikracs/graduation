
import os
import random

import torch
import numpy as np
from torch.utils.data import Dataset

from utils import PORTRAIT_DATASETS


class A2MDataset(Dataset):
    def __init__(self, datasets, audio, seq_n, pre_k):
        self.datasets = PORTRAIT_DATASETS.keys() if 'all' in datasets else datasets

        self.audio = audio
        self.seq_n = seq_n
        self.pre_k = pre_k

        for dataset in self.datasets:
            with open(PORTRAIT_DATASETS[dataset]['meta_file'], 'r') as f:
                videos = sorted([l.strip() for l in f.readlines()])
        
        self.data = [os.path.join(PORTRAIT_DATASETS['data_root'], video) for video in videos]

        self.transform = lambda x: torch.tensor(x, dtype=torch.float32)

    def __getitem__(self, index):
        afeats = np.load(os.path.join(self.data[index], f'{self.audio}.npy'))
        motion = np.load(os.path.join(self.data[index], 'motion.npy'))
        
        N = min(afeats.shape[0], motion.shape[0])

        beg = random.randint(0, N - 1)
        pre = max(beg - self.pre_k, 0)
        end = min(beg + self.seq_n, N)
        padding_front = self.pre_k - (beg - pre)
        padding_back = self.seq_n - (end - beg)

        afeats = self.transform(np.pad(afeats[pre: end], ((padding_front, padding_back), (0, 0))))
        motion = self.transform(np.pad(motion[beg: end], ((0, padding_back), (0, 0))))

        padding_mask = torch.ones(self.seq_n + self.pre_k, dtype=torch.bool)
        padding_mask[padding_front: self.seq_n + self.pre_k - padding_back] = False

        return afeats, motion, padding_mask

    def __len__(self):
        return len(self.data)
