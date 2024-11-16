import os
import json
import random
import multiprocessing
from tqdm import tqdm

import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from utils import iou, crop_image, expand_bbox, seg2msk
from utils import FACE_DATASETS


class ToPILImage(transforms.ToPILImage):
    def __call__(self, input):
        output = input.copy()
        output['src_img'] = F.to_pil_image(output['src_img'])
        output['dri_img'] = F.to_pil_image(output['dri_img'])
        output['cid_img'] = F.to_pil_image(output['cid_img'])
        output['src_msk'] = F.to_pil_image(output['src_msk'])
        output['dri_msk'] = F.to_pil_image(output['dri_msk'])
        output['cid_msk'] = F.to_pil_image(output['cid_msk'])
        return output


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, input):
        output = input.copy()
        if random.random() < self.p:
            output['src_img'] = F.hflip(output['src_img'])
            output['dri_img'] = F.hflip(output['dri_img'])
            output['src_msk'] = F.hflip(output['src_msk'])
            output['dri_msk'] = F.hflip(output['dri_msk'])
        return output


class ColorJitter(transforms.ColorJitter):
    def forward(self, input):
        output = input.copy()
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = \
            self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                output['src_img'] = F.adjust_brightness(output['src_img'], brightness_factor)
                output['dri_img'] = F.adjust_brightness(output['dri_img'], brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                output['src_img'] = F.adjust_contrast(output['src_img'], contrast_factor)
                output['dri_img'] = F.adjust_contrast(output['dri_img'], contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                output['src_img'] = F.adjust_saturation(output['src_img'], saturation_factor)
                output['dri_img'] = F.adjust_saturation(output['dri_img'], saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                output['src_img'] = F.adjust_hue(output['src_img'], hue_factor)
                output['dri_img'] = F.adjust_hue(output['dri_img'], hue_factor)

        return output


class ToTensor(transforms.ToTensor):
    def __call__(self, input):
        output = input.copy()
        output['src_img'] = F.to_tensor(output['src_img'])
        output['dri_img'] = F.to_tensor(output['dri_img'])
        output['cid_img'] = F.to_tensor(output['cid_img'])
        output['src_msk'] = F.to_tensor(output['src_msk'])
        output['dri_msk'] = F.to_tensor(output['dri_msk'])
        output['cid_msk'] = F.to_tensor(output['cid_msk'])
        return output


class FramesDataset(Dataset):
    def __init__(self, datasets, sampler, expand_ratio=1.8, img_size=256, num_workers=4):
        self.datasets = FACE_DATASETS.keys() if 'all' in datasets else datasets

        self.sampler = sampler
        self.expand_ratio = expand_ratio
        self.img_size = (img_size, img_size)

        for dataset in self.datasets:
            with open(FACE_DATASETS[dataset]['meta_file'], 'r') as f:
                videos = sorted([l.strip() for l in f.readlines()])

        pool = multiprocessing.Pool(num_workers)
        pbar = tqdm(videos, dynamic_ncols=True, desc='Loading videos')
        self.data = pool.map(self.prepare_data, pbar)

        self.transform = transforms.Compose([
            ToPILImage(),
            RandomHorizontalFlip(p=0.5),
            ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            ToTensor()
        ])

    def prepare_data(self, video):
        video_pth = os.path.join(FACE_DATASETS['data_root'], video)
        with open(os.path.join(video_pth, f'{video}.txt', 'r')) as f:
            frames = sorted([l.strip for l in f.readlines()])
        return {'video_pth': video_pth, 'frames': frames}

    def __getitem__(self, index):
        video_pth = self.data[index]['video_pth']
        frames = self.data[index]['frames']

        while True:
            src_frame, dri_frame = self.sampler(frames)

            with open(os.path.join(video_pth, os.path.splitext(src_frame)[0] + '.json'), 'r') as f:
                src_face_bbox = json.load(f)['face_bbox']

            with open(os.path.join(video_pth, os.path.splitext(dri_frame)[0] + '.json'), 'r') as f:
                dri_face_bbox = json.load(f)['face_bbox']

            if iou(src_face_bbox, dri_face_bbox) < 0.8: continue

            face_bbox = expand_bbox(src_face_bbox, self.expand_ratio)

            src_img = cv2.cvtColor(cv2.imread(os.path.join(video_pth, src_frame)), cv2.COLOR_BGR2RGB)
            dri_img = cv2.cvtColor(cv2.imread(os.path.join(video_pth, dri_frame)), cv2.COLOR_BGR2RGB)

            src_seg = np.load(os.path.join(video_pth, os.path.splitext(src_frame)[0] + '_seg.npy'))
            dri_seg = np.load(os.path.join(video_pth, os.path.splitext(dri_frame)[0] + '_seg.npy'))

            break

        cid_video = random.randint(len(self))
        cid_video_pth = self.data[cid_video]['video_pth']
        cid_frame = self.data[cid_video]['frames']

        with open(os.path.join(cid_video_pth, os.path.splitext(cid_frame)[0] + '.json'), 'r') as f:
            cid_face_bbox = expand_bbox(json.load(f)['face_bbox'], self.expand_ratio)

        cid_img = cv2.cvtColor(cv2.imread(os.path.join(cid_video_pth, cid_frame)), cv2.COLOR_BGR2RGB)
        cid_seg = np.load(os.path.join(cid_video_pth, os.path.splitext(cid_frame)[0] + '_seg.npy'))

        data_item = {}

        data_item['src_img'] = cv2.resize(crop_image(src_img, face_bbox), self.img_size)
        data_item['src_msk'] = cv2.resize(crop_image(seg2msk(src_seg), face_bbox), self.img_size)

        data_item['dri_img'] = cv2.resize(crop_image(dri_img, face_bbox), self.img_size)
        data_item['dri_msk'] = cv2.resize(crop_image(seg2msk(dri_seg), face_bbox), self.img_size)

        data_item['cid_img'] = cv2.resize(crop_image(cid_img, cid_face_bbox), self.img_size)
        data_item['cid_msk'] = cv2.resize(crop_image(seg2msk(cid_seg), cid_face_bbox), self.img_size)

        return self.transform(data_item)

    def __len__(self):
        return len(self.data)
