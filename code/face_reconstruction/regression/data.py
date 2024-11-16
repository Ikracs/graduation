import os
import json
import random
import multiprocessing
from tqdm import tqdm
from collections import Iterable

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from utils import crop_image
from utils import FaceDetector
from utils import NoFrameReturnedException

from utils import FACE_DATASETS, FACE_DATA_ROOT
from utils import FACE_META_ROOT, FACE_META_FILE

def expand_bbox(bbox, ratio):
    sx, sy, ex, ey = bbox
    size = (ex - sx + ey - sy) / 2 * ratio
    padding_x = size - (ex - sx)
    padding_y = size - (ey - sy)
    sx, ex = round(sx - padding_x / 2), round(ex + padding_x / 2)
    sy, ey = round(sy - padding_y / 2), round(ey + padding_y / 2)
    return [sx, sy, ex, ey]


class FramesDataset(Dataset):

    def __init__(self, datasets, num_frames=1, expand_ratio=1.8, img_size=256, lmk_type='68', num_workers=4):
        self.datasets = FACE_DATASETS if 'all' in datasets else datasets

        self.num_frames = num_frames
        self.expand_ratio = expand_ratio
        self.img_size = (img_size, img_size)
        self.lmk_type = lmk_type

        vids = []
        for dataset in self.datasets:
            with open(os.path.join(FACE_META_ROOT, FACE_META_FILE[dataset]), 'r') as f:
                vids.extend(f.readlines())

        pool = multiprocessing.Pool(num_workers)
        pbar = tqdm(vids, dynamic_ncols=True, desc='Loading videos')
        self.data = pool.map(self.prepare_data, pbar)

        self.img_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.ToTensor()
        ])
        self.lmk_transform = lambda x: torch.tensor(x, dtype=torch.float32)

    def prepare_data(self, vid):
        video_pth = os.path.join(FACE_DATA_ROOT, vid.strip())
        with open(os.path.join(video_pth, "wav2lip.json"), 'r') as f:
            info = json.load(f)
        return {
            "video_pth": video_pth,
            "frame_pths": sorted(
                [os.path.join(FACE_DATA_ROOT, video_pth, fid) for fid in info["paths"]["full"]]
            )
        }

    def expand_bbox(self, bbox, ratio):
        sx, sy, ex, ey = bbox
        size = (ex - sx + ey - sy) / 2 * ratio
        padding_x = size - (ex - sx)
        padding_y = size - (ey - sy)
        padding_l = random.randint(int(padding_x / 3), int(2 * padding_x / 3))
        padding_u = random.randint(int(padding_y / 3), int(2 * padding_y / 3))
        padding_r = padding_x - padding_l
        padding_b = padding_y - padding_u
        sx, ex = round(sx - padding_l), round(ex + padding_r)
        sy, ey = round(sy - padding_u), round(ey + padding_b)
        return [sx, sy, ex, ey]

    def read(self, pth):
        frame = cv2.cvtColor(cv2.imread(pth), cv2.COLOR_RGB2BGR)

        with open(os.path.splitext(pth)[0] + '.json', 'r') as f:
            frame_info = json.load(f)

        bbox = self.expand_bbox(frame_info['face_box'], self.expand_ratio)

        lmk = np.array(frame_info['landmark_' + self.lmk_type], dtype=np.float32)
        lmk[:, 0] = (lmk[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * self.img_size[0]
        lmk[:, 1] = (lmk[:, 1] - bbox[1]) / (bbox[3] - bbox[1]) * self.img_size[1]

        aligned = cv2.resize(crop_image(frame, bbox), self.img_size)

        return aligned, lmk

    def __getitem__(self, index):
        frame_pths = self.data[index]['frame_pths']

        frames, lms = [], []
        for frame_pth in np.random.permutation(frame_pths):
            frame, lmk = self.read(frame_pth)
            frames.append(self.img_transform(frame))
            lms.append(self.lmk_transform(lmk))
        return torch.stack(frames), torch.stack(lms)

    def __len__(self):
        return len(self.data)


class VideoReader:
    def __init__(self, video_pth, expand_ratio=1.8, frame_size=256):
        self.cap = cv2.VideoCapture(video_pth)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.expand_ratio = expand_ratio
        self.img_size = [frame_size] * 2 if not isinstance(frame_size, Iterable) else frame_size

        self.detector = FaceDetector()

    def relocate_face(self):
        if hasattr(self, 'bbox'):
            delattr(self, 'bbox')

    def get(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not hasattr(self, 'bbox'):
                self.bbox = expand_bbox(self.detector(frame), self.expand_ratio)
            return cv2.resize(crop_image(frame, self.bbox), self.img_size)
        else:
            raise NoFrameReturnedException()

    def release(self):
        self.cap.release()


class ImageReader:
    def __init__(self, expand_ratio=1.8, image_size=256):
        self.expand_ratio = expand_ratio
        self.img_size = [image_size] * 2 if not isinstance(image_size, Iterable) else image_size

        self.detector = FaceDetector()

    def get(self, image_pth):
        image = cv2.cvtColor(cv2.imread(image_pth), cv2.COLOR_BGR2RGB)
        bbox = expand_bbox(self.detector(image), self.expand_ratio)
        return cv2.resize(crop_image(image, bbox), self.img_size)
