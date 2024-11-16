import argparse

from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf

import cv2
import torch
import numpy as np
import torch.nn.functional as F

from utils import paste_back as paste_back_to_source
from utils import ImageReader, AudioEncoder
from utils import MODEL_PATH

def preprocess(x):
    return F.to_tensor(x).unsqueeze(0)

def postprocess(x):
    x = (x * 255).clamp(0, 255).type(torch.uint8)
    return x[0].permute(1, 2, 0).cpu().numpy()


class Driver:
    def __init__(self, a2m_cfg, m2v_cfg, device):
        self.device = device

        self.audio = a2m_cfg.dataset_params.audio
        self.seq_n = a2m_cfg.train_params.seq_n
        self.pre_k = a2m_cfg.train_params.pre_k

        self.expand_ratio = m2v_cfg.dataset_params.expand_ratio
        self.img_size = m2v_cfg.dataset_params.img_size

        self.a2m_model = self.load_a2m_model(a2m_cfg.model_params)
        self.m2v_model = self.load_m2v_model(m2v_cfg.model_params.generator_params)

        self.a2m_model.eval()
        self.m2v_model.eval()

        self.a2m_model.to(device)
        self.m2v_model.to(device)

        self.image_reader = ImageReader(self.expand_ratio, self.img_size)
        self.audio_encoder = AudioEncoder(self.audio, MODEL_PATH[self.audio])

    def load_a2m_model(self, model_params, checkpoint):
        raise NotImplementedError

    def load_m2v_model(self, model_params, checkpoint):
        raise NotImplementedError

    def run(self, input_image, input_audio, output, paste_back=False):
        timestamp = str(datetime.now())[:-7].replace(' ', '-')

        aligned_image, image, bbox = self.image_reader.get(input_image)
        afeats = self.audio_encoder(input_audio)

        if paste_back:
            out_size = (image.shape[1] * 2, image.shape[0])
        else:
            out_size = (aligned_image.shape[1] * 2, aligned_image.shape[0])

        with torch.inference_mode():
            source = self.m2v_model.process_source(preprocess(aligned_image).to(self.device))

            motion = []
            for i in range(0, afeats.shape[0], self.seq_n - self.pre_k):
                cond = afeats
                # padding_mask = 
                # motion = self.a2m_model.ddim_sample([1, 80, 256], cond, padding_mask).squeeze(0)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, 25, out_size)

            results = [postprocess(self.m2v_model.driving(source, m)['prediction']) for m in motion]

            if paste_back:
                result = np.concatenate((image, paste_back_to_source(image, bbox, result)), axis=1)
            else:
                result = np.concatenate((aligned_image, result), axis=1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run audio2video driving model.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--a2m_cfg', type=str, default='driving/audio2video/portrait/config/default.yaml')
    parser.add_argument('--m2v_cfg', type=str, default='driving/video2video/portrait/config/default.yaml')
    parser.add_argument('--a2m_ckpt', type=str, default=None, help='audio2motion model checkpoint load path')
    parser.add_argument('--m2v_ckpt', type=str, default=None, help='video2video model checkpoint load path')
    parser.add_argument('--input_image', type=str, default=None, help='input image path')
    parser.add_argument('--input_audio', type=str, default=None, help='input audio path')
    parser.add_argument('--output', type=str, default=None, help='output video path')
    parser.add_argument('--paste_back', action='store_true', default=None, help='paste result back')

    args = parser.parse_args()

    a2m_cfg = OmegaConf.load(args.a2m_cfg)
    a2m_cfg.train_params.checkpoint = args.a2m_ckpt

    m2v_cfg = OmegaConf.load(args.m2v_cfg)
    m2v_cfg.train_params.checkpoint = args.m2v_ckpt

    driver = Driver(a2m_cfg, m2v_cfg, args.device)
    driver.run()
