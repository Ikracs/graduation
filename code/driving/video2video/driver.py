from tqdm import tqdm
from datetime import datetime

import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F

from utils import paste_back as paste_back_to_source
from utils import ImageReader, VideoReader
from utils import NoFaceDetectedException, NoFrameReturnedException

def preprocess(x):
    return F.to_tensor(x).unsqueeze(0)

def postprocess(x):
    x = (x * 255).clamp(0, 255).type(torch.uint8)
    return x[0].permute(1, 2, 0).cpu().numpy()


class Driver:
    def __init__(self, cfg, device):
        self.device = device

        self.expand_ratio = cfg.dataset_params.expand_ratio
        self.img_size = cfg.dataset_params.img_size

        self.generator = self.load_model(cfg.model_params.generator_params, cfg.train_params.checkpoint)

        self.generator.eval()
        self.generator.to(device)

        self.image_reader = ImageReader(self.expand_ratio, self.img_size)

    def load_model(self, model_params, checkpoint):
        raise NotImplementedError

    def process(self, source, driving, **kwargs):
        raise NotImplementedError

    def run(self, input_image, input_video, output, paste_back=False):
        timestamp = str(datetime.now())[:-7].replace(' ', '-')

        video_reader = VideoReader(input_video, self.expand_ratio, self.img_size)

        try:
            aligned_image, image, bbox = self.image_reader.get(input_image)

            if paste_back:
                out_size = (image.shape[1] * 3, image.shape[0])
            else:
                out_size = (aligned_image.shape[1] * 3, aligned_image.shape[0])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output, fourcc, video_reader.fps, out_size)

            with torch.inference_mode():
                source = self.generator.process_source(preprocess(aligned_image).to(self.device))

                for i in tqdm(range(len(video_reader))):
                    try:
                        aligned_frame, frame, _ = video_reader.get()
                    except Exception as exception:
                        if isinstance(exception, NoFaceDetectedException):
                            with open(f'{timestamp}.error', 'a') as f:
                                f.write(f'No face detected in frame {i}.\n')
                        elif isinstance(exception, NoFrameReturnedException):
                            with open(f'{timestamp}.error', 'a') as f:
                                f.write(f'Broken frame {i}.\n')
                        else:
                            raise exception

                        video_reader.relocate_face()

                    driving = self.generator.process_driving(preprocess(aligned_frame).to(self.device))
                    result = postprocess(self.process(source, driving, paste_back)['prediction'])

                    if paste_back:
                        result = np.concatenate((image, frame, paste_back_to_source(image, bbox, result)), axis=1)
                    else:
                        result = np.concatenate((aligned_image, aligned_frame, result), axis=1)

                    out.write(cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

        finally:
            video_reader.release()
