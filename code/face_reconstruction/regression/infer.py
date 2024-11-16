import os
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf

import cv2
import torch
import numpy as np
import torchvision.transforms.functional as F

from face_reconstruction.regression.data import ImageReader
from face_reconstruction.regression.data import VideoReader
from face_reconstruction.regression.utils import prepare_pipeline

from utils import NoFaceDetectedException, NoFrameReturnedException
from utils import IMAGE_EXT, VIDEO_EXT

def preprocess(x):
    return F.to_tensor(x).unsqueeze(0)

def postprocess(x):
    x = (x * 255).clamp(0, 255).type(torch.uint8)
    return x[0].permute(1, 2, 0).cpu().numpy()

@torch.inference_mode()
def process_image(model, network, image, out_pth, device):
    render = model(network(preprocess(image).to(device)), render=True)['render_img']
    result = np.concatenate((image, postprocess(render)), axis=1)
    cv2.imwrite(out_pth, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

@torch.inference_mode()
def process_video(model, network, reader, out_pth, device):
    timestamp = str(datetime.now())[:-7].replace(' ', '-')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_size = (reader.img_size[0] * 2, reader.img_size[1])
    out = cv2.VideoWriter(out_pth, fourcc, reader.fps, out_size)

    try:
        for i in tqdm(range(reader.num_frames)):
            try:
                frame = reader.get()
            except Exception as exception:
                if isinstance(exception, NoFaceDetectedException):
                    with open(f'{timestamp}.error', 'a') as f:
                        f.write(f'No face detected in frame {i}.\n')
                elif isinstance(exception, NoFrameReturnedException):
                    with open(f'{timestamp}.error', 'a') as f:
                        f.write(f'Broken frame {i}.\n')
                else:
                    raise exception

                reader.relocate_face()

            render = model(network(preprocess(frame).to(device)), render=True)['render_img']
            out.write(cv2.cvtColor(np.concatenate((frame, postprocess(render)), axis=1), cv2.COLOR_RGB2BGR))

    finally:
        reader.release()
        out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Use Parameter Estimation Network for inference.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--config', type=str, default='face_reconstruction/regression/config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint load path')
    parser.add_argument('--input', type=str, required=True, help='input video path')
    parser.add_argument('--output', type=str, required=True, help='output video path')
    parser.add_argument('--smooth', type=int, default=1, help='smooth between n frames')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    model, network = prepare_pipeline(
        model=config.model,
        expand_ratio=config.dataset_params.expand_ratio,
        img_size=config.dataset_params.img_size,
        checkpoint=args.checkpoint,
        smooth=args.smooth
    )

    network.eval()

    model.to(args.device)
    network.to(args.device)

    print(f'Processing {args.input}...')
    if os.path.splitext(args.input)[-1] in IMAGE_EXT:
        reader = ImageReader(config.dataset_params.expand_ratio, config.dataset_params.img_size)
        process_image(model, network, reader.get(args.input), args.output, args.device)
    elif os.path.splitext(args.input)[-1] in VIDEO_EXT:
        reader = VideoReader(args.input, config.dataset_params.expand_ratio, config.dataset_params.img_size)
        process_video(model, network, reader, args.output, args.device)
    else:
        raise NotImplementedError(f"Unsupported input: {args.input}!")
