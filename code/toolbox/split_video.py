import os
import cv2
import argparse

from tqdm import tqdm
from datetime import datetime

from utils import VIDEO_EXT

def process(video_pth, out_dir, timestamp):
    os.makedirs(out_dir, exist_ok=True)

    reader = cv2.VideoCapture(video_pth)

    try:
        desc = f'Processing {video_pth}'
        for i in tqdm(range(int(reader.get(cv2.CAP_PROP_FRAME_COUNT))), desc=desc):
            ret, frame = reader.read()

            if not ret:
                with open(f'{timestamp}.error', 'a') as f:
                    f.write(f'[{video_pth}] No face detected in frame {i}.\n')
                continue

            cv2.imwrite(os.path.join(out_dir, f'{str(i).zfill(6)}.png'), frame)
    finally:
        reader.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Split video into frames.")
    parser.add_argument('input', type=str, default=None, help='input path')
    parser.add_argument('output', type=str, default=None, help='output path')

    args = parser.parse_args()

    timestamp = str(datetime.now())[:-7].replace(' ', '-')

    if os.path.isdir(args.input):
        os.makedirs(args.output, exist_ok=True)
        in_pths = [os.path.join(args.input, f) for f in os.listdir(args.input)]
        out_dirs = [os.path.join(args.output, os.path.splitext(f)[0]) for f in os.listdir(args.input)]
    else:
        in_pths, out_dirs = [args.input], [args.output]

    for in_pth, out_dir in zip(in_pths, out_dirs):
        if os.path.splitext(in_pth)[-1].lower() in VIDEO_EXT:
            try:
                process(in_pth, out_dir, timestamp)
            except Exception as exception:
                with open(f'{timestamp}.error', 'a') as f:
                    f.write(f'[{in_pth}] {str(exception)}.\n')
                continue
