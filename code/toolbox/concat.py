import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm

from utils import IMAGE_EXT, VIDEO_EXT


class DataItem:
    def __init__(self, pth):
        if os.path.splitext(pth)[-1] in IMAGE_EXT:
            self.dtype = 'image'
            self.ditem = cv2.imread(pth)
            self.H = self.ditem.shape[0]
            self.W = self.ditem.shape[1]
            self.N, self.fps = 0, 0
        elif os.path.splitext(pth)[-1] in VIDEO_EXT:
            self.dtype = 'video'
            self.ditem = cv2.VideoCapture(pth)
            self.H = int(self.ditem.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.W = int(self.ditem.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.N = int(self.ditem.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.ditem.get(cv2.CAP_PROP_FPS))
        else:
            raise NotImplementedError(f'Unsupported extension of {pth}')

    def read(self):
        return (True, self.ditem) if self.dtype == 'image' else self.ditem.read()

    def release(self):
        if self.dtype == 'video':
            self.ditem.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Concatenating two videos. One of the inputs can be image.")
    parser.add_argument("input1", type=str, help="path of the first image/video")
    parser.add_argument("input2", type=str, help="path of the second image/video")
    parser.add_argument("video_pth", type=str, help="path of the output video")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--H", action='store_true', help="concatenate horizontally")
    group.add_argument("--V", action='store_true', help="concatenate vertically")

    args = parser.parse_args()

    args.H = not args.V

    item1 = DataItem(args.input1)
    item2 = DataItem(args.input2)

    assert(item1.fps or item2.fps), "at least one of the inputs is a video!"
    assert(not args.H or (args.H and (item1.H == item2.H))), "height of inputs should be matched!"
    assert(not args.V or (args.V and (item1.W == item2.W))), "width of inputs should be matched!"

    H = item1.H if args.H else item1.H + item2.H
    W = item1.W if args.V else item1.W + item2.W

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(args.video_pth, fourcc, item1.fps or item2.fps, (W, H))

    try:
        for _ in tqdm(range(max(item1.N, item2.N))):
            ret1, frame1 = item1.read()
            ret2, frame2 = item2.read()
            if ret1 and ret2:
                video.write(np.concatenate((frame1, frame2), axis=1 if args.H else 0))
    except Exception as exception:
        raise exception
    finally:
        item1.release()
        item2.release()
        video.release()
