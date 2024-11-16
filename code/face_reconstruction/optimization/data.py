import os
import json

import cv2
from collections import Iterable

from utils import FaceDetector
from utils import NoFaceDetectedException, NoFrameReturnedException
from utils import crop_image

def expand_bbox(bbox, ratio):
    sx, sy, ex, ey = bbox
    size = (ex - sx + ey - sy) / 2 * ratio
    padding_x = size - (ex - sx)
    padding_y = size - (ey - sy)
    sx, ex = round(sx - padding_x / 2), round(ex + padding_x / 2)
    sy, ey = round(sy - padding_y / 2), round(ey + padding_y / 2)
    return [sx, sy, ex, ey]


class VideoReader:
    def __init__(self, video_pth, expand_ratio=1.8, frame_size=256):
        self.cap = cv2.VideoCapture(video_pth)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.expand_ratio = expand_ratio
        self.img_size = [frame_size] * 2 if not isinstance(frame_size, Iterable) else frame_size

        self.detector = FaceDetector()

    def get_frame_ptr(self):
        return self.cap.get(cv2.CAP_PROP_POS_FRAMES)

    def set_frame_ptr(self, frame_number):
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

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


class FrameFolder:
    def __init__(self, video_pth, expand_ratio=1.8, frame_size=256):
        self.data = self.prepare_data(video_pth)

        self.current = 0
        self.num_frames = len(self.data)

        self.expand_ratio = expand_ratio
        self.img_size = [frame_size] * 2 if not isinstance(frame_size, Iterable) else frame_size

    def prepare_data(self, path):
        with open(os.path.join(path, "wav2lip.json"), 'r') as f:
            info = json.load(f)
        return [os.path.join(path, i) for i in sorted(info["paths"]["full"])]

    def get_frame_ptr(self):
        return self.current

    def set_frame_ptr(self, frame_number):
        self.current = frame_number
        return self.current

    def relocate_face(self):
        if hasattr(self, 'bbox'):
            delattr(self, 'bbox')

    def get(self):
        frame_pth = self.data[self.current]

        frame = cv2.imread(frame_pth)
        if frame is None: raise NoFaceDetectedException()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not hasattr(self, 'bbox'):
            with open(os.path.splitext(frame_pth)[0] + '.json', 'r') as f:
                frame_info = json.load(f)
            self.bbox = expand_bbox(frame_info['face_box'], self.expand_ratio)

        frame = cv2.resize(crop_image(frame, self.bbox), self.img_size)

        self.current += 1
        return frame
