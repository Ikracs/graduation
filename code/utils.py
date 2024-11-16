import time
import random

import cv2
import torch
import open3d
import librosa
import numpy as np
import face_alignment
import mediapipe as mp
from functools import partial
from insightface.app import FaceAnalysis
from torchvision.transforms import functional as F
from torchvision.utils import draw_segmentation_masks, draw_keypoints

from transformers import Wav2Vec2Processor, Wav2Vec2Model
from transformers import Wav2Vec2FeatureExtractor, HubertModel


IMAGE_EXT = ['.jpg', '.jpeg', '.png']
VIDEO_EXT = ['.mp4', '.mp4v', '.gif']

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LBROW = [17, 18, 19, 20, 21]
RBROW = [22, 23, 24, 25, 26]
NOSE = [27, 28, 29, 30, 31, 32, 33, 34, 35]
LEYE = [42, 43, 44, 45, 46, 47]
REYE = [36, 37, 38, 39, 40, 41]
LIP_INNER = [60, 61, 62, 63, 64, 65, 66, 67]
LIP = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
FACE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

MP_RBROW_LOWER = [46, 53, 52, 65, 55]
MP_LBROW_LOWER = [285, 295, 282, 283, 276]
MP_LBROW = [276, 283, 282, 295, 300, 293, 334, 296, 336, 285]
MP_RBROW = [46, 53, 52, 65, 70, 63, 105, 66, 55, 107]
MP_NOSE = [4, 6, 98, 129, 197, 327, 358]
MP_LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 263, 466, 388, 387, 386, 385, 384, 398]
MP_REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
MP_LIRIS = [473, 474, 475, 476, 477]
MP_RIRIS = [468, 469, 470, 471, 472]
MP_LIP_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
MP_LIP = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
MP_FACE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

CONNECTIVITY = [
    # FACE
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
    [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16],
    # LBROW
    [17, 18], [18, 19], [19, 20], [20, 21],
    # RBROW
    [22, 23], [23, 24], [24, 25], [25, 26],
    # NOSE_UPPER
    [27, 28], [28, 29], [29, 30],
    # NOSE_LOWWER
    [30, 31], [31, 32], [32, 33], [33, 34], [34, 35], [30, 35],
    # LEYE
    [36, 37], [37, 38], [38, 39], [39, 40], [40, 41], [41, 36],
    # REYE
    [42, 43], [43, 44], [44, 45], [45, 46], [46, 47], [47, 42],
    # LIP_OUTTER
    [48, 49], [49, 50], [50, 51], [51, 52], [52, 53], [53, 54],
    [54, 55], [55, 56], [56, 57], [57, 58], [58, 59], [59, 48],
    # LIP_INNER
    [60, 61], [61, 62], [62, 63], [63, 64], [64, 65], [65, 66], [66, 67], [67, 60]
]

MP_CONNECTIVITY = [
    # FACE
    [10, 338], [338, 297], [297, 332], [332, 284], [284, 251],
    [251, 389], [389, 356], [356, 454], [454, 323], [323, 361],
    [361, 288], [288, 397], [397, 365], [365, 379], [379, 378],
    [378, 400], [400, 377], [377, 152], [152, 148], [148, 176],
    [176, 149], [149, 150], [150, 136], [136, 172], [172, 58],
    [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
    [162, 21], [21, 54], [54, 103], [103, 67], [67, 109],
    [109, 10],
    # RBROW
    [70, 63], [63, 105], [105, 66], [66, 107], [107, 55],
    [55, 65], [65, 52], [52, 53], [53, 46],
    # LBROW
    [300, 293], [293, 334], [334, 296], [296, 336], [336, 285],
    [285, 295], [295, 282], [282, 283], [283, 276],
    # NOSE
    [168, 6], [6, 197], [197, 195], [195, 5], [5, 4],
    [4, 1], [1, 19], [19, 94], [94, 2], [2, 164],
    [164, 0], [0, 11], [11, 12], [12, 13], [13, 14],
    [14, 15], [15, 16], [16, 17], [17, 18], [18, 200],
    [200, 199], [199, 175], [175, 152],
    # REYE
    [33, 7], [7, 163], [163, 144], [144, 145], [145, 153],
    [153, 154], [154, 155], [155, 133], [133, 173], [173, 157],
    [157, 158], [158, 159], [159, 160], [160, 161], [161, 246],
    # LEYE
    [263, 249], [249, 390], [390, 373], [373, 374], [374, 380],
    [380, 381], [381, 382], [382, 362], [362, 398], [398, 384],
    [384, 385], [385, 386], [386, 387], [387, 388], [388, 466],
    # RIRIS
    [469, 470], [470, 471], [471, 472], [472, 469],
    # LIRIS
    [474, 475], [475, 476], [476, 477], [477, 474],
    # LIP_OUTTER
    [61, 146], [146, 91], [91, 181], [181, 84], [84, 17],
    [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
    [291, 409], [409, 270], [270, 269], [269, 267], [267, 0],
    [0, 37], [37, 39], [39, 40], [40, 185], [185, 61],
    # LIP_INNER
    [78, 95], [95, 88], [88, 178], [178, 87], [87, 14],
    [14, 317], [317, 402], [402, 318], [318, 324], [324, 308],
    [308, 415], [415, 310], [310, 311], [311, 312], [312, 13],
    [13, 82], [82, 81], [81, 80], [80, 191], [191, 78]
]

SAPIENS_SEG_LABELS = [
    'Background',
    'Apparel',
    'Face_Neck',
    'Hair',
    'Left_Foot',
    'Left_Hand',
    'Left_Lower_Arm',
    'Left_Lower_Leg',
    'Left_Shoe',
    'Left_Sock',
    'Left_Upper_Arm',
    'Left_Upper_Leg',
    'Lower_Clothing',
    'Right_Foot',
    'Right_Hand',
    'Right_Lower_Arm',
    'Right_Lower_Leg',
    'Right_Shoe',
    'Right_Sock',
    'Right_Upper_Arm',
    'Right_Upper_Leg',
    'Torso',
    'Upper_Clothing',
    'Lower_Lip',
    'Upper_Lip',
    'Lower_Teeth',
    'Upper_Teeth',
    'Tongue'
]

FACE_DATASETS = {
    'HDTF': {
        'meta_file': '',
        'data_root': ''
    },
    'VFHQ': {
        'meta_file': '',
        'data_root': ''
    },
    'CCv2': {
        'meta_file': '',
        'data_root': ''
    }
}

PORTRAIT_DATASETS = {
    'HDTF': {
        'meta_file': '',
        'data_root': ''
    },
    'VFHQ': {
        'meta_file': '',
        'data_root': ''
    },
    'CCv2': {
        'meta_file': '',
        'data_root': ''
    }
}


SPEECH_DATASETS = {
    'PATs': {
        'meta_file': '',
        'data_root': '',
        'pose_root': '',
        'seg_root': ''
    },
    'TED': {
        'meta_file': '',
        'data_root': '',
        'pose_root': '',
        'seg_root': ''
    }
}

AUDIO_FEAT_DIM = {
    'wav2vec2-base-960h': 768,
    'hubert-large-ll60k': 1024
}

MODEL_PATH = {
    'resnet18d': '',
    'resnet50d': '',
    'hubert-large-ll60k': '',
    'wav2vec2-base-960h': ''
}

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.deterministic = True

def expand_bbox(bbox, ratio):
    sx, sy, ex, ey = bbox
    size = max(ex - sx, ey - sy) * ratio
    padding_x = size - (ex - sx)
    padding_y = size - (ey - sy)
    padding_l = padding_x / 2
    padding_u = padding_y / 2
    padding_r = padding_x - padding_l
    padding_b = padding_y - padding_u
    sx, ex = round(sx - padding_l), round(ex + padding_r)
    sy, ey = round(sy - padding_u), round(ey + padding_b)
    return [sx, sy, ex, ey]

def iou(bbox1, bbox2):
    i = (min(bbox1[2], bbox2[2]) - max(bbox1[0], bbox2[0])) * (min(bbox1[3], bbox2[3]) - max(bbox1[1], bbox2[1]))
    u = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) + (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - i
    return i / u

def crop_image(img, bbox):
    sx, ex = max(bbox[0], 0), min(bbox[2], img.shape[1])
    sy, ey = max(bbox[1], 0), min(bbox[3], img.shape[0])
    padding_l, padding_r = sx - bbox[0], bbox[2] - ex
    padding_u, padding_b = sy - bbox[1], bbox[3] - ey
    padding = [[padding_u, padding_b], [padding_l, padding_r]]
    padding += [[0, 0]] * (len(img.shape) - 2)
    return np.pad(img[sy: ey, sx: ex], padding)

def seg2msk(seg, labels=None):
    if labels is None:
        mask = seg > 0
    else:
        masks = np.stack([seg == label for label in labels])
        mask = masks.any(axis=0, keepdims=False)
    return (mask * 255).astype(np.uint8)

def lms2hmp(lms, size, var=1e-3):

    def make_coordinate_grid_2d(h, w):
        x = np.arange(w, dtype=np.float32)
        y = np.arange(h, dtype=np.float32)

        x = (2 * (x / (w - 1)) - 1)
        y = (2 * (y / (h - 1)) - 1)

        xx = x[None, ...].repeat(h, axis=0)[..., None]
        yy = y[..., None].repeat(w, axis=1)[..., None]

        return np.concatenate((xx, yy), axis=2)

    h, w = size

    lms = lms.copy()
    lms[:, 0] = 2 * lms[:, 0] / (w - 1) - 1
    lms[:, 1] = 2 * lms[:, 1] / (h - 1) - 1
    lms = np.clip(lms, a_min=-1.0, a_max=1.0)

    grid = make_coordinate_grid_2d(h, w)
    diff = grid[None, :, :, :] - lms[:, None, None, :]
    prob = np.exp(-0.5 * (diff ** 2).sum(axis=-1) / var)

    return prob.max(axis=1)[0]

def apply_imagenet_normalization(imgs):
    return torch.stack([F.normalize(img, IMAGENET_MEAN, IMAGENET_STD) for img in imgs])

def requires_grad(model, flag=True):
    for param in model.parameters():
        param.requires_grad = flag

def postprocess(imgs):
    return (imgs * 255).clamp(0, 255).type(torch.uint8)

def visualize_keypoints_3d(vs, colors=None, axis=False):
    assert(isinstance(vs, np.ndarray) and len(vs.shape) == 2)
    pcobj = open3d.geometry.PointCloud()
    pcobj.points = open3d.utility.Vector3dVector(vs)
    if colors is not None:
        pcobj.colors = open3d.utility.Vector3dVector(colors / 255)
    size, origin = (vs.max(axis=0) - vs.min(axis=0)).max().item() / 2, vs.mean(axis=0)
    coord = open3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    open3d.visualization.draw_geometries([pcobj, coord] if axis else [pcobj])

def paste_back(image, bbox, render):
    result = image.copy()
    sx, ex = max(bbox[0], 0), min(bbox[2], image.shape[1])
    sy, ey = max(bbox[1], 0), min(bbox[3], image.shape[0])
    padding_l, padding_r = sx - bbox[0], bbox[2] - ex
    padding_u, padding_b = sy - bbox[1], bbox[3] - ey
    render_ = cv2.resize(render, (bbox[2] - bbox[0], bbox[3] - bbox[1]))
    render_ = render_[padding_u: render_.shape[0] - padding_b, padding_l: render_.shape[1] - padding_r]
    result[sx: ex, sy: ey] = render_
    return result

def overlay_mask(imgs, masks, alpha=0.4, colors="red"):
    draw = partial(draw_segmentation_masks, alpha=alpha, colors=colors)
    return torch.stack([draw(img, mask) for img, mask in zip(imgs, masks)])

def overlay_landmarks(imgs, lms, connectivity=None, colors="red", radius=1, width=1):
    draw = partial(draw_keypoints, connectivity=connectivity, colors=colors, radius=radius, width=width)
    return torch.stack([draw(img, lmk.unsqueeze(0)) for img, lmk in zip(imgs, lms)])

def save_image(pth, img):
    img_array = img.permute(1, 2, 0).cpu().numpy()
    return cv2.imwrite(pth, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))


class NoFaceDetectedException(Exception):
    def __init__(self, *args, **kwargs):
        super(NoFaceDetectedException, self).__init__()


class NoFrameReturnedException(Exception):
    def __init__(self, *args, **kwargs):
        super(NoFrameReturnedException, self).__init__()


class FaceDetector:
    def __init__(self, detector='insightface'):
        self.detector = detector

        if self.detector == 'insightface':
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider'], allowed_modules=['detection'])
            self.app.prepare(ctx_id=0)
        elif self.detector == 'mediapipe':
            self.app = mp.solutions.face_detection.FaceDetection()
        else:
            raise NotImplementedError(f"Unsupported detector: {self.detector}!")

    def __call__(self, image):
        if self.detector == 'insightface':
            results = self.app.get(image)
            if not results: raise NoFaceDetectedException()
            return results[0]['bbox'].tolist()
        elif self.type == 'mediapipe':
            results = self.app.process(image)
            if not results.detections: raise NoFaceDetectedException()
            sx = results.detections[0].location_data.relative_bounding_box.xmin
            sy = results.detections[0].location_data.relative_bounding_box.ymin
            ex = sx + results.detections[0].location_data.relative_bounding_box.width
            ey = sy + results.detections[0].location_data.relative_bounding_box.height
            sx *= image.shape[1]; ex *= image.shape[1]
            sy *= image.shape[0]; ey *= image.shape[0]
            return [sx, sy, ex, ey]
        else:
            raise NotImplementedError(f"Unsupported detector: {self.detector}!")


class FaceAlignment:
    def __init__(self, detector='mediapipe'):
        self.detector = detector

        if self.detector == 'face_alignment':
            self.app = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        elif self.detector == 'mediapipe':
            self.app = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True)
        else:
            raise NotImplementedError(f"Unsupported detector: {self.detector}!")

    def __call__(self, image):
        if self.detector == 'face_alignment':
            results = self.app.get_landmarks_from_image(image)
            if not results:
                raise NoFaceDetectedException()
            return results[0].astype(np.float32)
        elif self.detector == 'mediapipe':
            results = self.app.process(image)
            if not results.multi_face_landmarks:
                raise NoFaceDetectedException
            lms = results.multi_face_landmarks[0].landmark
            lms = np.array([[lmk.x, lmk.y] for lmk in lms])
            lms[:, 0] *= image.shape[1]
            lms[:, 1] *= image.shape[0]
            return lms.astype(np.float32)
        else:
            raise NotImplementedError(f"Unsupported detector: {self.detector}!")


class VideoReader:
    def __init__(self, video_pth, expand_ratio, frame_size=256):
        self.cap = cv2.VideoCapture(video_pth)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.expand_ratio = expand_ratio
        self.img_size = [frame_size] * 2

        self.detector = FaceDetector()

    def relocate_face(self):
        if hasattr(self, 'face_bbox'):
            delattr(self, 'face_bbox')

    def get(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not hasattr(self, 'face_bbox'):
                self.face_bbox = expand_bbox(self.detector(frame), self.expand_ratio)
            aligned = cv2.resize(crop_image(frame, self.face_bbox), self.img_size)
            return aligned, frame, self.face_bbox
        else:
            raise NoFrameReturnedException()

    def __len__(self):
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self):
        self.cap.release()


class ImageReader:
    def __init__(self, expand_ratio, image_size=256):
        self.expand_ratio = expand_ratio
        self.img_size = [image_size] * 2

        self.detector = FaceDetector()

    def get(self, image_pth):
        image = cv2.cvtColor(cv2.imread(image_pth), cv2.COLOR_BGR2RGB)
        face_bbox = expand_bbox(self.detector(image), self.expand_ratio)
        aligned = cv2.resize(crop_image(image, face_bbox), self.img_size)
        return aligned, image, face_bbox


class AudioEncoder:
    def __init__(self, model, ckpt):
        if model == 'wav2vec2-base-960h':
            self.processor = Wav2Vec2Processor.from_pretrained(ckpt)
            self.network = Wav2Vec2Model.from_pretrained(ckpt)
        elif model == 'hubert-large-ll60k':
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(ckpt)
            self.network = HubertModel.from_pretrained(ckpt)
        else:
            raise NotImplementedError(f"Unsupported model type: {model}!")

        self.network.eval()

    @torch.inference_mode()
    def process(self, audio_path):
        audio, _ = librosa.load(audio_path, sr=16000)
        input = self.processor(audio, sampling_rate=16000, return_tensors='pt').input_values
        return self.network(input)['last_hidden_state'][0][::2]


class ExponentialMovingAverage:
    def __init__(self, beta):
        self.beta = beta

    def update_model(self, master_model, model):
        for old_params, new_params in zip(master_model.parameters(), model.parameters()):
            old_params.data = self.update(old_params.data, new_params.data)

    def update(self, old, new):
        return old * self.beta + (1 - self.beta) * new


class Timer:
    def start(self):
        self.start_time = time.time()

    def consume(self, format='h-m-s'):
        elapsed = int(time.time() - self.start_time)
        if format == 's':
            return elapsed
        elif format == 'h-m-s':
            hours = elapsed // 3600
            minutes = (elapsed % 3600) // 60
            seconds = (elapsed % 3600) % 60
            return hours, minutes, seconds
        elif format == 's-ms':
            seconds = int(elapsed)
            milliseconds = 1000 * (elapsed - seconds)
            return seconds, milliseconds
        else:
            raise NotImplementedError
