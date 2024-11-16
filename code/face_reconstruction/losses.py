import torch

from utils import LBROW, MP_LBROW
from utils import RBROW, MP_RBROW
from utils import NOSE, MP_NOSE
from utils import LEYE, MP_LEYE
from utils import REYE, MP_REYE
from utils import MP_LIRIS, MP_RIRIS
from utils import LIP, MP_LIP
from utils import FACE, MP_FACE

LEYE_CLOSURE_PAIRS = [[43, 47], [44, 46]]
REYE_CLOSURE_PAIRS = [[37, 41], [38, 40]]
EYE_CLOSURE_PAIRS = [*LEYE_CLOSURE_PAIRS, *REYE_CLOSURE_PAIRS]

MP_LEYE_CLOSURE_PAIRS = [[398, 382], [384, 381], [385, 380], [386, 374], [387, 373], [388, 390], [466, 249]]
MP_REYE_CLOSURE_PAIRS = [[246, 7], [161, 163], [160, 144], [159, 145], [158, 153], [157, 154], [173, 155]]
MP_EYE_CLOSURE_PAIRS = [*MP_LEYE_CLOSURE_PAIRS, *MP_REYE_CLOSURE_PAIRS]

LIP_CLOSURE_PAIRS = [[61, 67], [62, 66], [63, 65]]

MP_LIP_CLOSURE_PAIRS = [[191, 95], [80, 88], [81, 178], [82, 87], [13, 14], [312, 317], [311, 402], [310, 318], [415, 324]]

def get_lmk_weights_68():
    weights = torch.ones(68)
    weights[LIP] = 10
    weights[NOSE] = 10
    weights[LEYE] = 10
    weights[REYE] = 10
    weights[LBROW] = 10
    weights[RBROW] = 10
    weights[FACE] = 10
    return weights / weights.sum()

def get_lmk_weights_478():
    weights = torch.ones(478)
    weights[MP_LIP] = 20
    weights[MP_LEYE] = 20
    weights[MP_REYE] = 20
    weights[MP_LIRIS] = 20
    weights[MP_RIRIS] = 20
    weights[MP_LBROW] = 20
    weights[MP_RBROW] = 20
    weights[MP_FACE] = 20
    return weights / weights.sum()

def landmark_loss(pred_lms, gt_lms, w):
    loss = torch.nn.functional.huber_loss(pred_lms, gt_lms, reduction='none')
    return torch.mean(loss.sum(dim=-1) * w)

def photo_loss(pred_img, gt_img, pred_mask, gt_mask=None):
    mask = pred_mask * gt_mask if gt_mask is not None else pred_mask
    loss = torch.sum((pred_img - gt_img).abs(), dim=1) * mask
    loss = torch.sum(loss, dim=(1, 2)) / (torch.sum(mask, dim=(1, 2)) + 1e-5)
    return torch.mean(loss)

def identity_loss(pred_feature, gt_feature):
    loss = 1 - torch.nn.functional.cosine_similarity(pred_feature, gt_feature)
    return torch.mean(loss)

def get_reg(tensor):
    return torch.sum(tensor.pow(2))

def id_consistent_loss(id_coeff):
    diff = id_coeff - id_coeff.mean(dim=1, keepdim=True).detach()
    loss = torch.sqrt(torch.sum(diff.pow(2), dim=(1, 2)))
    return torch.mean(loss)

def closure_loss(pred_lms, gt_lms, pairs):
    pred_dists = (pred_lms[:, pairs[:, 0]] - pred_lms[:, pairs[:, 1]]).pow(2).sum(dim=-1).sqrt()
    gt_dists = (gt_lms[:, pairs[:, 0]] - gt_lms[:, pairs[:, 1]]).pow(2).sum(dim=-1).sqrt()
    return torch.mean((pred_dists - gt_dists).abs())

def gaze_loss(pred_lms, gt_lms, eye, iris):
    pred_dists = (pred_lms[:, eye] - pred_lms[:, iris]).pow(2).sum(dim=-1).sqrt()
    gt_dists = (gt_lms[:, eye] - gt_lms[:, iris]).pow(2).sum(dim=-1).sqrt()
    return torch.mean((pred_dists - gt_dists).abs())
