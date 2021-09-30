import torch
import torchvision.ops.boxes as box_ops

from torch import Tensor
from typing import List, Tuple

import time
import numpy as np
import os.path as osp
import torch
import pickle
# import yaml
import re
# from easydict import EasyDict as edict
# from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def custom_collate(batch):
    images = []
    detections = []
    targets = []
    for im, det, tar in batch:
        images.append(im)
        detections.append(det)
        targets.append(tar)
    return images, detections, targets


def compute_spatial_encodings(
        boxes_1: List[Tensor], boxes_2: List[Tensor],
        shapes: List[Tuple[int, int]], eps: float = 1e-10
) -> Tensor:
    """
    Parameters:
    -----------
        boxes_1: List[Tensor]
            First set of bounding boxes (M, 4)
        boxes_1: List[Tensor]
            Second set of bounding boxes (M, 4)
        shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        eps: float
            A small constant used for numerical stability

    Returns:
    --------
        Tensor
            Computed spatial encodings between the boxes (N, 36)
    """
    features = []
    for b1, b2, shape in zip(boxes_1, boxes_2, shapes):
        h, w = shape

        c1_x = (b1[:, 0] + b1[:, 2]) / 2
        c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2
        c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]
        b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]
        b2_h = b2[:, 3] - b2[:, 1]

        d_x = torch.abs(c2_x - c1_x) / (b1_w + eps)
        d_y = torch.abs(c2_y - c1_y) / (b1_h + eps)

        iou = torch.diag(box_ops.box_iou(b1, b2))

        # Construct spatial encoding
        f = torch.stack([
            # Relative position of box centre
            c1_x / w, c1_y / h, c2_x / w, c2_y / h,
            # Relative box width and height
            b1_w / w, b1_h / h, b2_w / w, b2_h / h,
            # Relative box area
            b1_w * b1_h / (h * w), b2_w * b2_h / (h * w),
            b2_w * b2_h / (b1_w * b1_h + eps),
            # Box aspect ratio
            b1_w / (b1_h + eps), b2_w / (b2_h + eps),
            # Intersection over union
            iou,
            # Relative distance and direction of the object w.r.t. the person
            (c2_x > c1_x).float() * d_x,
            (c2_x < c1_x).float() * d_x,
            (c2_y > c1_y).float() * d_y,
            (c2_y < c1_y).float() * d_y,
        ], 1)

        features.append(
            torch.cat([f, torch.log(f + eps)], 1)
        )
    return torch.cat(features)


def binary_focal_loss(
        x: Tensor, y: Tensor,
        alpha: float = 0.5,
        gamma: float = 2.0,
        reduction: str = 'mean',
        eps: float = 1e-6
) -> Tensor:
    """
    Focal loss by Lin et al.
    https://arxiv.org/pdf/1708.02002.pdf

    L = - |1-y-alpha| * |y-x|^{gamma} * log(|1-y-x|)

    Parameters:
    -----------
        x: Tensor[N, K]
            Post-normalisation scores
        y: Tensor[N, K]
            Binary labels
        alpha: float
            Hyper-parameter that balances between postive and negative examples
        gamma: float
            Hyper-paramter suppresses well-classified examples
        reduction: str
            Reduction methods
        eps: float
            A small constant to avoid NaN values from 'PowBackward'

    Returns:
    --------
        loss: Tensor
            Computed loss tensor
    """
    loss = (1 - y - alpha).abs() * ((y - x).abs() + eps) ** gamma * \
           torch.nn.functional.binary_cross_entropy(
               x, y, reduction='none'
           )
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError("Unsupported reduction method {}".format(reduction))


class Timer(object):
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.cnt = 0

    def update(self, val, k):
        self.avg = self.avg + (val - self.avg) * k / (self.cnt + k)
        self.cnt += k

    def __str__(self):
        return '%.4f' % self.avg


def get_config(config_path):
    loader = yaml.FullLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
         [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))

    config = edict(yaml.load(open(config_path, 'r'), Loader=loader))
    return config


def getSigmoid(b, c, d, x, a=6):
    e = 2.718281828459
    return a / (1 + e ** (b - c * x)) + d


def iou(bb1, bb2, debug=False):
    x1 = bb1[2] - bb1[0]
    y1 = bb1[3] - bb1[1]
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0

    x2 = bb2[1] - bb2[0]
    y2 = bb2[3] - bb2[2]
    if x2 < 0:
        x2 = 0
    if y2 < 0:
        y2 = 0

    xiou = min(bb1[2], bb2[1]) - max(bb1[0], bb2[0])
    yiou = min(bb1[3], bb2[3]) - max(bb1[1], bb2[2])
    if xiou < 0:
        xiou = 0
    if yiou < 0:
        yiou = 0

    if debug:
        print(x1, y1, x2, y2, xiou, yiou)
        print(x1 * y1, x2 * y2, xiou * yiou)
    if xiou * yiou <= 0:
        return 0
    else:
        return xiou * yiou / (x1 * y1 + x2 * y2 - xiou * yiou)


def calc_hit(det, gtbox):
    gtbox = gtbox.astype(np.float64)
    hiou = iou(det[:4], gtbox[:4])
    oiou = iou(det[4:], gtbox[4:])
    return min(hiou, oiou)


def calc_ap(scores, bboxes, keys, hoi_id, begin):
    if len(keys) == 0:
        return 0, 0
    score = scores[:, hoi_id - begin]
    hit = []
    idx = np.argsort(score)[::-1]
    gt_bbox = pickle.load(open('gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'), encoding='latin1')
    npos = 0
    used = {}

    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        return 0, 0
    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(hit)
    rec = hit / npos
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0

    return ap, np.max(rec)


def calc_ap_ko(scores, bboxes, keys, hoi_id, begin, ko_mask):
    score = scores[:, hoi_id - begin]
    hit, hit_ko = [], []
    idx = np.argsort(score)[::-1]
    gt_bbox = pickle.load(open('gt_hoi_py2/hoi_%d.pkl' % hoi_id, 'rb'))
    npos = 0
    used = {}

    for key in gt_bbox.keys():
        npos += gt_bbox[key].shape[0]
        used[key] = set()
    if len(idx) == 0:
        output = {
            'ap': 0, 'rec': 0, 'ap_ko': 0, 'rec_ko': 0
        }
        return output
    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
                hit_ko.append(0)
            else:
                hit.append(1)
                hit_ko.append(1)
                used[key].add(k)
        else:
            hit.append(0)
            if key in ko_mask:
                hit_ko.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit = np.cumsum(hit)
    rec = hit / npos
    prec = hit / bottom
    ap = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    if len(hit_ko) == 0:
        output = {
            'ap': ap, 'rec': np.max(rec), 'ap_ko': 0, 'rec_ko': 0
        }
        return output
    bottom_ko = np.array(range(len(hit_ko))) + 1
    hit_ko = np.cumsum(hit_ko)
    rec_ko = hit_ko / npos
    prec_ko = hit_ko / bottom_ko
    ap_ko = 0.0
    for i in range(11):
        mask = rec_ko >= (i / 10.)
        if np.sum(mask) > 0:
            ap_ko += np.max(prec_ko[mask]) / 11.
    output = {
        'ap': ap, 'rec': np.max(rec), 'ap_ko': ap_ko, 'rec_ko': np.max(rec_ko)
    }
    return output


def get_map(keys, scores, bboxes):
    map = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end = obj_range[i][1]
        for hoi_id in range(begin, end):
            score = scores[i]
            bbox = bboxes[i]
            key = keys[i]
            map[hoi_id], mrec[hoi_id] = calc_ap(score, bbox, key, hoi_id, begin)
    return map, mrec
