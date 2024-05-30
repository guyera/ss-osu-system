# Copyright 2023 University Of Maryland
# Contact Abhinav Shrivastava (abhinav@cs.umd.edu)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this code and associated documentation files (the "Code"), to deal
# in the Code without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Code, and to permit persons to whom the Code is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Code.

import torch
import torchvision.ops.boxes as box_ops
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

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


class LengthedIter:
    def __init__(self, l):
        self._iter = iter(l)
        self._length = len(l)

    def __len__(self):
        return self._length

    def __next__(self):
        return next(self._iter)


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


'''
Parameters: 
    jagged_indices: list[int]
        Specifies a set of indices corresponding to "jagged tensor" components
        of the batch item tuple, which should be kept in list form. Default is
        [3], which corresponds to the box_images component in a BoxImageDataset
'''
def gen_custom_collate(jagged_indices=None):
    if jagged_indices is None:
        jagged_indices = [3]
    jagged_indices_set = set(jagged_indices)
    def custom_collate(batch):
        res_list = []
        init_list = False
        for data_tuple in batch:
            for i, component in enumerate(data_tuple):
                if not init_list:
                    res_list.append([])
                res_list[i].append(component)
            init_list = True

        # Stack each of the list components, except for the ones corresponding
        # to jagged tensor indices (keep them in list form)
        for i, component in enumerate(res_list):
            if i in jagged_indices_set:
                continue
            res_list[i] = torch.stack(component, dim=0)

        # Convert res_list to tuple and return
        return tuple(res_list)
    return custom_collate


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

def fpn_backbone(backbone):
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    extra_blocks = LastLevelMaxPool()
    returned_layers = [1, 2, 3, 4]
    return_layers = {f'layer{k}': str(v) for v, k in enumerate(returned_layers)}
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [in_channels_stage2 * 2 ** (i - 1) for i in returned_layers]
    out_channels = 256
    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks=extra_blocks)

def gen_tqdm_description(title, **kwargs):
    desc = title
    for k, v in kwargs.items():
        if v is not None:
            desc = f'{desc} | {k}: {v}'
    return desc
