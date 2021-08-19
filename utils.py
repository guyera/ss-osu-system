import torch
import torchvision.ops.boxes as box_ops

from torch import Tensor
from typing import List, Tuple

import time
import numpy as np
import os.path as osp
import torch
import pickle
import yaml
import re
from easydict import EasyDict as edict
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

verb_mapping = torch.from_numpy(pickle.load(open('verb_mapping.pkl', 'rb'), encoding='latin1')).float()
args      = pickle.load(open('arguments.pkl', 'rb'))
HO_weight = torch.from_numpy(args['HO_weight'])
fac_i     = args['fac_i']
fac_a     = args['fac_a']
fac_d     = args['fac_d']
nis_thresh= args['nis_thresh']


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

        c1_x = (b1[:, 0] + b1[:, 2]) / 2; c1_y = (b1[:, 1] + b1[:, 3]) / 2
        c2_x = (b2[:, 0] + b2[:, 2]) / 2; c2_y = (b2[:, 1] + b2[:, 3]) / 2

        b1_w = b1[:, 2] - b1[:, 0]; b1_h = b1[:, 3] - b1[:, 1]
        b2_w = b2[:, 2] - b2[:, 0]; b2_h = b2[:, 3] - b2[:, 1]

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
    loss = (1 - y - alpha).abs() * ((y-x).abs() + eps) ** gamma * \
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


rare_index = np.array([ 9, 23,28, 45,51, 56,63, 64,67, 71,77, 78,81, 84,85, 91,100,101,105,108,113,128,136,137,150,159,166,167,169,173,180,182,185,189,190,193,196,199,206,207,215,217,223,228,230,239,240,255,256,258,261,262,263,275,280,281,282,287,290,293,304,312,316,318,326,329,334,335,346,351,352,355,359,365,380,382,390,391,392,396,398,399,400,402,403,404,405,406,408,411,417,419,427,428,430,432,437,440,441,450,452,464,470,475,483,486,499,500,505,510,515,518,521,523,527,532,536,540,547,548,549,550,551,552,553,556,557,561,579,581,582,587,593,594,596,597,598,600,]) - 1
rare = np.zeros(600)
rare[rare_index] += 2

obj_range = [
    (161, 170), (11, 24),   (66, 76),   (147, 160), (1, 10), 
    (55, 65),   (187, 194), (568, 576), (32, 46),   (563, 567), 
    (326, 330), (503, 506), (415, 418), (244, 247), (25, 31), 
    (77, 86),   (112, 129), (130, 146), (175, 186), (97, 107), 
    (314, 325), (236, 239), (596, 600), (343, 348), (209, 214), 
    (577, 584), (353, 356), (539, 546), (507, 516), (337, 342), 
    (464, 474), (475, 483), (489, 502), (369, 376), (225, 232), 
    (233, 235), (454, 463), (517, 528), (534, 538), (47, 54), 
    (589, 595), (296, 305), (331, 336), (377, 383), (484, 488), 
    (253, 257), (215, 224), (199, 208), (439, 445), (398, 407), 
    (258, 264), (274, 283), (357, 363), (419, 429), (306, 313), 
    (265, 273), (87, 92),   (93, 96),   (171, 174), (240, 243), 
    (108, 111), (551, 558), (195, 198), (384, 389), (394, 397), 
    (435, 438), (364, 368), (284, 290), (390, 393), (408, 414), 
    (547, 550), (450, 453), (430, 434), (248, 252), (291, 295), 
    (585, 588), (446, 449), (529, 533), (349, 352), (559, 562)
]


hoi_no_inter_all = [
    10,24,31,46,54,65,76,86,92,96,107,111,129,146,160,170,174,186,194,198,208,214,
    224,232,235,239,243,247,252,257,264,273,283,290,295,305,313,325,330,336,342,348,
    352,356,363,368,376,383,389,393,397,407,414,418,429,434,438,445,449,453,463,474,
    483,488,502,506,516,528,533,538,546,550,558,562,567,576,584,588,595,600
]


def getSigmoid(b,c,d,x,a=6):
    e = 2.718281828459
    return a/(1+e**(b-c*x))+d

def iou(bb1, bb2, debug = False):
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
        key  = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k    = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
            if k in used[key] or maxi < 0.5:
                hit.append(0)
            else:
                hit.append(1)
                used[key].add(k)
        else:
            hit.append(0)
    bottom = np.array(range(len(hit))) + 1
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
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
            'ap' : 0, 'rec': 0, 'ap_ko': 0, 'rec_ko': 0
        }
        return output
    for i in range(min(len(idx), 19999)):
        pair_id = idx[i]
        bbox = bboxes[pair_id, :]
        key  = keys[pair_id]
        if key in gt_bbox:
            maxi = 0.0
            k    = -1
            for i in range(gt_bbox[key].shape[0]):
                tmp = calc_hit(bbox, gt_bbox[key][i, :])
                if maxi < tmp:
                    maxi = tmp
                    k    = i
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
    hit    = np.cumsum(hit)
    rec    = hit / npos
    prec   = hit / bottom
    ap     = 0.0
    for i in range(11):
        mask = rec >= (i / 10.0)
        if np.sum(mask) > 0:
            ap += np.max(prec[mask]) / 11.0
    if len(hit_ko) == 0:
        output = {
            'ap' : ap, 'rec': np.max(rec), 'ap_ko': 0, 'rec_ko': 0
        }
        return output
    bottom_ko = np.array(range(len(hit_ko))) + 1
    hit_ko    = np.cumsum(hit_ko)
    rec_ko    = hit_ko / npos
    prec_ko   = hit_ko / bottom_ko
    ap_ko     = 0.0
    for i in range(11):
        mask = rec_ko >= (i / 10.)
        if np.sum(mask) > 0:
            ap_ko += np.max(prec_ko[mask]) / 11.
    output = {
        'ap' : ap, 'rec': np.max(rec), 'ap_ko': ap_ko, 'rec_ko': np.max(rec_ko)
    }
    return output

def get_map(keys, scores, bboxes):
    map  = np.zeros(600)
    mrec = np.zeros(600)
    for i in range(80):
        begin = obj_range[i][0] - 1
        end   = obj_range[i][1]
        for hoi_id in range(begin, end):
            score = scores[i]
            bbox  = bboxes[i]
            key   = keys[i]
            map[hoi_id], mrec[hoi_id] = calc_ap(score, bbox, key, hoi_id, begin)
    return map, mrec