import os
import json
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset_idn import HICO_test_set
from prefetch_generator import BackgroundGenerator

from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.ops.boxes import box_iou
from torchvision.transforms.functional import hflip

from .vcoco import VCOCO
from .hicodet import HICODet

import pocket
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, HandyTimer, BoxPairAssociation, all_gather
from utils import custom_collate, get_config, DataLoaderX
import pickle

import yaml
import re
from easydict import EasyDict as edict


class CustomInput(object):
    """Generates an input specific to the one required by official implementation of concerned model"""
    def __init__(self, model_name):
        self.func_map = {
            'scg': self.scg,
            'drg': self.drg,
            'idn': self.idn,
            'cascaded-hoi': self.cascaded_hoi,
        }
        self.converter = self.func_map[model_name]

    def scg(self, image, boxes, labels, scores):
        """Merges the arguments into a data point for the scg model

        Args:
            image (np.array)
            boxes (list of list): detected box coords
            labels (list): detected box labels
            scores (list): detected box scores for selected class
        """
        data_point = list()
        data_point.append([torch.from_numpy(image)])
        detections = [{
            'boxes': torch.from_numpy(boxes),
            'labels': torch.from_numpy(labels),
            'scores': torch.from_numpy(scores),
        }]
        data_point.append(detections)
        return data_point

    def drg(self):
        raise NotImplementedError

    def idn(self, image, boxes, labels, scores):
        args_idn = pickle.load(open('configs/arguments.pkl', 'rb'))
        config = get_config('configs/IDN.yml')
        test_set    = HICO_test_set(config.TRAIN.DATA_DIR, split='test')
        test_loader = DataLoaderX(test_set, batch_size=1, shuffle=False, collate_fn=test_set.collate_fn, pin_memory=False, drop_last=False)
        verb_mapping = torch.from_numpy(pickle.load(open('configs/verb_mapping.pkl', 'rb'), encoding='latin1')).float()
        for i, batch in enumerate(test_loader):
            n = batch['shape'].shape[0]
            batch['shape']   = batch['shape'].cuda(non_blocking=True)
            batch['spatial'] = batch['spatial'].cuda(non_blocking=True)
            batch['sub_vec'] = batch['sub_vec'].cuda(non_blocking=True)
            batch['obj_vec'] = batch['obj_vec'].cuda(non_blocking=True)
            batch['uni_vec'] = batch['uni_vec'].cuda(non_blocking=True)
            batch['labels_s']   = batch['labels_s'].cuda(non_blocking=True)
            batch['labels_ro']  = batch['labels_ro'].cuda(non_blocking=True)
            batch['labels_r']   = batch['labels_r'].cuda(non_blocking=True)
            batch['labels_sro'] = batch['labels_sro'].cuda(non_blocking=True)
            verb_mapping    = verb_mapping.cuda(non_blocking=True)
            break
            
        return batch

    def cascaded_hoi(self):
        raise NotImplementedError


class DataFactory(Dataset):
    def __init__(self,
            name, partition,
            data_root, detection_root,
            flip=False,
            box_score_thresh_h=0.2,
            box_score_thresh_o=0.2
            ):
        if name not in ['hicodet', 'vcoco']:
            raise ValueError("Unknown dataset ", name)

        if name == 'hicodet':
            assert partition in ['train2015', 'test2015'], \
                "Unknown HICO-DET partition " + partition
            self.dataset = HICODet(
                root=os.path.join(data_root, 'hico_20160224_det/images', partition),
                anno_file=os.path.join(data_root, 'instances_{}.json'.format(partition)),
                target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 49
        else:
            assert partition in ['train', 'val', 'trainval', 'test'], \
                "Unknown V-COCO partition " + partition
            image_dir = dict(
                train='mscoco2014/train2014',
                val='mscoco2014/train2014',
                trainval='mscoco2014/train2014',
                test='mscoco2014/val2014'
            )
            self.dataset = VCOCO(
                root=os.path.join(data_root, image_dir[partition]),
                anno_file=os.path.join(data_root, 'instances_vcoco_{}.json'.format(partition)
                ), target_transform=pocket.ops.ToTensor(input_format='dict')
            )
            self.human_idx = 1

        self.name = name
        self.detection_root = detection_root

        self.box_score_thresh_h = box_score_thresh_h
        self.box_score_thresh_o = box_score_thresh_o
        self._flip = torch.randint(0, 2, (len(self.dataset),)) if flip \
            else torch.zeros(len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def filter_detections(self, detection):
        """Perform NMS and remove low scoring examples"""

        boxes = torch.as_tensor(detection['boxes'])
        labels = torch.as_tensor(detection['labels'])
        scores = torch.as_tensor(detection['scores'])

        # Filter out low scoring human boxes
        idx = torch.nonzero(labels == self.human_idx).squeeze(1)
        keep_idx = idx[torch.nonzero(scores[idx] >= self.box_score_thresh_h).squeeze(1)]

        # Filter out low scoring object boxes
        idx = torch.nonzero(labels != self.human_idx).squeeze(1)
        keep_idx = torch.cat([
            keep_idx,
            idx[torch.nonzero(scores[idx] >= self.box_score_thresh_o).squeeze(1)]
        ])

        boxes = boxes[keep_idx].view(-1, 4)
        scores = scores[keep_idx].view(-1)
        labels = labels[keep_idx].view(-1)

        return dict(boxes=boxes, labels=labels, scores=scores)

    def flip_boxes(self, detection, target, w):
        detection['boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['boxes'])
        target['boxes_h'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_h'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])

    def __getitem__(self, i):
        image, target = self.dataset[i]
        if self.name == 'hicodet':
            target['labels'] = target['verb']
            # Convert ground truth boxes to zero-based index and the
            # representation from pixel indices to coordinates
            target['boxes_h'][:, :2] -= 1
            target['boxes_o'][:, :2] -= 1
        else:
            target['labels'] = target['actions']
            target['object'] = target.pop('objects')

        detection_path = os.path.join(
            self.detection_root,
            self.dataset.filename(i).replace('jpg', 'json')
        )
        with open(detection_path, 'r') as f:
            detection = pocket.ops.to_tensor(json.load(f),
                input_format='dict')

        if self._flip[i]:
            image = hflip(image)
            w, _ = image.size
            self.flip_boxes(detection, target, w)
        image = pocket.ops.to_tensor(image, 'pil')

        return image, detection, target
