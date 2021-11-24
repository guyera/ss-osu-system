import torch

from torch.utils.data import Dataset
from torchvision.transforms.functional import hflip

from .custom import CustomDet

import pocket


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

    def scg(self, image, subject_boxes, subject_labels, object_boxes, object_labels):
        """Merges the arguments into a data point for the scg model

        Args:
            image (np.array)
            subject_boxes (list of list): detected box coords for subjects
            subject_labels (list): detected box labels for subjects
            object_boxes (list of list): detected box coords for objects
            object_labels (list): detected box labels for objects
        """
        data_point = list()
        data_point.append([torch.from_numpy(image)])
        detections = [{
            'subject_boxes': torch.from_numpy(subject_boxes),
            'subject_labels': torch.from_numpy(subject_labels),
            'object_boxes': torch.from_numpy(object_boxes),
            'object_labels': torch.from_numpy(object_labels),
        }]
        data_point.append(detections)
        return data_point

    def drg(self):
        raise NotImplementedError

    def idn(self, image, boxes, labels, scores):
        raise NotImplementedError
    #     args_idn = pickle.load(open('configs/arguments.pkl', 'rb'))
    #     config = get_config('configs/IDN.yml')
    #     test_set = HICO_test_set(config.TRAIN.DATA_DIR, split='test')
    #     test_loader = DataLoaderX(test_set, batch_size=1, shuffle=False, collate_fn=test_set.collate_fn,
    #                               pin_memory=False, drop_last=False)
    #     verb_mapping = torch.from_numpy(pickle.load(open('configs/verb_mapping.pkl', 'rb'), encoding='latin1')).float()
    #     for i, batch in enumerate(test_loader):
    #         n = batch['shape'].shape[0]
    #         batch['shape'] = batch['shape'].cuda(non_blocking=True)
    #         batch['spatial'] = batch['spatial'].cuda(non_blocking=True)
    #         batch['sub_vec'] = batch['sub_vec'].cuda(non_blocking=True)
    #         batch['obj_vec'] = batch['obj_vec'].cuda(non_blocking=True)
    #         batch['uni_vec'] = batch['uni_vec'].cuda(non_blocking=True)
    #         batch['labels_s'] = batch['labels_s'].cuda(non_blocking=True)
    #         batch['labels_ro'] = batch['labels_ro'].cuda(non_blocking=True)
    #         batch['labels_r'] = batch['labels_r'].cuda(non_blocking=True)
    #         batch['labels_sro'] = batch['labels_sro'].cuda(non_blocking=True)
    #         verb_mapping = verb_mapping.cuda(non_blocking=True)
    #         break
    #
    #     return batch

    def cascaded_hoi(self):
        raise NotImplementedError


class DataFactory(Dataset):
    def __init__(self,
                 name, data_root,
                 csv_path=None,
                 flip=False,
                 box_score_thresh_h=0.2,
                 box_score_thresh_o=0.2,
                 training=True,
                 partition=None):
        self.training = training
        # Ensuring explicit awareness of user about this fact
        if name != 'Custom':
            raise ValueError("Unknown dataset ", name)

        self.dataset = CustomDet(root=data_root,csv_path=csv_path, target_transform=pocket.ops.ToTensor(input_format='dict'))

        self.name = name

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

        # Filter out low scoring human boxes
        subject_idxs = torch.nonzero(labels == self.subject_idx).squeeze(1)
        object_idxs = torch.nonzero(labels != self.subject_idx).squeeze(1)

        subject_boxes = boxes[subject_idxs].view(-1, 4)
        subject_labels = labels[subject_idxs].view(-1)

        object_boxes = boxes[object_idxs].view(-1, 4)
        object_labels = labels[object_idxs].view(-1)

        if self.training:
            return dict(subject_boxes=subject_boxes, subject_labels=subject_labels,
                        object_boxes=object_boxes, object_labels=object_labels)
        else:
            return dict(subject_boxes=subject_boxes, subject_labels=subject_labels,
                        object_boxes=object_boxes, object_labels=object_labels,
                        img_id=detection['img_id'])

    def flip_boxes(self, detection, target, w):
        detection['subject_boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['subject_boxes'])
        detection['object_boxes'] = pocket.ops.horizontal_flip_boxes(w, detection['object_boxes'])
        target['boxes_s'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_s'])
        target['boxes_o'] = pocket.ops.horizontal_flip_boxes(w, target['boxes_o'])

    def __getitem__(self, i):
        image, target = self.dataset[i]
        target["labels"] = target['verb']
        detections = self.dataset.get_detections(i)
        detection = pocket.ops.to_tensor(detections, input_format='dict')

        if not self.training:
            detection['img_id'] = self.dataset.filename(i)

        if self._flip[i]:
            image = hflip(image)
            w, _ = image.size
            self.flip_boxes(detection, target, w)
        image = pocket.ops.to_tensor(image, 'pil')

        return image, detection, target
