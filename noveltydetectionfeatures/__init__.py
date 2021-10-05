import pickle
import os
import torch
import pocket
from utils import compute_spatial_encodings, binary_focal_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform
from models.scg.scg import HOINetworkTransform
import pocket.models as models
from data.data_factory import DataFactory
from torch.utils.data import DataLoader, DistributedSampler
from utils import custom_collate

from tqdm import tqdm

class NoveltyFeatureDataset(torch.utils.data.Dataset):
    """
    Params:
        name: As in data.data_factory.DataFactory()
        data_root: As in data.data_factory.DataFactory()
        csv_path: As in data.data_factory.DataFactory()
        num_subj_cls: As in data.data_factory.DataFactory()
        num_obj_cls: As in data.data_factory.DataFactory()
        num_action_cls: As in data.data_factory.DataFactory()
        training: As in data.data_factory.DataFactory()
        image_batch_size: If the features have not yet been computed as
            persisted to disk, then they will be. This is done by constructing
            a data loader for the raw image data and precomputing the features.
            This parameter specifies the batch size to use on the raw image
            data in such a case.
        output_size: As in torchvision.ops.MultiScaleRoIAlign()
        sampling_ratio: As in torchvision.ops.MultiScaleRoIAlign()
        min_size: As in models.scg.scg.HOINetworkTransform()
        max_size: As in models.scg.scg.HOINetworkTransform()
        image_mean: As in models.scg.scg.HOINetworkTransform()
        image_std: As in models.scg.scg.HOINetworkTransform()
    """
    def __init__(
            self,
            name,
            data_root,
            csv_path,
            num_subj_cls,
            num_obj_cls,
            num_action_cls,
            training,
            image_batch_size,
            output_size = 7,
            sampling_ratio = 2,
            min_size = 800,
            max_size = 1333,
            image_mean = None,
            image_std = None,
            feature_extraction_device = 'cpu'):
        super().__init__()
        
        filename = os.path.join(f'{os.path.splitext(csv_path)[0]}_novelty_features.pth')
        if not os.path.exists(filename):
            with torch.no_grad():
                print('Image features have not yet been computed. Computing image features...')
                dataset = DataFactory(
                    name = name,
                    partition = None,
                    data_root = data_root,
                    csv_path = csv_path,
                    training = training,
                    num_subj_cls = num_subj_cls,
                    num_obj_cls = num_obj_cls,
                    num_action_cls = num_action_cls
                )
                
                data_loader = DataLoader(
                    dataset = dataset,
                    collate_fn = custom_collate,
                    batch_size = image_batch_size
                )
                
                detector = models.fasterrcnn_resnet_fpn('resnet50', pretrained=True)
                backbone = detector.backbone.to(feature_extraction_device)
                
                box_roi_pool = MultiScaleRoIAlign(
                    featmap_names=['0', '1', '2', '3'],
                    output_size=output_size,
                    sampling_ratio=sampling_ratio
                )
                
                if image_mean is None:
                    image_mean = [0.485, 0.456, 0.406]
                if image_std is None:
                    image_std = [0.229, 0.224, 0.225]
                
                i_transform = HOINetworkTransform(
                    min_size,
                    max_size,
                    image_mean,
                    image_std
                )
                
                box_pair_spatial = list()
                subject_labels = list()
                object_labels = list()
                verb_labels = list()
                subject_box_features = list()
                object_box_features = list()
                verb_box_features = list()
                
                for images, detections, targets in tqdm(data_loader):
                    original_image_sizes = [img.shape[-2:] for img in images]
                    images, targets = i_transform(images, targets)
                    for det, o_im_s, im_s in zip(
                            detections, original_image_sizes, images.image_sizes
                    ):
                        sub_boxes = det['subject_boxes']
                        sub_boxes = transform.resize_boxes(sub_boxes, o_im_s, im_s)
                        det['subject_boxes'] = sub_boxes
                        obj_boxes = det['object_boxes']
                        obj_boxes = transform.resize_boxes(obj_boxes, o_im_s, im_s)
                        det['object_boxes'] = obj_boxes
                    
                    features = backbone(images.tensors.to(feature_extraction_device))
                    features = {k: v.cpu() for k, v in features.items()}
                    
                    image_shapes = images.image_sizes
                    
                    subject_box_coords = list()
                    object_box_coords = list()
                    verb_box_coords = list()
                    batch_box_pair_spatial = list()
                    batch_subject_labels = list()
                    batch_object_labels = list()
                    batch_verb_labels = list()
                    for b_idx, detection in enumerate(detections):
                        subject_box_coords.append(detection['subject_boxes'])
                        object_box_coords.append(detection['object_boxes'])
                        
                        n_s = len(detection['subject_boxes'])
                        n_o = len(detection['object_boxes'])
                        if n_s == 1 and n_o == 1:
                            s_xmin, s_ymin, s_xmax, s_ymax = detection['subject_boxes'][0]
                            o_xmin, o_ymin, o_xmax, o_ymax = detection['object_boxes'][0]
                            v_xmin = min(s_xmin, o_xmin)
                            v_ymin = min(s_ymin, o_ymin)
                            v_xmax = max(s_xmax, o_xmax)
                            v_ymax = max(s_ymax, o_ymax)
                            verb_box_coords.append(torch.tensor([[v_xmin, v_ymin, v_xmax, v_ymax]]))
                        elif n_o == 0:
                            verb_box_coords.append(detection['subject_boxes'].clone().detach())
                        else:
                            return NotImplemented
                        
                        x, y = torch.meshgrid(
                            torch.arange(n_s),
                            torch.arange(n_s + n_o)
                        )
                        x = x.flatten()
                        y = y.flatten()
                        coords = torch.cat([detection['subject_boxes'], detection['object_boxes']])
                        batch_box_pair_spatial.append(compute_spatial_encodings(
                            [coords[x]], [coords[y]], [image_shapes[b_idx]]
                        ))
                        
                        batch_subject_labels.append(detection['subject_labels'])
                        batch_object_labels.append(detection['object_labels'])
                        batch_verb_labels.append(targets[b_idx]['verb'] if targets is not None else None)
                        
                    batch_box_pair_spatial = torch.stack(batch_box_pair_spatial)
                    batch_subject_labels = torch.cat(batch_subject_labels)
                    batch_object_labels = torch.cat(batch_object_labels)
                    batch_verb_labels = torch.cat(batch_verb_labels)
                    batch_subject_box_features = box_roi_pool(features, subject_box_coords, image_shapes)
                    batch_object_box_features = box_roi_pool(features, object_box_coords, image_shapes)
                    batch_verb_box_features = box_roi_pool(features, verb_box_coords, image_shapes)
                    
                    box_pair_spatial.append(batch_box_pair_spatial)
                    subject_labels.append(batch_subject_labels)
                    object_labels.append(batch_object_labels)
                    verb_labels.append(batch_verb_labels)
                    subject_box_features.append(batch_subject_box_features)
                    object_box_features.append(batch_object_box_features)
                    verb_box_features.append(batch_verb_box_features)
                
                box_pair_spatial = torch.cat(box_pair_spatial)
                subject_labels = torch.cat(subject_labels)
                object_labels = torch.cat(object_labels)
                verb_labels = torch.cat(verb_labels)
                subject_box_features = torch.cat(subject_box_features)
                object_box_features = torch.cat(object_box_features)
                verb_box_features = torch.cat(verb_box_features)
                
                self.spatial_features = box_pair_spatial.detach()
                self.subject_labels = subject_labels.detach()
                self.object_labels = object_labels.detach()
                self.verb_labels = verb_labels.detach()
                self.subject_appearance_features = subject_box_features.detach()
                self.object_appearance_features = object_box_features.detach()
                self.verb_appearance_features = verb_box_features.detach()
                
                data = {
                    'spatial_features': self.spatial_features,
                    'subject_labels': self.subject_labels,
                    'object_labels': self.object_labels,
                    'verb_labels': self.verb_labels,
                    'subject_appearance_features': self.subject_appearance_features,
                    'object_appearance_features': self.object_appearance_features,
                    'verb_appearance_features': self.verb_appearance_features,
                }
                
                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
        else:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            self.spatial_features = data['spatial_features']
            self.subject_labels = data['subject_labels']
            self.object_labels = data['object_labels']
            self.verb_labels = data['verb_labels']
            self.subject_appearance_features = data['subject_appearance_features']
            self.object_appearance_features = data['object_appearance_features']
            self.verb_appearance_features = data['verb_appearance_features']

    def __getitem__(self, idx):
        return self.spatial_features[idx], self.subject_appearance_features[idx], self.object_appearance_features[idx], self.verb_appearance_features[idx], self.subject_labels[idx], self.object_labels[idx], self.verb_labels[idx],

    def __len__(self):
        return len(self.spatial_features)
