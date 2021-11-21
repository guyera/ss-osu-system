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
            feature_extraction_device = 'cpu',
            cache_to_disk = True):
        super().__init__()
        
        filename = os.path.join(f'{os.path.splitext(csv_path)[0]}_novelty_features.pth')

        if cache_to_disk:
            if not os.path.exists(filename):
                print('Image features have not yet been computed. Computing image features...')

                self._compute_image_features(name, data_root, csv_path, training, num_subj_cls, num_obj_cls, num_action_cls, 
                    image_batch_size, feature_extraction_device, output_size, sampling_ratio, image_mean, image_std, 
                    min_size, max_size)
            
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
        else:
            self._compute_image_features(name, data_root, csv_path, training, num_subj_cls, num_obj_cls, num_action_cls, 
                image_batch_size, feature_extraction_device, output_size, sampling_ratio, image_mean, image_std, 
                min_size, max_size)

    def __getitem__(self, idx):
        return self.spatial_features[idx], self.subject_appearance_features[idx], self.object_appearance_features[idx], self.verb_appearance_features[idx], self.subject_labels[idx], self.object_labels[idx], self.verb_labels[idx],

    def __len__(self):
        return len(self.spatial_features)

    def _cache(self):
        data = {
            'spatial_features': self.spatial_features,
            'subject_labels': self.subject_labels,
            'object_labels': self.object_labels,
            'verb_labels': self.verb_labels,
            'subject_appearance_features': self.subject_appearance_features,
            'object_appearance_features': self.object_appearance_features,
            'verb_appearance_features': self.verb_appearance_features,
        }

        with open('dataset.pkl', 'wb') as f:
            pickle.dump(data, f)

        print('dataset was cached to disk.')

    def _compute_image_features(self, name, data_root, csv_path, training, num_subj_cls, num_obj_cls, num_action_cls, 
        image_batch_size, feature_extraction_device, output_size, sampling_ratio, image_mean, image_std, 
        min_size, max_size):

        with torch.no_grad():
            dataset = DataFactory(
                name = name,
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
                #batch_size = image_batch_size
                batch_size = 1
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
            
            for images, detections, targets in data_loader:
                original_image_sizes = [img.shape[-2:] for img in images]
                images, targets = i_transform(images, targets)
                for det, o_im_s, im_s in zip(
                        detections, original_image_sizes, images.image_sizes
                ):
                    sub_boxes = det['subject_boxes']
                    if sub_boxes[0][0].item() == -1:
                        det['subject_boxes'] = None
                    else:
                        sub_boxes = transform.resize_boxes(sub_boxes, o_im_s, im_s)
                        det['subject_boxes'] = sub_boxes
                    
                    obj_boxes = det['object_boxes']
                    if obj_boxes[0][0].item() == -1:
                        det['object_boxes'] = None
                    else:
                        obj_boxes = transform.resize_boxes(obj_boxes, o_im_s, im_s)
                        det['object_boxes'] = obj_boxes
                
                features = backbone(images.tensors.to(feature_extraction_device))
                features = {k: v.cpu() for k, v in features.items()}

                image_shapes = images.image_sizes
                
                for b_idx, detection in enumerate(detections):
                    if targets is None:
                        subject_label = None
                        object_label = None
                        verb_label = None
                    else:
                        subject_label = targets[b_idx]['subject'][0].detach()
                        object_label = targets[b_idx]['object'][0].detach()
                        verb_label = targets[b_idx]['verb'][0].detach()
                        raw_subject_label = subject_label
                        raw_object_label = object_label
                        raw_verb_label = verb_label
                        subject_label = None if raw_subject_label.item() == -1 else raw_subject_label
                        object_label = None if raw_object_label.item() == -1 else raw_object_label
                        verb_label = None if raw_subject_label.item() == -1 else raw_verb_label
                    
                    subject_labels.append(subject_label)
                    object_labels.append(object_label)
                    verb_labels.append(verb_label)
                    
                    if detection['subject_boxes'] is not None and detection['object_boxes'] is not None:
                        s_xmin, s_ymin, s_xmax, s_ymax = detection['subject_boxes'][0]
                        o_xmin, o_ymin, o_xmax, o_ymax = detection['object_boxes'][0]
                        v_xmin = min(s_xmin, o_xmin)
                        v_ymin = min(s_ymin, o_ymin)
                        v_xmax = max(s_xmax, o_xmax)
                        v_ymax = max(s_ymax, o_ymax)
                        verb_box_coords = torch.tensor([[v_xmin, v_ymin, v_xmax, v_ymax]])
                    
                        x, y = torch.meshgrid(
                            torch.arange(1),
                            torch.arange(2)
                        )
                        x = x.flatten()
                        y = y.flatten()
                        coords = torch.cat([detection['subject_boxes'], detection['object_boxes']])
                        box_pair_spatial.append(compute_spatial_encodings(
                            [coords[x]], [coords[y]], [image_shapes[b_idx]]
                        ).detach())

                        subject_box_features.append(box_roi_pool(features, [detection['subject_boxes']], image_shapes).detach())
                        object_box_features.append(box_roi_pool(features, [detection['object_boxes']], image_shapes).detach())
                        verb_box_features.append(box_roi_pool(features, [verb_box_coords], image_shapes).detach())
                    elif detection['subject_boxes'] is not None:
                        verb_box_coords = detection['subject_boxes'].clone().detach()
                        
                        x, y = torch.meshgrid(
                            torch.arange(1),
                            torch.arange(2)
                        )
                        x = x.flatten()
                        y = y.flatten()
                        coords = torch.cat([detection['subject_boxes'], detection['subject_boxes']])
                        box_pair_spatial.append(compute_spatial_encodings(
                            [coords[x]], [coords[y]], [image_shapes[b_idx]]
                        ).detach())
                        
                        # x, y = torch.meshgrid(
                        #     torch.arange(1),
                        #     torch.arange(1)
                        # )
                        # x = x.flatten()
                        # y = y.flatten()
                        # coords = torch.cat([detection['subject_boxes'], []])
                        # box_pair_spatial.append(compute_spatial_encodings(
                        #     [coords[x]], [coords[y]], [image_shapes[b_idx]]
                        # ))
                        
                        subject_box_features.append(box_roi_pool(features, [detection['subject_boxes']], image_shapes).detach())
                        object_box_features.append(None)
                        verb_box_features.append(box_roi_pool(features, [verb_box_coords], image_shapes).detach())
                    elif detection['object_boxes'] is not None:
                        box_pair_spatial.append(None)
                        subject_box_features.append(None)
                        object_box_features.append(box_roi_pool(features, [detection['object_boxes']], image_shapes).detach())
                        verb_box_features.append(None)
                    else:
                        box_pair_spatial.append(None)
                        subject_box_features.append(None)
                        object_box_features.append(None)
                        verb_box_features.append(None)
            
            self.spatial_features = box_pair_spatial
            self.subject_labels = subject_labels
            self.object_labels = object_labels
            self.verb_labels = verb_labels
            self.subject_appearance_features = subject_box_features
            self.object_appearance_features = object_box_features
            self.verb_appearance_features = verb_box_features

def list_collate(batch):
    spatial_features = [batch[0] for item in batch]
    subject_appearance_features = [batch[1] for item in batch]
    object_appearance_features = [batch[2] for item in batch]
    verb_appearance_features = [batch[3] for item in batch]
    subject_labels = [batch[4] for item in batch]
    object_labels = [batch[5] for item in batch]
    verb_labels = [batch[6] for item in batch]
    return [spatial_features, subject_appearance_features, object_appearance_features, verb_appearance_features, subject_labels, object_labels, verb_labels]
