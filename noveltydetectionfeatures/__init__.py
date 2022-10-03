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
import torchvision

class ResizePad:
    def __init__(self, size):
        self.size = size
    
    def __call__(self, x):
        # Determine which side length (height or width) is smaller and bigger
        if x.shape[-2] > x.shape[-1]:
            min_side_length_idx = -1
            max_side_length_idx = -2
        else:
            min_side_length_idx = -2
            max_side_length_idx = -1
        
        min_side_length = x.shape[min_side_length_idx]
        max_side_length = x.shape[max_side_length_idx]

        # Determine the scale factor such that the maximum side length is equal
        # to self.size
        scale_factor = float(self.size) / max_side_length

        # Determine the new minimum side length by applying that same scale
        # factor
        new_min_side_length = int(min_side_length * scale_factor)
        
        # Construct the new size list
        if min_side_length_idx == -2:
            new_size = [new_min_side_length, self.size]
        else:
            new_size = [self.size, new_min_side_length]
        
        # Resize x to the new size
        resized_x = torchvision.transforms.functional.resize(x, new_size)
        
        # X needs to be padded to square. Determine the amount of padding
        # necessary, i.e. the difference between the max and min side lengths.
        padding = self.size - new_min_side_length
        if padding != 0:
            if min_side_length_idx == -2:
                # Height is smaller than width. Vertical padding is necessary.
                left_padding = 0
                top_padding = int(padding / 2)
                right_padding = 0
                bottom_padding = padding - top_padding
            else:
                # Width is smaller than height. Horizontal padding is necessary.
                left_padding = int(padding / 2)
                top_padding = 0
                right_padding = padding - left_padding
                bottom_padding = 0
            # Pad resized_x to square
            padded_x = torchvision.transforms.functional.pad(resized_x, [left_padding, top_padding, right_padding, bottom_padding])
        else:
            # The new minimum side length is equal to the new maximum side
            # length, i.e. resized_x is already a square and doesn't need to
            # be padded.
            padded_x = resized_x

        return padded_x

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
            training,
            image_batch_size,
            backbone,
            output_size = 7,
            sampling_ratio = 2,
            min_size = 800,
            max_size = 1333,
            image_mean = None,
            image_std = None,
            feature_extraction_device = 'cpu',
            cache_to_disk = False):
        super().__init__()
        
        filename = os.path.join(f'{os.path.splitext(csv_path)[0]}_novelty_features.pth')
        
        if cache_to_disk:
            if not os.path.exists(filename):
                print('Image features have not yet been computed. Computing image features...')

                self._compute_image_features(name, data_root, csv_path, training, 
                    image_batch_size, feature_extraction_device, output_size, sampling_ratio, image_mean, image_std, 
                    min_size, max_size)
            
                data = {
                    'spatial_features': self.spatial_features,
                    'subject_labels': self.subject_labels,
                    'object_labels': self.object_labels,
                    'verb_labels': self.verb_labels,
                    'subject_roi_features': self.subject_roi_features,
                    'object_roi_features': self.object_roi_features,
                    'verb_roi_features': self.verb_roi_features,
                    'subject_images': self.subject_images,
                    'object_images': self.object_images,
                    'verb_images': self.verb_images
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
                self.subject_roi_features = data['subject_roi_features']
                self.object_roi_features = data['object_roi_features']
                self.verb_roi_features = data['verb_roi_features']
                self.subject_images = data['subject_images']
                self.object_images = data['object_images']
                self.verb_images = data['verb_images']
        else:
            self._compute_image_features(name, data_root, csv_path, training, 
                image_batch_size, feature_extraction_device, output_size, sampling_ratio, image_mean, image_std, 
                min_size, max_size)

    def __getitem__(self, idx):
        return self.spatial_features[idx], self.subject_roi_features[idx], self.object_roi_features[idx], self.verb_roi_features[idx], self.subject_labels[idx], self.object_labels[idx], self.verb_labels[idx] , self.subject_images[idx], self.object_images[idx], self.verb_images[idx]

    def __len__(self):
        return len(self.spatial_features)

    def _compute_image_features(self, name, data_root, csv_path, training, 
        image_batch_size, feature_extraction_device, output_size, sampling_ratio, image_mean, image_std, 
        min_size, max_size):

        with torch.no_grad():
            box_transform = torchvision.transforms.Compose([ResizePad(224), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            dataset = DataFactory(
                name = name,
                data_root = data_root,
                csv_path = csv_path,
                training = training)
            
            data_loader = DataLoader(
                dataset = dataset,
                collate_fn = custom_collate,
                #batch_size = image_batch_size
                batch_size = 1
            )
            
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
            subject_roi_features = list()
            object_roi_features = list()
            verb_roi_features = list()
            subject_images = list()
            object_images = list()
            verb_images = list()
            
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
                
                image_tensors = images.tensors.to(feature_extraction_device)
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
                        r_s_xmin, r_s_ymin, r_s_xmax, r_s_ymax = torch.round(
                            detection['subject_boxes'][0]).to(torch.int)
                        
                        o_xmin, o_ymin, o_xmax, o_ymax = detection['object_boxes'][0]
                        r_o_xmin, r_o_ymin, r_o_xmax, r_o_ymax = torch.round(
                            detection['object_boxes'][0]).to(torch.int)
                        
                        v_xmin = min(s_xmin, o_xmin)
                        v_ymin = min(s_ymin, o_ymin)
                        v_xmax = max(s_xmax, o_xmax)
                        v_ymax = max(s_ymax, o_ymax)

                        r_v_xmin = min(r_s_xmin, r_o_xmin)
                        r_v_ymin = min(r_s_ymin, r_o_ymin)
                        r_v_xmax = max(r_s_xmax, r_o_xmax)
                        r_v_ymax = max(r_s_ymax, r_o_ymax)

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
                        ).detach().to(feature_extraction_device))
                        
                        subject_roi_features.append(None)
                        object_roi_features.append(None)
                        verb_roi_features.append(None)
                        
                        # Extract image boxes via cropping
                        subject_images.append(box_transform(image_tensors[b_idx, :, r_s_ymin: r_s_ymax, r_s_xmin: r_s_xmax]))
                        object_images.append(box_transform(image_tensors[b_idx, :, r_o_ymin: r_o_ymax, r_o_xmin: r_o_xmax]))
                        verb_images.append(box_transform(image_tensors[b_idx, :, r_v_ymin: r_v_ymax, r_v_xmin: r_v_xmax]))
                    elif detection['subject_boxes'] is not None:
                        r_s_xmin, r_s_ymin, r_s_xmax, r_s_ymax = torch.round(
                            detection['subject_boxes'][0]).to(torch.int)
                        
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
                        ).detach().to(feature_extraction_device))
                        
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
                        
                        subject_roi_features.append(None)
                        object_roi_features.append(None)
                        verb_roi_features.append(None)
                        subject_images.append(box_transform(image_tensors[b_idx, :, r_s_ymin: r_s_ymax, r_s_xmin: r_s_xmax]))
                        object_images.append(None)
                        verb_images.append(box_transform(image_tensors[b_idx, :, r_s_ymin: r_s_ymax, r_s_xmin: r_s_xmax]))
                    elif detection['object_boxes'] is not None:
                        r_o_xmin, r_o_ymin, r_o_xmax, r_o_ymax = torch.round(
                            detection['object_boxes'][0]).to(torch.int)
                        
                        box_pair_spatial.append(None)
                        subject_roi_features.append(None)
                        object_roi_features.append(None)
                        verb_roi_features.append(None)
                        subject_images.append(None)
                        object_images.append(box_transform(image_tensors[b_idx, :, r_o_ymin: r_o_ymax, r_o_xmin: r_o_xmax]))
                        verb_images.append(None)
                    else:
                        box_pair_spatial.append(None)
                        subject_roi_features.append(None)
                        object_roi_features.append(None)
                        verb_roi_features.append(None)
                        subject_images.append(None)
                        object_images.append(None)
                        verb_images.append(None)
            
            self.spatial_features = box_pair_spatial
            self.subject_labels = subject_labels
            self.object_labels = object_labels
            self.verb_labels = verb_labels
            self.subject_roi_features = subject_roi_features
            self.object_roi_features = object_roi_features
            self.verb_roi_features = verb_roi_features
            self.subject_images = subject_images
            self.object_images = object_images
            self.verb_images = verb_images

def list_collate(batch):
    spatial_features = [batch[0] for item in batch]
    subject_roi_features = [batch[1] for item in batch]
    object_roi_features = [batch[2] for item in batch]
    verb_roi_features = [batch[3] for item in batch]
    subject_labels = [batch[4] for item in batch]
    object_labels = [batch[5] for item in batch]
    verb_labels = [batch[6] for item in batch]
    subject_images = [batch[7] for item in batch]
    object_images = [batch[8] for item in batch]
    verb_images = [batch[9] for item in batch]
    return [spatial_features, subject_roi_features, object_roi_features, verb_roi_features, subject_labels, object_labels, verb_labels, subject_images, object_images, verb_images]
