import pickle
import os
import torch
import pocket
from utils import binary_focal_loss
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection import transform
from models.scg.scg import HOINetworkTransform
import pocket.models as models
from data.data_factory import DataFactory
from torch.utils.data import DataLoader, DistributedSampler
from utils import custom_collate
import torchvision

class _ResizePad:
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

class BoxImageDataset(torch.utils.data.Dataset):
    """
    Params:
        name: As in data.data_factory.DataFactory()
        data_root: As in data.data_factory.DataFactory()
        csv_path: As in data.data_factory.DataFactory()
        training: As in data.data_factory.DataFactory()
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
            output_size = 7,
            sampling_ratio = 2,
            min_size = 800,
            max_size = 1333,
            image_mean = None,
            image_std = None,
            cache_to_disk = False):
        super().__init__()

        filename = os.path.join(f'{os.path.splitext(csv_path)[0]}_novelty_features.pth')

        if cache_to_disk:
            if not os.path.exists(filename):
                print('Image features have not yet been computed. Computing image features...')

                self._compute_image_features(name, data_root, csv_path, training, 
                    output_size, sampling_ratio, image_mean, image_std, 
                    min_size, max_size)

                data = {
                    'species_labels': self.species_labels,
                    'activity_labels': self.activity_labels,
                    'box_images': self.box_images,
                    'whole_images': self.whole_images
                }

                with open(filename, 'wb') as f:
                    pickle.dump(data, f)
            
            else:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
            
                self.species_labels = data['species_labels']
                self.activity_labels = data['activity_labels']
                self.box_images = data['box_images']
                self.whole_images = data['whole_images']
        else:
            self._compute_image_features(name, data_root, csv_path, training, 
                output_size, sampling_ratio, image_mean, image_std, 
                min_size, max_size)

    def __getitem__(self, idx):
        return self.species_labels[idx], self.activity_labels[idx], self.box_images[idx], self.whole_images[idx]

    def __len__(self):
        return len(self.species_labels)

    def _compute_image_features(self, name, data_root, csv_path, training, 
        output_size, sampling_ratio, image_mean, image_std, 
        min_size, max_size):

        with torch.no_grad():
            box_transform = torchvision.transforms.Compose([_ResizePad(224), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
            dataset = DataFactory(
                name = name,
                data_root = data_root,
                csv_path = csv_path,
                training = training
            )

            data_loader = DataLoader(
                dataset = dataset,
                collate_fn = custom_collate,
                batch_size = 1
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
            
            species_labels = list()
            activity_labels = list()
            box_images = list()
            whole_images = list()

            for images, detections, targets in data_loader:
                original_image_sizes = [img.shape[-2:] for img in images]
                images, targets = i_transform(images, targets)
                for det, o_im_s, im_s in zip(
                        detections, original_image_sizes, images.image_sizes
                ):
                    det['boxes'] = transform.resize_boxes(
                        det['boxes'],
                        o_im_s,
                        im_s
                    )

                image_tensors = images.tensors
                image_shapes = images.image_sizes
                
                for b_idx, detection in enumerate(detections):
                    if targets is None:
                        img_species_labels = None
                        img_activity_labels = None
                    else:
                        img_species_labels = targets[b_idx]['species'].detach()
                        img_activity_labels = targets[b_idx]['activity'].detach()

                    species_labels.append(img_species_labels)
                    activity_labels.append(img_activity_labels)

                    img_boxes = []
                    for xmin, ymin, xmax, ymax in detection['boxes']
                        r_xmin, r_ymin, r_xmax, r_ymax = torch.round(
                            box_detection
                        ).to(torch.int)

                        # Extract image boxes via cropping
                        img_boxes.append(box_transform(image_tensors[b_idx, :, r_ymin : r_ymax, r_xmin : r_xmax]))
                    box_images.append(torch.stack(img_boxes, dim=0))
                    whole_images.append(box_transform(image_tensors[b_idx]))

            self.species_labels = species_labels
            self.activity_labels = activity_labels
            self.box_images = box_images
            self.whole_images = whole_images
