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
            n_species_cls,
            n_activity_cls,
            min_size = 800,
            max_size = 1333,
            image_mean = None,
            image_std = None):
        super().__init__()

        filename = os.path.join(f'{os.path.splitext(csv_path)[0]}_novelty_features.pth')

        self._dataset = DataFactory(
            name=name,
            data_root=data_root,
            n_species_cls=n_species_cls,
            n_activity_cls=n_activity_cls,
            csv_path=csv_path,
            training=training
        )
        self._box_transform = torchvision.transforms.Compose([_ResizePad(224), torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self._i_transform = HOINetworkTransform(
            min_size,
            max_size,
            image_mean,
            image_std
        )

    def __len__(self):
        return len(self.species_labels)

    def __getitem__(self, idx):
        with torch.no_grad():
            image, detection, target = self._dataset[idx]
            original_image_size = image.shape[-2:]
            images, targets = i_transform([image], [target])
            image_tensor = images.tensors[0]
            target = targets[0]
            detection['boxes'] = transform.resize_boxes(
                detection['boxes'],
                original_image_size,
                image_size
            )

            if targets is None:
                species_labels = None
                activity_labels = None
            else:
                species_labels = target['species'].detach()
                activity_labels = target['activity'].detach()

            box_images = []
            for xmin, ymin, xmax, ymax in detection['boxes']
                r_xmin = torch.round(xmin).to(torch.int)
                r_ymin = torch.round(ymin).to(torch.int)
                r_xmax = torch.round(xmax).to(torch.int)
                r_ymax = torch.round(ymax).to(torch.int)

                # Crop out image boxes
                box_images.append(
                    self._box_transform(
                        image_tensor[:, r_ymin : r_ymax, r_xmin : r_xmax]
                    )
                )
            box_images = torch.stack(box_images, dim=0)
            whole_image = self._box_transform(image_tensor)

            return species_labels, activity_labels, box_images, whole_image
