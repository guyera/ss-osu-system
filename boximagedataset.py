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
import torchvision

class BoxImageDataset(torch.utils.data.Dataset):
    class LabelDataset(torch.utils.data.Dataset):
        def __init__(self, dataset):
            self._dataset = dataset

        def __len__(self):
            return len(self._dataset)

        def __getitem__(self, idx):
            with torch.no_grad():
                target = self._dataset.label(idx)
                species_labels = target['species'].detach()
                activity_labels = target['activity'].detach()
                novelty_type_labels = target['novelty_type'].detach()

                return species_labels,\
                    activity_labels,\
                    novelty_type_labels

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
            label_mapper,
            min_size = 800,
            max_size = 1333,
            image_mean = None,
            image_std = None,
            box_transform=None):
        super().__init__()

        filename = os.path.join(f'{os.path.splitext(csv_path)[0]}_novelty_features.pth')

        self._dataset = DataFactory(
            name=name,
            data_root=data_root,
            n_species_cls=n_species_cls,
            n_activity_cls=n_activity_cls,
            label_mapper=label_mapper,
            csv_path=csv_path,
            training=training
        )
        self._box_transform = box_transform

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
        return len(self._dataset)

    def __getitem__(self, idx):
        with torch.no_grad():
            image, detection, target = self._dataset[idx]
            original_image_size = image.shape[-2:]
            images, targets = self._i_transform([image], [target])
            image_tensor = images.tensors[0]
            image_size = images.image_sizes[0]
            target = targets[0]
            detection['boxes'] = transform.resize_boxes(
                detection['boxes'],
                original_image_size,
                image_size
            )

            species_labels = target['species'].detach()
            activity_labels = target['activity'].detach()
            novelty_type_labels = target['novelty_type'].detach()

            box_images = []
            for xmin, ymin, xmax, ymax in detection['boxes']:
                r_xmin = torch.round(xmin).to(torch.int)
                r_ymin = torch.round(ymin).to(torch.int)
                r_xmax = torch.round(xmax).to(torch.int)
                r_ymax = torch.round(ymax).to(torch.int)

                # Crop out image boxes
                box_image = image_tensor[:, r_ymin : r_ymax, r_xmin : r_xmax]
                if self._box_transform is not None:
                    box_image = self._box_transform(box_image)
                box_images.append(raw_box_image)
            box_images = torch.stack(box_images, dim=0)
            whole_image = self._box_transform(image_tensor)

            return species_labels,\
                activity_labels,\
                novelty_type_labels,\
                box_images,\
                whole_image

    def label_dataset(self):
        return self.LabelDataset(self._dataset)

    def box_count(self, i):
        return self._dataset.box_count(i)
