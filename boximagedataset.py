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

from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image

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
            box_transform=None,
            cache_dir=None,
            write_cache=False):
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

        self._box_transform = box_transform

        if box_transform is not None:
            cache_dir =\
                os.path.join(cache_dir, box_transform.path())
        self._cache_dir = cache_dir
        self._write_cache = write_cache

    def __len__(self):
        return len(self._dataset)

    def _load_cached_data(self, cache_dir):
        label_cache_file =\
            os.path.join(cache_dir, f'labels.pth')
        t = torch.load(label_cache_file)
        species_labels = t[0]
        activity_labels = t[1]
        novelty_type_labels = t[2]

        whole_image_cache_file =\
            os.path.join(cache_dir, 'whole_image.JPG')
        pil_whole_image = Image.open(whole_image_cache_file)
        whole_image = to_tensor(pil_whole_image)

        box_images_cache_dir = os.path.join(cache_dir, 'box_images')
        _, _, box_image_files = next(os.walk(box_images_cache_dir))
        box_images = []
        for box_image_file in box_image_files:
            box_image_cache_file =\
                os.path.join(box_images_cache_dir, box_image_file)
            pil_box_image = Image.open(box_image_cache_file)
            box_image = to_tensor(pil_box_image)
            box_images.append(box_image)
        box_images = torch.stack(box_images, dim=0)

        return species_labels,\
            activity_labels,\
            novelty_type_labels,\
            box_images,\
            whole_image

    def __getitem__(self, idx):
        with torch.no_grad():
            if self._cache_dir is not None:
                cur_cache_dir = os.path.join(self._cache_dir, f'{idx}')
                if os.path.exists(cur_cache_dir):
                    return self._load_cached_data(cur_cache_dir)

            # If we made it this far, then the data could not be loaded from
            # the cache. Proceed to load it normally.
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
                box_images.append(box_image)
            box_images = torch.stack(box_images, dim=0)
            whole_image = self._box_transform(image_tensor)

            # Cache data if configured to do so
            if self._cache_dir is not None and self._write_cache:
                cur_cache_dir = os.path.join(self._cache_dir, f'{idx}')
                if not os.path.exists(cur_cache_dir):
                    os.makedirs(cur_cache_dir, exist_ok=True)
                    label_cache_file =\
                        os.path.join(cur_cache_dir, f'labels.pth')
                    t = (
                        species_labels,
                        activity_labels,
                        novelty_type_labels
                    )
                    torch.save(t, label_cache_file)

                    whole_image_cache_file =\
                        os.path.join(cur_cache_dir, 'whole_image.JPG')
                    pil_whole_image = to_pil_image(whole_image)
                    pil_whole_image.save(whole_image_cache_file)

                    box_images_cache_dir = os.path.join(cur_cache_dir, 'box_images')
                    os.makedirs(box_images_cache_dir, exist_ok=True)
                    for box_idx, box_image in enumerate(box_images):
                        box_image_cache_file =\
                            os.path.join(box_images_cache_dir, f'{box_idx}.JPG')
                        pil_box_image = to_pil_image(box_image)
                        pil_box_image.save(box_image_cache_file)

            return species_labels,\
                activity_labels,\
                novelty_type_labels,\
                box_images,\
                whole_image

    def label_dataset(self):
        return self.LabelDataset(self._dataset)

    def box_count(self, i):
        return self._dataset.box_count(i)

    def commit_cache(self):
        if not os.path.exists(self._cache_dir):
            print('Caching data...')
            for _ in self:
                pass
