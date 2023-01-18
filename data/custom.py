import os
import math
import pandas as pd
from PIL import Image
import json
from ast import literal_eval

import torch
from typing import Any, Optional, List, Callable, Tuple
from torch.utils.data import Dataset

from labelmapping import LabelMapper

class StandardTransform:
    """https://github.com/pytorch/vision/blob/master/torchvision/datasets/vision.py"""

    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, inputs: Any, target: Any) -> Tuple[Any, Any]:
        if self.transform is not None:
            inputs = self.transform(inputs)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return inputs, target

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])

    def __repr__(self) -> str:
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += self._format_transform_repr(self.transform,
                                                "Transform: ")
        if self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform,
                                                "Target transform: ")

        return '\n'.join(body)


class CustomDet(Dataset):
    """
    Arguments:
        root(str): csv path
        transform(callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version
        target_transform(callable, optional): A function/transform that takes in the
            target and transforms it
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
    """

    def __init__(self, root: str, csv_path: str, json_path: str,
                n_species_cls: int, n_activity_cls: int,
                label_mapper: LabelMapper,
                transform: Optional[Callable] = None,
                target_transform: Optional[Callable] = None,
                transforms: Optional[Callable] = None,
                image_filter='nonblank') -> None:
        super(CustomDet, self).__init__()
        self._n_species_cls = n_species_cls
        self._n_activity_cls = n_activity_cls
        self._image_filter = image_filter

        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms

        self.root = root

        # Load annotations
        self._load_annotation_and_metadata(csv_path, json_path, label_mapper)

    def __len__(self) -> int:
        """Return the number of images"""
        return len(self._idx)

    def __getitem__(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image

        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "species": None or list[N]
                    "activity" : None or list[N]
        """
        intra_idx = self._idx[i]
        target = dict()
        annot = self._anno[intra_idx]
        target['species'] = annot['species'].clone().detach()
        target['activity'] = annot['activity'].clone().detach()
        target['novelty_type'] = annot['novelty_type'].clone().detach()
        return self._transforms(
            self.load_image(os.path.join(self.root, self._filenames[intra_idx])),
            target
        )

    def label(self, i: int) -> tuple:
        """
        Arguments:
            i(int): Index to an image

        Returns:
            tuple[image, target]: By default, the tuple consists of a PIL image and a
                dict with the following keys:
                    "species": None or list[N]
                    "activity" : None or list[N]
        """
        intra_idx = self._idx[i]
        target = dict()
        annot = self._anno[intra_idx]
        target['species'] = annot['species'].clone().detach()
        target['activity'] = annot['activity'].clone().detach()
        target['novelty_type'] = annot['novelty_type'].clone().detach()
        _, transformed_target = self._transforms(
            None,
            target
        )
        return transformed_target

    def get_detections(self, idx: int):
        det = dict()

        det['boxes'] = self._anno[idx]['boxes']

        return det

    def load_image(self, path: str) -> Image:
        """Load an image as PIL.Image"""
        return Image.open(path).convert('RGB')

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(csvPath=' + repr(self._root)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        return reprstr

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return os.path.join(self.root, self._filenames[self._idx[idx]])

    def _load_annotation_and_metadata(self, df_f: str, json_f: str, label_mapper) -> None:
        """
        Arguments:
            df_f(str): path for csv 
            json_f(str): path for json
        """

        df = pd.read_csv(df_f)
        if self._image_filter == 'nonblank':
            df = df[~(df['agent1_count'].isna())]
        df = df.astype({'activities_id': object})
        df['activities_id'] = df['activities_id'].apply(literal_eval)

        self._filenames = list(df['image_path'])

        with open(json_f) as f:
            box_dict = json.load(f)

        self._anno = self.create_annotation(df, box_dict, label_mapper)

        idx = list(range(len(df)))

        self._idx = idx

    def create_annotation(self, df, box_dict, label_mapper):
        # TODO : Make This Faster
        annots = list()
        
        for i, row in df.iterrows():
            annot = dict()

            species = torch.zeros(self._n_species_cls)
            for species_idx in [1, 2, 3]:
                id_string = f'agent{species_idx}_id'
                count_string = f'agent{species_idx}_count'
                species_id = row[id_string]
                species_count = row[count_string]
                if math.isnan(species_id):
                    break
                mapped_label = label_mapper(int(species_id))
                if mapped_label is not None:
                    species[mapped_label] = species_count
            activities = torch.zeros(self._n_activity_cls, dtype=torch.long)
            activity_ids = row['activities_id']
            activities[activity_ids] = 1

            image_path = row['image_path']
            basename = os.path.basename(image_path)
            boxes = box_dict[basename]

            novelty_type = torch.tensor(row['novelty_type'], dtype=torch.long)
            # There is no novelty type 1, but we want novelty type class labels
            # to be contiguous. Adjust labels appropriately.
            if novelty_type >= 2:
                novelty_type = novelty_type - 1

            annot["boxes"] = boxes
            annot["species"] = species
            annot["activity"] = activities
            annot['novelty_type'] = novelty_type

            annots.append(annot)

        return annots

    def box_count(self, i):
        boxes = self._anno[i]['boxes']
        return len(boxes)

def build_species_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    unique_species_list = []
    unique_species = set()
    for i, row in df.iterrows():
        for species_idx in [1, 2, 3]:
            id_string = f'agent{species_idx}_id'
            species_id = row[id_string]
            if math.isnan(species_id):
                break
            species_id = int(species_id)
            if not species_id in unique_species:
                unique_species.add(species_id)
                unique_species_list.append(species_id)
    return {k: v for v, k in enumerate(unique_species_list)}
