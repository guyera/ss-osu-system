import os
import pandas as pd
from PIL import Image
import json

from typing import Any, Optional, List, Callable, Tuple
from torch.utils.data import Dataset


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
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 transforms: Optional[Callable] = None) -> None:
        super(CustomDet, self).__init__()

        if transforms is None:
            self._transforms = StandardTransform(transform, target_transform)
        elif transform is not None or target_transform is not None:
            print("WARNING: Argument transforms is given, transform/target_transform are ignored.")
            self._transforms = transforms
        else:
            self._transforms = transforms


        self.root = root

        # Load annotations
        self._load_annotation_and_metadata(csv_path, json_path)

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
                    "boxes": list[list[4]]
                    "species": None or list[N]
                    "activity" : None or list[N]
        """
        intra_idx = self._idx[i]
        return self._transforms(
            self.load_image(os.path.join(self.root, self._filenames[intra_idx])),
            self._anno[intra_idx]
        )

    def get_detections(self, idx: int):
        det = dict()

        det["boxes"] = self._anno[idx]['boxes']

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

    # def image_size(self, idx: int) -> Tuple[int, int]:
    #     """Return the size (width, height) of an image"""
    #     return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, df_f: str, json_f: str) -> None:
        """
        Arguments:
            df_f(str): path for csv 
            json_f(str): path for json
        """

        df = pd.read_csv(df_f)

        self._filenames = list(df['image_path'])

        box_dict = json.load(json_f)

        self._anno = self.create_annotation(df, box_dict)

        # self._image_sizes = self.create_sizes(df)

        idx = list(range(len(df)))

        self._idx = idx

    def create_annotation(self, df, box_dict):
        # TODO : Make This Faster
        annots = list()
        
        for i, row in df.iterrows():
            annot = dict()

            boxes = list()
            species = list()
            activities = list()
            
            species.append(row['species'])
            activities.append(row['activity'])
            image_path = row['image_path']
            basename = os.path.basename(image_path)
            img_boxes = box_dict[basename]
            boxes.append(box_dict[basename])

            annot["boxes"] = boxes
            annot["species"] = species
            annot["activity"] = activities

            annots.append(annot)

        return annots
