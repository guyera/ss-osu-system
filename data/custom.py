import os
import pandas as pd
from PIL import Image

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

    def __init__(self, root: str, csv_path: str,
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
        self._load_annotation_and_metadata(csv_path)

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
                    "boxes_s": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
                    "subject" : list[N]
        """
        intra_idx = self._idx[i]
        return self._transforms(
            self.load_image(os.path.join(self.root, self._filenames[intra_idx])),
            self._anno[intra_idx]
        )

    def get_detections(self, idx: int):
        det = dict()

        det["object_boxes"] = self._anno[idx]['boxes_o']
        det["subject_boxes"] = self._anno[idx]['boxes_s']
        det["object_labels"] = self._anno[idx]["object"]
        det["subject_labels"] = self._anno[idx]["subject"]

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

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def objects(self) -> List[str]:
        """
        Object names

        Returns:
            list[str]
        """
        return self._objects.copy()

    @property
    def verbs(self) -> List[str]:
        """
        Verb (action) names

        Returns:
            list[str]
        """
        return self._verbs.copy()

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return os.path.join(self.root, self._filenames[self._idx[idx]])

    # def image_size(self, idx: int) -> Tuple[int, int]:
    #     """Return the size (width, height) of an image"""
    #     return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: str) -> None:
        """
        Arguments:
            f(str): path for csv 
        """

        df = pd.read_csv(f)

        self._filenames = list(df['new_image_path'])
        self._objects = list(df['object_name'].unique())
        self._subjects = list(df['subject_name'].unique())
        self._verbs = list(df['verb_name'].unique())

        self._anno = self.create_annotation(df)

        # self._image_sizes = self.create_sizes(df)

        idx = list(range(len(df)))

        self._idx = idx

    def create_annotation(self, df):
        # TODO : Make This Faster
        annots = list()
        
        for i, row in df.iterrows():
            annot = dict()

            boxes_s = list()
            boxes_o = list()
            objects = list()
            subjects = list()
            verbs = list()
            
            boxes_s.append([row["subject_xmin"], row["subject_ymin"], row["subject_xmax"], row["subject_ymax"]])
            boxes_o.append([row["object_xmin"], row["object_ymin"], row["object_xmax"], row["object_ymax"]])
            objects.append(row["object_id"])
            subjects.append(row["subject_id"])
            verbs.append(row["verb_id"])

            annot["boxes_s"] = boxes_s
            annot["boxes_o"] = boxes_o
            annot["object"] = objects
            annot["subject"] = subjects
            annot["verb"] = verbs

            annots.append(annot)

        return annots

    def create_correspondence(self, df, objects, verbs):
        corr = df.groupby(['object_name', 'verb_name']).size().reset_index().rename(columns={0: 'count'})
        corr["object_id"] = corr['object_name'].apply(lambda x: objects.index(x))
        corr["verb_id"] = corr['verb_name'].apply(lambda x: verbs.index(x))

        corr['idx'] = range(len(corr))

        return corr[['idx', 'object_id', 'verb_id']].values.tolist()

    # def create_sizes(self, df):
    #     return df[['image_width', 'image_height']].values.tolist()