import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import pocket
import random

from typing import Any, Optional, List, Callable, Tuple
from pocket.data import ImageDataset, DataSubset
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

class CustomDetSubset(DataSubset):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def filename(self, idx: int) -> str:
        """Override: return the image file name in the subset"""
        return self._filenames[self._idx[self.pool[idx]]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Override: return the size (width, height) of an image in the subset"""
        return self._image_sizes[self._idx[self.pool[idx]]]

    @property
    def anno_interaction(self) -> List[int]:
        """Override: Number of annotated box pairs for each interaction class"""
        num_anno = [0 for _ in range(self.num_interation_cls)]
        intra_idx = [self._idx[i] for i in self.pool]
        for idx in intra_idx:
            for hoi in self._anno[idx]['hoi']:
                num_anno[hoi] += 1
        return num_anno

    @property
    def anno_object(self) -> List[int]:
        """Override: Number of annotated box pairs for each object class"""
        num_anno = [0 for _ in range(self.num_object_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[1]] += anno_interaction[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """Override: Number of annotated box pairs for each action class"""
        num_anno = [0 for _ in range(self.num_action_cls)]
        anno_interaction = self.anno_interaction
        for corr in self._class_corr:
            num_anno[corr[2]] += anno_interaction[corr[0]]
        return num_anno


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

    def __init__(self, root: str,
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

        # Load annotations
        self._load_annotation_and_metadata(root)

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
                    "boxes_h": list[list[4]]
                    "boxes_o": list[list[4]]
                    "hoi":: list[N]
                    "verb": list[N]
                    "object": list[N]
                    "subject" : list[N]
        """
        intra_idx = self._idx[i]
        return self._transforms(
            self.load_image(self._filenames[intra_idx]),
            self._anno[intra_idx]
        )
    
    def get_detections(self, idx: int):
        det = dict()
        
        det["object_boxes"] = self._anno[idx]['boxes_o']
        det["subject_boxes"] = self._anno[idx]['boxes_h']
        det["object_labels"] = self._anno[idx]["object"]
        det["subject_labels"] = self._anno[idx]["subject"]
        
        return det

    def load_image(self, path: str) -> Image: 
        """Load an image as PIL.Image"""
        return Image.open(path)

    def __repr__(self) -> str:
        """Return the executable string representation"""
        reprstr = self.__class__.__name__ + '(csvPath=' + repr(self._root)
        # reprstr += ', anno_file='
        # reprstr += repr(self._anno_file)
        reprstr += ')'
        # Ignore the optional arguments
        return reprstr

    def __str__(self) -> str:
        """Return the readable string representation"""
        reprstr = 'Dataset: ' + self.__class__.__name__ + '\n'
        reprstr += '\tNumber of images: {}\n'.format(self.__len__())
        # reprstr += '\tImage directory: {}\n'.format(self._root)
        # reprstr += '\tAnnotation file: {}\n'.format(self._root)
        return reprstr

    @property
    def annotations(self) -> List[dict]:
        return self._anno

    @property
    def class_corr(self) -> List[Tuple[int, int, int]]:
        """
        Class correspondence matrix in zero-based index
        [
            [hoi_idx, obj_idx, verb_idx],
            ...
        ]

        Returns:
            list[list[3]]
        """
        return self._class_corr.copy()

    @property
    def object_n_verb_to_interaction(self) -> List[list]:
        """
        The interaction classes corresponding to an object-verb pair

        CustomDet.object_n_verb_to_interaction[obj_idx][verb_idx] gives interaction class
        index if the pair is valid, None otherwise

        Returns:
            list[list[117]]
        """
        lut = np.full([self.num_object_cls, self.num_action_cls], None)
        for i, j, k in self._class_corr:
            lut[j, k] = i
        return lut.tolist()

    @property
    def object_to_interaction(self) -> List[list]:
        """
        The interaction classes that involve each object type

        Returns:
            list[list]
        """
        obj_to_int = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_int[corr[1]].append(corr[0])
        return obj_to_int

    @property
    def object_to_verb(self) -> List[list]:
        """
        The valid verbs for each object type

        Returns:
            list[list]
        """
        obj_to_verb = [[] for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            obj_to_verb[corr[1]].append(corr[2])
        return obj_to_verb

    @property
    def anno_interaction(self) -> List[int]:
        """
        Number of annotated box pairs for each interaction class

        Returns:
            list[600]
        """
        return self._num_anno.copy()

    @property
    def anno_object(self) -> List[int]:
        """
        Number of annotated box pairs for each object class

        Returns:
            list[80]
        """
        num_anno = [0 for _ in range(self.num_object_cls)]
        for corr in self._class_corr:
            num_anno[corr[1]] += self._num_anno[corr[0]]
        return num_anno

    @property
    def anno_action(self) -> List[int]:
        """
        Number of annotated box pairs for each action class

        Returns:
            list[117]
        """
        num_anno = [0 for _ in range(self.num_action_cls)]
        for corr in self._class_corr:
            num_anno[corr[2]] += self._num_anno[corr[0]]
        return num_anno

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

    @property
    def interactions(self) -> List[str]:
        """
        Combination of verbs and objects

        Returns:
            list[str]
        """
        return [self._verbs[j] + ' ' + self.objects[i]
                for _, i, j in self._class_corr]

    def split(self, ratio: float) -> Tuple[CustomDetSubset, CustomDetSubset]:
        """
        Split the dataset according to given ratio

        Arguments:
            ratio(float): The percentage of training set between 0 and 1
        Returns:
            train(Dataset)
            val(Dataset)
        """
        perm = np.random.permutation(len(self._idx))
        n = int(len(perm) * ratio)
        return CustomDetSubset(self, perm[:n]), CustomDetSubset(self, perm[n:])

    def filename(self, idx: int) -> str:
        """Return the image file name given the index"""
        return self._filenames[self._idx[idx]]

    def image_size(self, idx: int) -> Tuple[int, int]:
        """Return the size (width, height) of an image"""
        return self._image_sizes[self._idx[idx]]

    def _load_annotation_and_metadata(self, f: dict) -> None:
        """
        Arguments:
            f(str): path for csv 
        """

        df = pd.read_csv(f)
        
        self._filenames = list(df['new_image_path'].unique())
        self._objects = list(df['object_name'].unique())
        self._subjects= list(df['subject_name'].unique())
        self._verbs = list(df['verb_name'].unique())

        self._anno = self.create_annotation(df, self._filenames)
        
        self._class_corr = self.create_correspondence(df, self._objects, self._verbs)
        self._image_sizes = self.create_sizes(df)

        self.num_object_cls = len(self._objects)
        self.num_subject_cls = len(self._subjects)
        self.num_interation_cls = len(self._class_corr)
        self.num_action_cls = len(self._verbs)

        idx = list(range(len(self._filenames)))
    
        self._idx = idx

    def create_annotation(self, df, imnames):
        annots = list()
        for imname in imnames:
            annot = dict()
            temp = df[df["new_image_path"] == imname]

            boxes_h = list()
            boxes_o = list()
            objects = list()
            subjects = list()
            verbs = list()
            for i, row in temp.iterrows():
                boxes_h.append([row["subject_xmin"], row["subject_ymin"], row["subject_xmax"], row["subject_ymax"]])
                boxes_o.append([row["object_xmin"], row["object_ymin"], row["object_xmax"], row["object_ymax"]])
                objects.append(row["object_id"])
                subjects.append(row["subject_id"])
                verbs.append(row["verb_id"])
                
            annot["boxes_h"] = boxes_h
            annot["boxes_o"] = boxes_o
            annot["object"] = objects
            annot["subject"] = subjects
            annot["verb"] = verbs

            annots.append(annot)

        return annots
    
    def create_correspondence(self, df, objects, verbs):
        corr = df.groupby(['object_name','verb_name']).size().reset_index().rename(columns={0:'count'})
        corr["object_id"] = corr['object_name'].apply(lambda x : objects.index(x))
        corr["verb_id"] = corr['verb_name'].apply(lambda x : verbs.index(x))

        corr['idx'] = range(len(corr))

        return corr[['idx', 'object_id', 'verb_id']].values.tolist()
    
    def create_sizes(self, df):
        sizes = df.groupby(['new_image_path','image_width', 'image_height']).size().reset_index().rename(columns={0:'count'})
        return sizes[['image_width', 'image_height']].values.tolist()



    