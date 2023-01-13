from abc import ABC, abstractmethod
import os

import torch
from torchvision.transforms import\
    Normalize as TorchNormalize,\
    RandAugment as TorchRandAugment
from torchvision.transforms.functional import resize, pad

class NamedTransform(ABC):
    def path(self):
        return NotImplemented


class Compose(NamedTransform):
    def __init__(self, transforms):
        self._transforms = transforms

    def __call__(self, x):
        for t in self._transforms:
            x = t(x)
        return x

    def path(self):
        return os.path.join(*([t.path() for t in self._transforms]))


class Normalize(NamedTransform):
    def __init__(self, mean, std):
        self._mean = mean
        self._std = std
        self._transform = TorchNormalize(mean, std)

    def __call__(self, x):
        return self._transform(x)

    def path(self):
        return 'normalized'


class RandAugment(NamedTransform):
    def __init__(self, *args, **kwargs):
        self._transform = TorchRandAugment(*args, **kwargs)

    def __call__(self, x):
        return self._transform((x * 256).to(torch.uint8)).to(torch.float) / 256

    def path(self):
        return 'randaugment'


class RandomHorizontalFlip(NamedTransform):
    def __init__(self, *args, **kwargs):
        self._tarnsform = TorchRandomHorizontalFlip(*args, **kwargs)

    def __call__(self, x):
        return self._transform(x)

    def path(self):
        return 'horizontal-flip'


class ResizePad(NamedTransform):
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
        resized_x = resize(x, new_size)
        
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
            padded_x = pad(resized_x, [left_padding, top_padding, right_padding, bottom_padding])
        else:
            # The new minimum side length is equal to the new maximum side
            # length, i.e. resized_x is already a square and doesn't need to
            # be padded.
            padded_x = resized_x

        return padded_x

    def path(self):
        return f'resizepad={self.size}'
