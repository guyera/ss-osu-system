# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

from swin_et import swin_et

import torch
from torch.nn import Module

class SideTuningBackbone(Module):
    def __init__(self, backbone):
        super().__init__()
        self._backbone = backbone
        self.device = backbone.device
        # The swin_et is intially None and set only upon the first backbone
        # reset, which should occur just before the first retraining
        self._model = None

    def reset(self):
        # This does NOT reset the underlying backbone, since the backbone should
        # remain frozen during retraining / accommodation finetuning.
        # It only resets the swin_et side network
        self._model = swin_et(num_classes=256).to(self.device)

    def forward(self, x):
        # If retraining has occurred at least once, then the features are a
        # concatenation between the backbone's features and the side network's
        # features. Otherwise, they're just the backbone's features
        if self._model is not None:
            y1 = self._backbone(x)
            y2 = self._model(x)
            x = torch.cat((y1, y2), dim=-1)
        else:
            x = self._backbone(x)
        return x

    def to(self, device):
        super().to(device)
        self._backbone = self._backbone.to(device)
        if self._model is not None:
            self._model = self._model.to(device)
        self.device = device
        return self

    def register_feature_hook(self, hook):
        return self._backbone.register_feature_hook(hook)

    def retrainable_parameters(self):
        # Backbone parameters are not retrainable; only return swin_et's
        # parameters
        return self._model.parameters()

    '''
    Computes the features just from the backbone---not the side network.
    Useful when training only the side network, leaving the backbone frozen.
    '''
    def compute_backbone_features(self, x):
        return self._backbone(x)

    '''
    Computes the features just from the side network---not the backbone.
    Useful when training only the side network, leaving the backbone frozen.
    '''
    def compute_side_features(self, x):
        if self._model is None:
            return RuntimeError(('Cannot compute side features before '
                                 'resetting model for retraining at least once'
                                 '---side network not initialized'))
        return self._model(x)

    def eval_backbone(self):
        self._backbone.eval()

    def train_backbone(self):
        self._backbone.train()

    def eval_side(self):
        self._model.eval()

    def train_side(self):
        self._model.train()
