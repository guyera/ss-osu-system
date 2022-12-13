import torch
import argparse
import re
import copy
import numpy as np
import pickle
import os

from tqdm import tqdm
from sklearn.neighbors import KernelDensity


def _state_dict(module):
    return {k: v.cpu() for k, v in module.state_dict().items()}

def _load_state_dict(module, state_dict):
    next_param = next(module.parameters())
    if next_param.is_cuda:
        cuda_state_dict = {k: v.to(next_param.device) for k, v in state_dict.items()}
        module.load_state_dict(cuda_state_dict)
    else:
        module.load_state_dict(state_dict)

class ClassifierV2:
    def __init__(
            self,
            bottleneck_dim,
            num_subj_cls,
            num_obj_cls,
            num_action_cls,
            spatial_encoding_dim):
        self.device = 'cpu'
        self.subject_classifier = torch.nn.Linear(bottleneck_dim, num_subj_cls - 1)
        self.object_classifier = torch.nn.Linear(bottleneck_dim, num_obj_cls - 1)
        self.verb_classifier = torch.nn.Linear(bottleneck_dim + spatial_encoding_dim, num_action_cls - 1)
    
    def predict(self, subject_features, object_features, verb_features):
        self.subject_classifier.eval()
        self.object_classifier.eval()
        self.verb_classifier.eval()
        subject_logits = self.subject_classifier(subject_features)
        object_logits = self.object_classifier(object_features)
        verb_logits = self.verb_classifier(verb_features)
        return subject_logits, object_logits, verb_logits

    def predict_score_subject(self, features):
        self.subject_classifier.eval()
        logits = self.subject_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_score_object(self, features):
        self.object_classifier.eval()
        logits = self.object_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_score_verb(self, features):
        self.verb_classifier.eval()
        logits = self.verb_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_subject(self, features):
        self.subject_classifier.eval()
        logits = self.subject_classifier(features)
        return logits

    def predict_object(self, features):
        self.object_classifier.eval()
        logits = self.object_classifier(features)
        return logits

    def predict_verb(self, features):
        self.verb_classifier.eval()
        logits = self.verb_classifier(features)
        return logits

    def score_subject(self, features):
        self.subject_classifier.eval()
        logits = self.subject_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores

    def score_object(self, features):
        self.object_classifier.eval()
        logits = self.object_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores

    def score_verb(self, features):
        self.verb_classifier.eval()
        logits = self.verb_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores
    
    def to(self, device):
        self.device = device
        self.subject_classifier = self.subject_classifier.to(device)
        self.object_classifier = self.object_classifier.to(device)
        self.verb_classifier = self.verb_classifier.to(device)
        return self
    
    def state_dict(self):
        state_dict = {}
        state_dict['subject_classifier'] = _state_dict(self.subject_classifier)
        state_dict['object_classifier'] = _state_dict(self.object_classifier)
        state_dict['verb_classifier'] = _state_dict(self.verb_classifier)
        return state_dict

    def load_state_dict(self, state_dict):
        _load_state_dict(
            self.subject_classifier,
            state_dict['subject_classifier']
        )
        _load_state_dict(
            self.object_classifier,
            state_dict['object_classifier']
        )
        _load_state_dict(
            self.verb_classifier,
            state_dict['verb_classifier']
        )


class ActivationStatisticalModel:
    def __init__(self, model_name):
        self._model_name = model_name
        if model_name == 'resnet':
            self._kde = KernelDensity(kernel='gaussian', bandwidth=5)
        elif model_name == 'swin_t':
            self._kde = KernelDensity(kernel='gaussian', bandwidth=50)
        else:
            raise NotImplementedError()

        self._device = None
        self._v = None

    def forward_hook(self, module, inputs, outputs):
        self._features = outputs

    def compute_features(self, backbone, batch):
        if self._model_name == 'resnet':
            handle = backbone.layer4.register_forward_hook(self.forward_hook)
        else:
            handle = backbone.features[2].register_forward_hook(self.forward_hook)
        backbone(batch)
        handle.remove()
        return self._features.view(self._features.shape[0], -1)

    def pca_reduce(self, v, features, k):
        return torch.matmul(features, v[:, :k])

    def fit(self, features):
        # Fit PCA
        _, _, v = torch.svd(features)
        self._v = v

        # PCA-project features
        projected_features = self.pca_reduce(v, features, 64)

        # Fit self._kde to pca-projected features
        self._kde.fit(projected_features.cpu().numpy())

    def score(self, features):
        # Compute and return negative log likelihood under self._kde
        projected_features = self.pca_reduce(self._v, features, 64)
        return -self._kde.score_samples(projected_features.cpu().numpy())

    def reset(self):
        self._v = None
        if self._model_name == 'resnet':
            self._kde = KernelDensity(kernel='gaussian', bandwidth=5)
        else:
            self._kde = KernelDensity(kernel='gaussian', bandwidth=50)

    def to(self, device):
        self._device = device
        if self._v is not None:
            self._v = self._v.to(device)
        return self

    def state_dict(self):
        sd = {}
        if self._v is not None:
            sd['v'] = self._v.to('cpu')
        else:
            sd['v'] = None
        sd['kde'] = self._kde
        return sd

    def load_state_dict(self, sd):
        self._v = sd['v']
        self._kde = sd['kde']
        if self._device is not None and self._v is not None:
            self._v = self._v.to(self._device)


class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.temperature


class ConfidenceCalibrator:
    def __init__(self):
        self.device = 'cpu'
        self.reset()

    def reset(self):
        self.subject_calibrator = TemperatureScaler().to(self.device)
        self.object_calibrator = TemperatureScaler().to(self.device)
        self.verb_calibrator = TemperatureScaler().to(self.device)

    def calibrate_subject(self, logits):
        calibrated_logits = self.subject_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate_object(self, logits):
        calibrated_logits = self.object_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate_verb(self, logits):
        calibrated_logits = self.verb_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate(self, subject_logits, object_logits, verb_logits):
        calibrated_subject_logits = self.subject_calibrator(subject_logits)
        calibrated_object_logits = self.object_calibrator(object_logits)
        calibrated_verb_logits = self.verb_calibrator(verb_logits)
        
        subject_predictions = torch.nn.functional.softmax(calibrated_subject_logits, dim = 1)
        object_predictions = torch.nn.functional.softmax(calibrated_object_logits, dim = 1)
        verb_predictions = torch.nn.functional.softmax(calibrated_verb_logits, dim = 1)
        
        return subject_predictions, object_predictions, verb_predictions
    
    def to(self, device):
        self.device = device
        self.subject_calibrator = self.subject_calibrator.to(device)
        self.object_calibrator = self.object_calibrator.to(device)
        self.verb_calibrator = self.verb_calibrator.to(device)
        return self
    
    def state_dict(self):
        state_dict = {}
        state_dict['subject_calibrator'] =\
            _state_dict(self.subject_calibrator)
        state_dict['object_calibrator'] =\
            _state_dict(self.object_calibrator)
        state_dict['verb_calibrator'] =\
            _state_dict(self.verb_calibrator)
        return state_dict

    def load_state_dict(self, state_dict):
        _load_state_dict(
            self.subject_calibrator,
            state_dict['subject_calibrator']
        )
        _load_state_dict(
            self.object_calibrator,
            state_dict['object_calibrator']
        )
        _load_state_dict(
            self.verb_calibrator,
            state_dict['verb_calibrator']
        )
