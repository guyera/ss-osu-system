import torch
import argparse
import re
import copy
import numpy as np
import pickle
import os

from tqdm import tqdm
from sklearn.neighbors import KernelDensity

"""
Description: Gets the indices of the datapoints in the given dataset with
    any of the given labels.

Parameters:
    dataset: The dataset into which to index.
    labels: The labels for which to acquire indices.
"""
def get_indices_of_labels(dataset, labels):
    indices = []
    for idx, (_, label) in enumerate(dataset):
        if label in labels:
            indices.append(idx)
    return indices

"""
Description: Provides functions for generating, saving, loading, and using class
    splits to bipartition datasets into "in-distribution" and
    "out-of-distribution" subsets according to class label.
"""
class ClassSplit:
    """
    Parameters:
        id_labels: List of in-distribution class labels.
        ood_labels: List of out-of-distribution class labels.
    """
    def __init__(self, id_labels, ood_labels):
        self._id_labels = id_labels
        self._ood_labels = ood_labels

    """
    Description: Partitions a given dataset into an in-distribution dataset and
        an out-of-distribution dataset according to the class labels of each
        data point.

    Parameters:
        dataset: The dataset to partition.
    """
    def split_dataset(self, dataset):
        id_indices = get_indices_of_labels(
            dataset,
            self._id_labels)
        ood_indices = get_indices_of_labels(
            dataset,
            self._ood_labels)
        
        id_dataset = torch.utils.data.Subset(
            dataset,
            id_indices
        )
        ood_dataset = torch.utils.data.Subset(
            dataset,
            ood_indices
        )

        return id_dataset, ood_dataset

    def id_labels(self):
        return copy.deepcopy(self._id_labels)
    
    def ood_labels(self):
        return copy.deepcopy(self._ood_labels)

"""
Description: Generates a random class split.

Parameters:
    labels: An enumerable of all class labels.
    num_id: The number of classes to hold out as in-distribution.
"""
def generate_class_split(labels, num_id: int):
    # Copy labels to ood_labels
    ood_labels = labels[:]
    id_labels = []
    
    # Remove random labels from ood_labels and put them in id_labels
    for i in range(num_id):
        idx = np.random.randint(0, len(ood_labels))
        label = ood_labels.pop(idx)
        id_labels.append(label)
    
    # Construct and return ClassSplit
    return ClassSplit(id_labels, ood_labels)

"""
Description: A dataset which wraps around another dataset, returning the same
    data points after remapping their labels. Used to remap labels to [0, N-1]
    after splitting datasets by class.
"""
class LabelMappingDataset(torch.utils.data.Dataset):
    """
    Parameters:
        dataset: The dataset from which to retrieve data points prior to
            remapping labels.
        labels: An enumerable of labels of the form labels[i] = j. Labels of
            value j are remapped to value i.
    """
    def __init__(self, dataset, labels):
        self._dataset = dataset
    
        label_mapping = {}
        new_label = 0
        for label in labels:
            label_mapping[label] = new_label
            new_label += 1
        self._label_mapping = label_mapping

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        data, label = self._dataset.__getitem__(index)
        label = int(label.item())
        label = self._label_mapping[label]
        return data, label

"""
Description: A dataset which wraps around another dataset, returning the same
    data points after applying a transformation to them. Used to apply
    transformations separately to training and testing datasets after splitting
    them by in-distribution and out-of-distribution classes.
"""
class TransformingDataset(torch.utils.data.Dataset):
    """
    Parameters:
        dataset: Underlying dataset from which to retrieve data points prior
            to transformation.
        transform: Transformation to apply to data.
        target_transform: Target transformation to apply to data targets.
    """
    def __init__(self, dataset, transform = None, target_transform = None):
        self._dataset = dataset
        self._transform = transform
        self._target_transform = target_transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        res = self._dataset.__getitem__(index)
        item = res[0]
        target = res[1]
        if self._transform is not None:
            item = self._transform(item)
        if self._target_transform is not None:
            target = self._target_transform(target)
        return (item, target)

"""
Description: Target transform which converts all labels to a fixed constant.
    Used to convert in-distribution labels to zeros and out-of-distribution
    labels to ones.
"""
class FixedTargetTransform(object):
    """
    Parameters:
        target: Fixed target to which all labels are converted.
    """
    def __init__(self, target):
        self._target = target

    def __call__(self, prev_target):
        return self._target

"""
Description: Target transform which converts shifts all labels up by 1.
    Used to convert standard labels into labels where label 0 is considered
    the novelty label.
"""
class UpshiftTargetTransform(object):
    def __call__(self, prev_target):
        return prev_target + 1

"""
Description: Combines an in-distribution and out-of-distribution dataset into a
    single dataset, remapping labels to zeros for in-distribution points and
    ones for out-of-distribution points. This can be used for evaluation to
    avoid bugs associated with batch-aggregation (like leaving a model with
    batch normalization in training mode during evaluation), e.g. by
    constructing a shuffling data loader from an AnomalyDetectionDataset.
    It can also be used to train and evaluate binary Oracle classifiers.
"""
class AnomalyDetectionDataset(torch.utils.data.Dataset):
    """
    Parameters:
        id_dataset: Dataset of in-distribution points.
        ood_dataset: Dataset of out-of-distribution points.
    """
    def __init__(self, id_dataset, ood_dataset):
        relabeled_id_dataset = TransformingDataset(
            id_dataset,
            target_transform = FixedTargetTransform(0)
        )
        relabeled_ood_dataset = TransformingDataset(
            ood_dataset,
            target_transform = FixedTargetTransform(1)
        )
        self._dataset = torch.utils.data.ConcatDataset((relabeled_id_dataset, relabeled_ood_dataset))

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset.__getitem__(index)

def _generate_tqdm_description(title, **kwargs):
    desc = title
    for name, value in kwargs.items():
        if value is not None:
            desc += f' {name}: {value}'
    return desc

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

class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset, train = False):
        super().__init__()
        self.svo_dataset = svo_dataset
        indices = []
        for idx, (subject_appearance_features, _, _, subject_label, _, _) in\
                enumerate(self.svo_dataset):
            if subject_appearance_features is None:
                continue
            if train and subject_label is None:
                continue
            indices.append(idx)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subject_features, _, _, subject_labels, _, _ = self.svo_dataset[self.indices[idx]]
        return subject_features, subject_labels

class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset, train = False):
        super().__init__()
        self.svo_dataset = svo_dataset
        indices = []
        for idx, (_, object_appearance_features, _, _, object_label, _) in\
                enumerate(self.svo_dataset):
            if object_appearance_features is None:
                continue
            if train and object_label is None:
                continue
            indices.append(idx)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        _, object_features, _, _, object_labels, _ = self.svo_dataset[self.indices[idx]]
        return object_features, object_labels

class VerbDataset(torch.utils.data.Dataset):
    def __init__(self, svo_dataset, train = False):
        super().__init__()
        self.svo_dataset = svo_dataset
        indices = []
        for idx, (_, _, verb_appearance_features, _, _, verb_label) in\
                enumerate(self.svo_dataset):
            if verb_appearance_features is None:
                continue
            if train and verb_label is None:
                continue
            indices.append(idx)
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        _, _, verb_features, _, _, verb_labels = self.svo_dataset[self.indices[idx]]
        return verb_features, verb_labels

def bipartition_dataset(dataset, p = 0.5):
    full_indices = list(range(len(dataset)))

    num = int(len(dataset) * p)
    first_indices = list(range(num))
    first_indices = [int(float(index) / p) for index in first_indices]

    second_indices = [index for index in full_indices if index not in first_indices]

    first_dataset = torch.utils.data.Subset(dataset, first_indices)
    second_dataset = torch.utils.data.Subset(dataset, second_indices)
    return first_dataset, second_dataset

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.temperature

class ConfidenceCalibrator:
    def __init__(self):
        self.device = 'cpu'
        self.subject_calibrator = TemperatureScaler()
        self.object_calibrator = TemperatureScaler()
        self.verb_calibrator = TemperatureScaler()

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
