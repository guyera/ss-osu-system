import torch
import argparse
import re
import copy
import numpy as np
import pickle
import os

from tqdm import tqdm

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

def reshaped_list_collate(batch):
    subject_appearance_features = [batch[0] for item in batch]
    object_appearance_features = [batch[1] for item in batch]
    verb_features = [batch[2] for item in batch]
    subject_labels = [batch[3] for item in batch]
    object_labels = [batch[4] for item in batch]
    verb_labels = [batch[5] for item in batch]
    return [subject_appearance_features, object_appearance_features, verb_features, subject_labels, object_labels, verb_labels]

class ReshapedNoveltyFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        spatial_features,\
            subject_appearance_features,\
            object_appearance_features,\
            verb_appearance_features,\
            subject_label,\
            object_label,\
            verb_label = self.dataset[idx]
        
        if subject_appearance_features is not None:
            subject_appearance_features = torch.flatten(
                subject_appearance_features,
                start_dim = 0
            )

        if object_appearance_features is not None:
            object_appearance_features = torch.flatten(
                object_appearance_features,
                start_dim = 0
            )
        
        if spatial_features is not None and verb_appearance_features is not None:
            spatial_features = torch.flatten(spatial_features, start_dim = 0)
            verb_appearance_features = torch.flatten(
                verb_appearance_features,
                start_dim = 0
            )
            verb_features =\
                torch.cat((spatial_features, verb_appearance_features), dim = 0)
        else:
            verb_features = None
        
        return subject_appearance_features,\
            object_appearance_features,\
            verb_features,\
            subject_label,\
            object_label,\
            verb_label

class Classifier:
    def __init__(
            self,
            num_appearance_features,
            num_verb_features,
            num_hidden_nodes,
            num_subj_cls,
            num_obj_cls,
            num_action_cls):
        self.device = 'cpu'
        self.subject_classifier = torch.nn.Sequential(
            torch.nn.Linear(num_appearance_features, num_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_nodes, num_subj_cls)
        )
        self.object_classifier = torch.nn.Sequential(
            torch.nn.Linear(num_appearance_features, num_hidden_nodes),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_nodes, num_obj_cls)
        )
        self.verb_classifier = torch.nn.Sequential(
            torch.nn.Linear(
                num_verb_features,
                num_hidden_nodes
            ),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hidden_nodes, num_action_cls)
        )

    def fit(self, lr, weight_decay, epochs, subject_loader, object_loader, verb_loader):
        # Create optimizers
        subject_optimizer = torch.optim.SGD(
            self.subject_classifier.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = weight_decay
        )
        object_optimizer = torch.optim.SGD(
            self.object_classifier.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = weight_decay
        )
        verb_optimizer = torch.optim.SGD(
            self.verb_classifier.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = weight_decay
        )
        
        # Create loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Progress bar; updated per batch
        progress = tqdm(
            total = epochs,
            desc = 'Training classifiers',
            leave = False
        )
        
        # Construct previous epoch losses for display
        prev_epoch_subject_loss = None
        prev_epoch_object_loss = None
        prev_epoch_verb_loss = None
        
        # For each epoch
        for epoch in range(epochs):
            # Update progress description
            progress.set_description(
                _generate_tqdm_description(
                    'Training classifiers',
                    epoch = f'{epoch + 1} / {epochs}',
                    epoch_subject_loss = prev_epoch_subject_loss,
                    epoch_object_loss = prev_epoch_object_loss,
                    epoch_verb_loss = prev_epoch_verb_loss
                )
            )

            # Init accumulated average losses for the epoch
            epoch_subject_loss = 0.0
            epoch_object_loss = 0.0
            epoch_verb_loss = 0.0
                
            # For each batch
            for features, targets in subject_loader:
                # Relocate data if appropriate
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Make predictions
                predictions = self.subject_classifier(features)
                
                # Compute losses
                loss = criterion(predictions, targets)
                
                # Step optimizers
                subject_optimizer.zero_grad()
                loss.backward()
                subject_optimizer.step()
                
                # Update epoch losses
                epoch_subject_loss += float(loss.item()) / len(subject_loader)

            # For each batch
            for features, targets in object_loader:
                # Relocate data if appropriate
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Make predictions
                predictions = self.object_classifier(features)
                
                # Compute losses
                loss = criterion(predictions, targets)
                
                # Step optimizers
                object_optimizer.zero_grad()
                loss.backward()
                object_optimizer.step()
                
                # Update epoch losses
                epoch_object_loss += float(loss.item()) / len(object_loader)

            # For each batch
            for features, targets in verb_loader:
                # Relocate data if appropriate
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Make predictions
                predictions = self.verb_classifier(features)
                
                # Compute losses
                loss = criterion(predictions, targets)
                
                # Step optimizers
                verb_optimizer.zero_grad()
                loss.backward()
                verb_optimizer.step()
                
                # Update epoch losses
                epoch_verb_loss += float(loss.item()) / len(verb_loader)
                
            # Update progress bar
            progress.update()
            
            # Update previous losses for display
            prev_epoch_subject_loss = epoch_subject_loss
            prev_epoch_object_loss = epoch_object_loss
            prev_epoch_verb_loss = epoch_verb_loss
        
        # Close progress bar
        progress.close()

    def predict(self, subject_features, object_features, verb_features):
        subject_logits = self.subject_classifier(subject_features)
        object_logits = self.object_classifier(object_features)
        verb_logits = self.verb_classifier(verb_features)
        return subject_logits, object_logits, verb_logits

    def predict_score_subject(self, features):
        logits = self.subject_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_score_object(self, features):
        logits = self.object_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_score_verb(self, features):
        logits = self.verb_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_subject(self, features):
        logits = self.subject_classifier(features)
        return logits

    def predict_object(self, features):
        logits = self.object_classifier(features)
        return logits

    def predict_verb(self, features):
        logits = self.verb_classifier(features)
        return logits

    def score_subject(self, features):
        logits = self.subject_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores

    def score_object(self, features):
        logits = self.object_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores

    def score_verb(self, features):
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
        self.sigmoid = torch.nn.Sigmoid()
    
    def fit(
            self,
            classifier,
            lr,
            weight_decay,
            epochs,
            subject_loader,
            object_loader,
            verb_loader):
        # Create optimizers
        subject_optimizer = torch.optim.SGD(
            self.subject_calibrator.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = weight_decay
        )
        object_optimizer = torch.optim.SGD(
            self.object_calibrator.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = weight_decay
        )
        verb_optimizer = torch.optim.SGD(
            self.verb_calibrator.parameters(),
            lr = lr,
            momentum = 0.9,
            weight_decay = weight_decay
        )
        
        # Create loss function
        criterion = torch.nn.CrossEntropyLoss()
        
        # Progress bar; updated per batch
        progress = tqdm(
            total = epochs,
            desc = 'Training calibrators',
            leave = False
        )
        
        # Construct previous epoch losses for display
        prev_epoch_subject_loss = None
        prev_epoch_object_loss = None
        prev_epoch_verb_loss = None
        
        # For each epoch
        for epoch in range(epochs):
            # Update progress description
            progress.set_description(
                _generate_tqdm_description(
                    'Training calibrators',
                    epoch = f'{epoch + 1} / {epochs}',
                    epoch_subject_loss = prev_epoch_subject_loss,
                    epoch_object_loss = prev_epoch_object_loss,
                    epoch_verb_loss = prev_epoch_verb_loss
                )
            )
            
            # Init accumulated average losses for the epoch
            epoch_subject_loss = 0.0
            epoch_object_loss = 0.0
            epoch_verb_loss = 0.0
            
            # For each subject batch
            for features, targets in subject_loader:
                # Relocate data if appropriate
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Get raw logits from classifier
                logits = classifier.predict_subject(features)
                
                # Calibrate logits
                predictions = self.subject_calibrator(logits)
                
                # Compute losses
                loss = criterion(predictions, targets)
                
                # Step optimizer
                subject_optimizer.zero_grad()
                loss.backward()
                subject_optimizer.step()
                
                # Update epoch loss
                epoch_subject_loss += float(loss.item()) / len(subject_loader)

            # For each object batch
            for features, targets in object_loader:
                # Relocate data if appropriate
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Get raw logits from classifier
                logits = classifier.predict_object(features)
                
                # Calibrate logits
                predictions = self.object_calibrator(logits)
                
                # Compute losses
                loss = criterion(predictions, targets)
                
                # Step optimizer
                object_optimizer.zero_grad()
                loss.backward()
                object_optimizer.step()
                
                # Update epoch loss
                epoch_object_loss += float(loss.item()) / len(object_loader)

            # For each verb batch
            for features, targets in verb_loader:
                # Relocate data if appropriate
                features = features.to(self.device)
                targets = targets.to(self.device)
                
                # Get raw logits from classifier
                logits = classifier.predict_verb(features)
                
                # Calibrate logits
                predictions = self.verb_calibrator(logits)
                
                # Compute losses
                loss = criterion(predictions, targets)
                
                # Step optimizer
                verb_optimizer.zero_grad()
                loss.backward()
                verb_optimizer.step()
                
                # Update epoch loss
                epoch_verb_loss += float(loss.item()) / len(verb_loader)

            # Update progress bar
            progress.update()
            
            # Update previous losses for display
            prev_epoch_subject_loss = epoch_subject_loss
            prev_epoch_object_loss = epoch_object_loss
            prev_epoch_verb_loss = epoch_verb_loss
        
        # Close progress bar
        progress.close()

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

class NoveltyPresenceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset,
            id_subject_labels,
            id_object_labels,
            id_verb_labels,
            ood_subject_labels,
            ood_object_labels,
            ood_verb_labels):
        super().__init__()

        self.dataset = dataset
        self.id_subject_labels = id_subject_labels
        self.id_object_labels = id_object_labels
        self.id_verb_labels = id_verb_labels

        indices = []
        for idx, (_, _, _, subject_label, object_label, verb_label) in\
                enumerate(self.dataset):
            relevant_joint_label = True
            if subject_label not in id_subject_labels and\
                    subject_label not in ood_subject_labels:
                relevant_joint_label = False
            if object_label not in id_object_labels and\
                    object_label not in ood_object_labels:
                relevant_joint_label = False
            if verb_label not in id_verb_labels and\
                    verb_label not in ood_verb_labels:
                relevant_joint_label = False
            
            if relevant_joint_label:
                indices.append(idx)
        self.indices = indices
    
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        subject_features,\
            object_features,\
            verb_features,\
            subject_label,\
            object_label,\
            verb_label = self.dataset[self.indices[idx]]
        
        if subject_label not in self.id_subject_labels or\
                object_label not in self.id_object_labels or\
                verb_label not in self.id_verb_labels:
            # If any of the labels are not ID (and thus are OOD), mark the label
            # as 1
            label = 1
        else:
            # Otherwise, mark the label as 0
            label = 0

        return subject_features,\
            object_features,\
            verb_features,\
            label
