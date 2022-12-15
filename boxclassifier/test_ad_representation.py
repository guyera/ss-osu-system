import torch
import argparse
import pickle
import os
import math
import sklearn.metrics
import numpy as np

from boximagedataset import BoxImageDataset
import boxclassifier

import unittest

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
        _, subject_labels, _, _, subject_images, _, _, _ = self.svo_dataset[self.indices[idx]]
        return subject_images, subject_labels

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
        _, _, object_labels, _, _, object_images, _, _ = self.svo_dataset[self.indices[idx]]
        return object_images, object_labels

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
        _, _, _, verb_labels, _, _, verb_images, _ = self.svo_dataset[self.indices[idx]]
        return verb_images, verb_labels

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.num_bins = 15
        training_set = BoxImageDataset(
            name = 'Custom',
            data_root = 'Custom',
            csv_path = 'Custom/annotations/dataset_v4_train.csv',
            training = True,
            image_batch_size = 16,
            feature_extraction_device = self.device
        )

        testing_set = BoxImageDataset(
            name = 'Custom',
            data_root = 'Custom',
            csv_path = 'Custom/annotations/dataset_v4_val.csv',
            training = True,
            image_batch_size = 16,
            feature_extraction_device = self.device
        )
        
        subject_training_set = SubjectDataset(training_set, train = True)
        object_training_set = ObjectDataset(training_set, train = True)
        verb_training_set = VerbDataset(training_set, train = True)

        subject_testing_set = SubjectDataset(testing_set, train = True)
        object_testing_set = ObjectDataset(testing_set, train = True)
        verb_testing_set = VerbDataset(testing_set, train = True)
        
        num_id_subject = 5 // 2
        num_id_object = 12 // 2
        num_id_verb = 8 // 2
        
        # generate labels randomly and store them within self for later
        # export
        subject_labels = np.arange(1, 5)
        object_labels = np.arange(1, 12)
        verb_labels = np.arange(1, 8)
        np.random.shuffle(subject_labels)
        np.random.shuffle(object_labels)
        np.random.shuffle(verb_labels)
        subject_labels = subject_labels.tolist()
        object_labels = object_labels.tolist()
        verb_labels = verb_labels.tolist()

        self.id_subject_labels = subject_labels[:num_id_subject]
        self.ood_subject_labels = subject_labels[num_id_subject:]
        self.id_object_labels = object_labels[:num_id_object]
        self.ood_object_labels = object_labels[num_id_object:]
        self.id_verb_labels = verb_labels[:num_id_verb]
        self.ood_verb_labels = verb_labels[num_id_verb:]
        
        subject_class_split = ClassSplit(self.id_subject_labels, self.ood_subject_labels)
        object_class_split = ClassSplit(self.id_object_labels, self.ood_object_labels)
        verb_class_split = ClassSplit(self.id_verb_labels, self.ood_verb_labels)
        
        id_subject_training_set, ood_subject_training_set = subject_class_split.split_dataset(subject_training_set)
        id_object_training_set, ood_object_training_set = object_class_split.split_dataset(object_training_set)
        id_verb_training_set, ood_verb_training_set = verb_class_split.split_dataset(verb_training_set)

        # Remap ID training labels to [0, ..., K - 1]
        id_subject_training_set = LabelMappingDataset(id_subject_training_set, self.id_subject_labels)
        id_object_training_set = LabelMappingDataset(id_object_training_set, self.id_object_labels)
        id_verb_training_set = LabelMappingDataset(id_verb_training_set, self.id_verb_labels)
        
        # Upshift ID training labels to [1, ..., K], leaving 0 for novelty
        transform = UpshiftTargetTransform()
        id_subject_training_set = TransformingDataset(id_subject_training_set, target_transform = transform)
        id_object_training_set = TransformingDataset(id_object_training_set, target_transform = transform)
        id_verb_training_set = TransformingDataset(id_verb_training_set, target_transform = transform)
        
        id_subject_testing_set, ood_subject_testing_set = subject_class_split.split_dataset(subject_testing_set)
        id_object_testing_set, ood_object_testing_set = object_class_split.split_dataset(object_testing_set)
        id_verb_testing_set, ood_verb_testing_set = verb_class_split.split_dataset(verb_testing_set)
        
        ood_subject_set = torch.utils.data.ConcatDataset((ood_subject_training_set, ood_subject_testing_set))
        ood_object_set = torch.utils.data.ConcatDataset((ood_object_training_set, ood_object_testing_set))
        ood_verb_set = torch.utils.data.ConcatDataset((ood_verb_training_set, ood_verb_testing_set))
        
        # Construct anomaly detection sets (i.e. remap ID testing labels to 0,
        # OOD testing labels to 1)
        subject_anomaly_detection_set = AnomalyDetectionDataset(id_subject_testing_set, ood_subject_set)
        object_anomaly_detection_set = AnomalyDetectionDataset(id_object_testing_set, ood_object_set)
        verb_anomaly_detection_set = AnomalyDetectionDataset(id_verb_testing_set, ood_verb_set)

        # Construct data loaders
        id_subject_training_loader = torch.utils.data.DataLoader(
            dataset = id_subject_training_set,
            batch_size = 128,
            shuffle = True
        )
        id_object_training_loader = torch.utils.data.DataLoader(
            dataset = id_object_training_set,
            batch_size = 128,
            shuffle = True
        )
        id_verb_training_loader = torch.utils.data.DataLoader(
            dataset = id_verb_training_set,
            batch_size = 128,
            shuffle = True
        )
        
        self.subject_anomaly_detection_loader = torch.utils.data.DataLoader(
            dataset = subject_anomaly_detection_set,
            batch_size = 128,
            shuffle = True
        )
        self.object_anomaly_detection_loader = torch.utils.data.DataLoader(
            dataset = object_anomaly_detection_set,
            batch_size = 128,
            shuffle = True
        )
        self.verb_anomaly_detection_loader = torch.utils.data.DataLoader(
            dataset = verb_anomaly_detection_set,
            batch_size = 128,
            shuffle = True
        )
        
        # Create classifier
        classifier = boxclassifier.ClassifierV2(
            256,
            len(self.id_subject_labels) + 1, # Add 1 for anomaly label = 0
            len(self.id_object_labels) + 1, # Add 1 for anomaly label = 0
            len(self.id_verb_labels) + 1, # Add 1 for anomaly label = 0
            72
        ).to(self.device)
        
        # TODO train the classifier on the ID training data
        
        self.classifier = classifier
    
    def test_auc_acceptable(self):
        subject_scores = []
        subject_trues = []
        object_scores = []
        object_trues = []
        verb_scores = []
        verb_trues = []
        
        for features, labels in self.subject_anomaly_detection_loader:
            subject_trues.append(labels)
            
            features = features.to(self.device)
            scores = self.classifier.score_subject(features)
            
            subject_scores.append(scores)
        
        subject_scores = torch.cat(subject_scores, dim = 0)
        subject_trues = torch.cat(subject_trues, dim = 0)

        for features, labels in self.object_anomaly_detection_loader:
            object_trues.append(labels)
            
            features = features.to(self.device)
            scores = self.classifier.score_object(features)
            
            object_scores.append(scores)
        
        object_scores = torch.cat(object_scores, dim = 0)
        object_trues = torch.cat(object_trues, dim = 0)

        for features, labels in self.verb_anomaly_detection_loader:
            verb_trues.append(labels)
            
            features = features.to(self.device)
            scores = self.classifier.score_verb(features)
            
            verb_scores.append(scores)
        
        verb_scores = torch.cat(verb_scores, dim = 0)
        verb_trues = torch.cat(verb_trues, dim = 0)
        
        num_ood_subject = int(subject_trues.sum().item())
        num_id_subject = len(subject_trues) - num_ood_subject
        num_ood_object = int(object_trues.sum().item())
        num_id_object = len(object_trues) - num_ood_object
        num_ood_verb = int(verb_trues.sum().item())
        num_id_verb = len(verb_trues) - num_ood_verb

        subject_auc = sklearn.metrics.roc_auc_score(subject_trues.numpy(), subject_scores.detach().cpu().numpy())
        object_auc = sklearn.metrics.roc_auc_score(object_trues.numpy(), object_scores.detach().cpu().numpy())
        verb_auc = sklearn.metrics.roc_auc_score(verb_trues.numpy(), verb_scores.detach().cpu().numpy())
        partial_subject_auc = sklearn.metrics.roc_auc_score(subject_trues.numpy(), subject_scores.detach().cpu().numpy(), max_fpr = 0.25)
        partial_object_auc = sklearn.metrics.roc_auc_score(object_trues.numpy(), object_scores.detach().cpu().numpy(), max_fpr = 0.25)
        partial_verb_auc = sklearn.metrics.roc_auc_score(verb_trues.numpy(), verb_scores.detach().cpu().numpy(), max_fpr = 0.25)

        # save self.classifier, self.XXX_labels, and XXX_auc variables
        # in a dictionary via pickle for comparative supervised experiments.
        results = {
            'classifier': self.classifier.state_dict(),
            'subject_auc': subject_auc,
            'object_auc': object_auc,
            'verb_auc': verb_auc,
            'partial_subject_auc': partial_subject_auc,
            'partial_object_auc': partial_object_auc,
            'partial_verb_auc': partial_verb_auc,
            'id_subject_labels': self.id_subject_labels,
            'id_object_labels': self.id_object_labels,
            'id_verb_labels': self.id_verb_labels,
            'ood_subject_labels': self.ood_subject_labels,
            'ood_object_labels': self.ood_object_labels,
            'ood_verb_labels': self.ood_verb_labels,
        }
        with open(f'results.pkl', 'wb') as f:
            pickle.dump(results, f)
        
        print(f'Subject auc: {subject_auc}')
        print(f'Object auc: {object_auc}')
        print(f'Verb auc: {verb_auc}')
        print(f'Partial subject auc: {partial_subject_auc}')
        print(f'Partial object auc: {partial_object_auc}')
        print(f'Partial verb auc: {partial_verb_auc}')
        
        self.assertTrue(subject_auc >= 0.5)
        self.assertTrue(object_auc >= 0.5)
        self.assertTrue(verb_auc >= 0.5)
        self.assertTrue(partial_subject_auc >= 0.5)
        self.assertTrue(partial_object_auc >= 0.5)
        self.assertTrue(partial_verb_auc >= 0.5)
        
if __name__ == '__main__':
    unittest.main()
