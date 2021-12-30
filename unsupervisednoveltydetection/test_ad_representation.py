import torch
import argparse
import pickle
import os
import math
import sklearn.metrics
import numpy as np

import noveltydetectionfeatures
import unsupervisednoveltydetection.common

import unittest

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.num_bins = 15
        training_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
            noveltydetectionfeatures.NoveltyFeatureDataset(
                name = 'Custom',
                data_root = 'Custom',
                csv_path = 'Custom/annotations/dataset_v4_train.csv',
                training = True,
                image_batch_size = 16,
                feature_extraction_device = self.device
            )
        )

        testing_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
            noveltydetectionfeatures.NoveltyFeatureDataset(
                name = 'Custom',
                data_root = 'Custom',
                csv_path = 'Custom/annotations/dataset_v4_val.csv',
                training = True,
                image_batch_size = 16,
                feature_extraction_device = self.device
            )
        )
        
        subject_training_set = unsupervisednoveltydetection.common.SubjectDataset(training_set, train = True)
        object_training_set = unsupervisednoveltydetection.common.ObjectDataset(training_set, train = True)
        verb_training_set = unsupervisednoveltydetection.common.VerbDataset(training_set, train = True)

        subject_testing_set = unsupervisednoveltydetection.common.SubjectDataset(testing_set, train = True)
        object_testing_set = unsupervisednoveltydetection.common.ObjectDataset(testing_set, train = True)
        verb_testing_set = unsupervisednoveltydetection.common.VerbDataset(testing_set, train = True)
        
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
        
        subject_class_split = unsupervisednoveltydetection.common.ClassSplit(self.id_subject_labels, self.ood_subject_labels)
        object_class_split = unsupervisednoveltydetection.common.ClassSplit(self.id_object_labels, self.ood_object_labels)
        verb_class_split = unsupervisednoveltydetection.common.ClassSplit(self.id_verb_labels, self.ood_verb_labels)
        
        id_subject_training_set, ood_subject_training_set = subject_class_split.split_dataset(subject_training_set)
        id_object_training_set, ood_object_training_set = object_class_split.split_dataset(object_training_set)
        id_verb_training_set, ood_verb_training_set = verb_class_split.split_dataset(verb_training_set)

        # Remap ID training labels to [0, ..., K - 1]
        id_subject_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_subject_training_set, self.id_subject_labels)
        id_object_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_object_training_set, self.id_object_labels)
        id_verb_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_verb_training_set, self.id_verb_labels)
        
        # Upshift ID training labels to [1, ..., K], leaving 0 for novelty
        transform = unsupervisednoveltydetection.common.UpshiftTargetTransform()
        id_subject_training_set = unsupervisednoveltydetection.common.TransformingDataset(id_subject_training_set, target_transform = transform)
        id_object_training_set = unsupervisednoveltydetection.common.TransformingDataset(id_object_training_set, target_transform = transform)
        id_verb_training_set = unsupervisednoveltydetection.common.TransformingDataset(id_verb_training_set, target_transform = transform)
        
        id_subject_testing_set, ood_subject_testing_set = subject_class_split.split_dataset(subject_testing_set)
        id_object_testing_set, ood_object_testing_set = object_class_split.split_dataset(object_testing_set)
        id_verb_testing_set, ood_verb_testing_set = verb_class_split.split_dataset(verb_testing_set)
        
        ood_subject_set = torch.utils.data.ConcatDataset((ood_subject_training_set, ood_subject_testing_set))
        ood_object_set = torch.utils.data.ConcatDataset((ood_object_training_set, ood_object_testing_set))
        ood_verb_set = torch.utils.data.ConcatDataset((ood_verb_training_set, ood_verb_testing_set))
        
        # Construct anomaly detection sets (i.e. remap ID testing labels to 0,
        # OOD testing labels to 1)
        subject_anomaly_detection_set = unsupervisednoveltydetection.common.AnomalyDetectionDataset(id_subject_testing_set, ood_subject_set)
        object_anomaly_detection_set = unsupervisednoveltydetection.common.AnomalyDetectionDataset(id_object_testing_set, ood_object_set)
        verb_anomaly_detection_set = unsupervisednoveltydetection.common.AnomalyDetectionDataset(id_verb_testing_set, ood_verb_set)

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
        classifier = unsupervisednoveltydetection.common.Classifier(
            12544,
            12616,
            1024,
            len(self.id_subject_labels) + 1, # Add 1 for anomaly label = 0
            len(self.id_object_labels) + 1, # Add 1 for anomaly label = 0
            len(self.id_verb_labels) + 1 # Add 1 for anomaly label = 0
        ).to(self.device)
        
        classifier.fit(0.01, 0.0001, 300, id_subject_training_loader, id_object_training_loader, id_verb_training_loader)
        
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
