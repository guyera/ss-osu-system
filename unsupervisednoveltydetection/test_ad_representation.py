import torch
import argparse
import pickle
import os
import math
import sklearn.metrics

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
                csv_path = 'Custom/annotations/val_dataset_v1_train.csv',
                num_subj_cls = 6,
                num_obj_cls = 9,
                num_action_cls = 8,
                training = True,
                image_batch_size = 16,
                feature_extraction_device = self.device
            )
        )

        testing_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
            noveltydetectionfeatures.NoveltyFeatureDataset(
                name = 'Custom',
                data_root = 'Custom',
                csv_path = 'Custom/annotations/val_dataset_v1_val.csv',
                num_subj_cls = 6,
                num_obj_cls = 9,
                num_action_cls = 8,
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
        
        id_subject_labels = list(range(0, 4))
        ood_subject_labels = list(range(4, 6))
        id_object_labels = list(range(0, 4))
        ood_object_labels = list(range(4, 6))
        id_verb_labels = list(range(0, 4))
        ood_verb_labels = list(range(4, 6))
        subject_class_split = unsupervisednoveltydetection.common.ClassSplit(id_subject_labels, ood_subject_labels)
        object_class_split = unsupervisednoveltydetection.common.ClassSplit(id_object_labels, ood_object_labels)
        verb_class_split = unsupervisednoveltydetection.common.ClassSplit(id_verb_labels, ood_verb_labels)
        
        id_subject_training_set, ood_subject_training_set = subject_class_split.split_dataset(subject_training_set)
        id_object_training_set, ood_object_training_set = object_class_split.split_dataset(object_training_set)
        id_verb_training_set, ood_verb_training_set = verb_class_split.split_dataset(verb_training_set)
        
        id_subject_testing_set, ood_subject_testing_set = subject_class_split.split_dataset(subject_testing_set)
        id_object_testing_set, ood_object_testing_set = object_class_split.split_dataset(object_testing_set)
        id_verb_testing_set, ood_verb_testing_set = verb_class_split.split_dataset(verb_testing_set)

        ood_subject_set = torch.utils.data.ConcatDataset((ood_subject_training_set, ood_subject_testing_set))
        ood_object_set = torch.utils.data.ConcatDataset((ood_object_training_set, ood_object_testing_set))
        ood_verb_set = torch.utils.data.ConcatDataset((ood_verb_training_set, ood_verb_testing_set))

        # Remap ID training labels
        id_subject_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_subject_training_set, id_subject_labels)
        id_object_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_object_training_set, id_object_labels)
        id_verb_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_verb_training_set, id_verb_labels)
        
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
            len(id_subject_labels),
            len(id_object_labels),
            len(id_verb_labels)
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

        # AUC is the proportion of ID-OOD pairs which are ordered correctly
        # by their anomaly scores. To do a one-sample test of proportion,
        # we'd need to compute the number of pairs. However, having many e.g.
        # OOD examples while having few ID examples should not guarantee a
        # narrow confidence interval, despite having many pairs, because the
        # information about the distribution of scores over ID data is very
        # limited, and so the ordering of the pairs is more likely to be biased
        # depending on the few ID scores we have. So we use the minimum of
        # the number of ID examples and the number of OOD examples as N in the
        # tests of proportion. Perhaps there's a more principled statistical
        # test for AUC, but this is sufficient for now.
        num_subject = min(num_id_subject, num_ood_subject)
        num_object = min(num_id_object, num_ood_object)
        num_verb = min(num_id_verb, num_ood_verb)

        subject_auc = sklearn.metrics.roc_auc_score(subject_trues.numpy(), subject_scores.detach().cpu().numpy())
        object_auc = sklearn.metrics.roc_auc_score(object_trues.numpy(), object_scores.detach().cpu().numpy())
        verb_auc = sklearn.metrics.roc_auc_score(verb_trues.numpy(), verb_scores.detach().cpu().numpy())
        
        # Do one-sided one-sample tests of proportion to test if estimated
        # auc > null hypothesis auc (random guessing auc = 0.5)
        subject_z = (subject_auc - 0.5) / math.sqrt(0.25 / num_subject)
        object_z = (object_auc - 0.5) / math.sqrt(0.25 / num_object)
        verb_z = (verb_auc - 0.5) / math.sqrt(0.25 / num_verb)
        
        # Using alpha = 0.05, z should be greater than or equal to 1.64 to
        # reject the null hypothesis
        self.assertTrue(subject_z >= 1.64)
        self.assertTrue(object_z >= 1.64)
        self.assertTrue(verb_z >= 1.64)
        
if __name__ == '__main__':
    unittest.main()
