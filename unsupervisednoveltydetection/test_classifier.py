import torch
import argparse
import pickle
import os
import math

import noveltydetectionfeatures
import unsupervisednoveltydetection.common

import unittest

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.num_bins = 15
        testing_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
            noveltydetectionfeatures.NoveltyFeatureDataset(
                name = 'Custom',
                data_root = 'Custom',
                csv_path = 'Custom/annotations/dataset_v3_val.csv',
                num_subj_cls = 5,
                num_obj_cls = 13,
                num_action_cls = 8,
                training = False,
                image_batch_size = 16,
                feature_extraction_device = self.device
            )
        )
    
        subject_set = unsupervisednoveltydetection.common.SubjectDataset(testing_set, train = False)
        object_set = unsupervisednoveltydetection.common.ObjectDataset(testing_set, train = False)
        verb_set = unsupervisednoveltydetection.common.VerbDataset(testing_set, train = False)

        # Remove novel 0 labels
        id_subject_labels = list(range(1, 5))
        id_object_labels = list(range(1, 13))
        id_verb_labels = list(range(1, 8))
        id_subject_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_set, id_subject_labels)
        id_object_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_set, id_object_labels)
        id_verb_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_set, id_verb_labels)
        subject_set = torch.utils.data.Subset(subject_set, id_subject_indices)
        object_set = torch.utils.data.Subset(object_set, id_object_indices)
        verb_set = torch.utils.data.Subset(verb_set, id_verb_indices)

        # The remaining datasets are used for evaluating AUC of subject, object,
        # and verb classifiers in the open set setting.
        
        # Construct data loader
        self.subject_loader = torch.utils.data.DataLoader(
            dataset = subject_set,
            batch_size = 128,
            shuffle = False
        )
        self.object_loader = torch.utils.data.DataLoader(
            dataset = object_set,
            batch_size = 128,
            shuffle = False
        )
        self.verb_loader = torch.utils.data.DataLoader(
            dataset = verb_set,
            batch_size = 128,
            shuffle = False
        )
        
        # Create classifier
        classifier = unsupervisednoveltydetection.common.Classifier(
            12544,
            12616,
            1024,
            5,
            13,
            8
        )
        
        classifier_state_dict = torch.load('unsupervisednoveltydetection/classifier.pth')
        classifier.load_state_dict(classifier_state_dict)
        
        self.classifier = classifier.to(self.device)
    
    def test_accuracy_acceptable(self):
        num_subject_correct = 0
        num_subject = 0
        num_object_correct = 0
        num_object = 0
        num_verb_correct = 0
        num_verb = 0
        for features, labels in self.subject_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.classifier.predict_subject(features)
            
            predictions = torch.argmax(logits, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            batch_correct = predictions == labels
            num_subject_correct += batch_correct.to(torch.int).sum().cpu().item()
            num_subject += features.shape[0]
        
        for features, labels in self.object_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.classifier.predict_object(features)
            
            predictions = torch.argmax(logits, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            batch_correct = predictions == labels
            num_object_correct += batch_correct.to(torch.int).sum().cpu().item()
            num_object += features.shape[0]
        
        for features, labels in self.verb_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
            
            logits = self.classifier.predict_verb(features)
            
            predictions = torch.argmax(logits, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            batch_correct = predictions == labels
            num_verb_correct += batch_correct.to(torch.int).sum().cpu().item()
            num_verb += features.shape[0]

        subject_accuracy = float(num_subject_correct) / float(num_subject)
        object_accuracy = float(num_object_correct) / float(num_object)
        verb_accuracy = float(num_verb_correct) / float(num_verb)

        print(f'Subject accuracy: {subject_accuracy}')
        print(f'Object accuracy: {object_accuracy}')
        print(f'Verb accuracy: {verb_accuracy}')
        
        # Do one-sided one-sample tests of proportion, to test if estimated
        # accuracy > null hypothesis accuracy (random guessing accuracy)
        subject_p0 = 1.0 / 5.0
        subject_z = (subject_accuracy - subject_p0) / math.sqrt(subject_p0 * (1 - subject_p0) / num_subject)
        object_p0 = 1.0 / 12.0
        object_z = (object_accuracy - object_p0) / math.sqrt(object_p0 * (1 - object_p0) / num_object)
        verb_p0 = 1.0 / 8.0
        verb_z = (verb_accuracy - verb_p0) / math.sqrt(verb_p0 * (1 - verb_p0) / num_verb)
        
        self.assertTrue(subject_z >= 1.64)
        self.assertTrue(object_z >= 1.64)
        self.assertTrue(verb_z >= 1.64)
        
if __name__ == '__main__':
    unittest.main()