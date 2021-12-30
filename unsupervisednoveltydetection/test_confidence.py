import torch
import argparse
import pickle
import os

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
                csv_path = 'Custom/annotations/dataset_v4_val.csv',
                training = False,
                image_batch_size = 16,
                feature_extraction_device = self.device
            )
        )
        
        # Initialize the datasets in "train" mode. All this actually does is
        # omit null instances (e.g. the subject_set will omit images which
        # don't have subject boxes), which is helpful for testing of
        # individual units.
        subject_set = unsupervisednoveltydetection.common.SubjectDataset(testing_set, train = True)
        object_set = unsupervisednoveltydetection.common.ObjectDataset(testing_set, train = True)
        verb_set = unsupervisednoveltydetection.common.VerbDataset(testing_set, train = True)

        # Remove novel 0 labels
        id_subject_labels = list(range(1, 5))
        id_object_labels = list(range(1, 12))
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
            12,
            8
        )
        
        classifier_state_dict = torch.load('unsupervisednoveltydetection/confidence_classifier.pth')
        classifier.load_state_dict(classifier_state_dict)
        
        self.classifier = classifier.to(self.device)
        
        calibrator = unsupervisednoveltydetection.common.ConfidenceCalibrator().to(self.device)
        calibrator_state_dict = torch.load('unsupervisednoveltydetection/confidence_calibrator.pth')
        calibrator.load_state_dict(calibrator_state_dict)
        self.calibrator = calibrator
    
    def ece(self, confidences, correct):
        bin_starts = torch.arange(self.num_bins, device = self.device, dtype = torch.float) / self.num_bins
        bin_ends = (torch.arange(self.num_bins, device = self.device, dtype = torch.float) + 1.0) / self.num_bins
        bin_ends[-1] += 0.000001
        
        # Compute NxB confidence membership matrix, where B = self.num_bins
        bin_memberships = torch.logical_and(
            confidences.unsqueeze(1) >= bin_starts,
            confidences.unsqueeze(1) < bin_ends
        ).to(torch.float)
        
        # Use bin memberships to compute ECE
        bin_counts = bin_memberships.sum(dim = 0)
        bin_weights = bin_counts / (torch.sum(bin_counts) + 0.000001)
        bin_correct = (bin_memberships * correct.unsqueeze(1)).sum(dim = 0)
        bin_accuracies = bin_correct / (bin_counts + 0.000001)
        bin_confidences = (bin_memberships * confidences.unsqueeze(1)).sum(dim = 0) / (bin_counts + 0.000001)
        bin_abs_errors = torch.abs(bin_accuracies - bin_confidences)
        bin_weighted_errors = bin_abs_errors * bin_weights
        ece = bin_weighted_errors.sum()

        return ece.item()

    def test_reduces_ece(self):
        uncalibrated_subject_confidences = []
        uncalibrated_object_confidences = []
        uncalibrated_verb_confidences = []
        calibrated_subject_confidences = []
        calibrated_object_confidences = []
        calibrated_verb_confidences = []
        subject_correct = []
        object_correct = []
        verb_correct = []
        for features, labels in self.subject_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
        
            logits = self.classifier.predict_subject(features)
            
            uncalibrated_probabilities = torch.nn.functional.softmax(logits, dim = 1)
            calibrated_probabilities = self.calibrator.calibrate_subject(logits)
            
            batch_uncalibrated_confidences, predictions =\
                torch.max(uncalibrated_probabilities, dim = 1)
            batch_calibrated_confidences, _ =\
                torch.max(calibrated_probabilities, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            uncalibrated_subject_confidences.append(batch_uncalibrated_confidences)
            calibrated_subject_confidences.append(batch_calibrated_confidences)
            
            batch_correct = predictions == labels
            subject_correct.append(batch_correct)

        for features, labels in self.object_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
        
            logits = self.classifier.predict_object(features)
            
            uncalibrated_probabilities = torch.nn.functional.softmax(logits, dim = 1)
            calibrated_probabilities = self.calibrator.calibrate_object(logits)
            
            batch_uncalibrated_confidences, predictions =\
                torch.max(uncalibrated_probabilities, dim = 1)
            batch_calibrated_confidences, _ =\
                torch.max(calibrated_probabilities, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            uncalibrated_object_confidences.append(batch_uncalibrated_confidences)
            calibrated_object_confidences.append(batch_calibrated_confidences)
            
            batch_correct = predictions == labels
            object_correct.append(batch_correct)

        for features, labels in self.verb_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)
        
            logits = self.classifier.predict_verb(features)
            
            uncalibrated_probabilities = torch.nn.functional.softmax(logits, dim = 1)
            calibrated_probabilities = self.calibrator.calibrate_verb(logits)
            
            batch_uncalibrated_confidences, predictions =\
                torch.max(uncalibrated_probabilities, dim = 1)
            batch_calibrated_confidences, _ =\
                torch.max(calibrated_probabilities, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            uncalibrated_verb_confidences.append(batch_uncalibrated_confidences)
            calibrated_verb_confidences.append(batch_calibrated_confidences)
            
            batch_correct = predictions == labels
            verb_correct.append(batch_correct)
        
        uncalibrated_subject_confidences = torch.cat(uncalibrated_subject_confidences, dim = 0)
        uncalibrated_object_confidences = torch.cat(uncalibrated_object_confidences, dim = 0)
        uncalibrated_verb_confidences = torch.cat(uncalibrated_verb_confidences, dim = 0)

        calibrated_subject_confidences = torch.cat(calibrated_subject_confidences, dim = 0)
        calibrated_object_confidences = torch.cat(calibrated_object_confidences, dim = 0)
        calibrated_verb_confidences = torch.cat(calibrated_verb_confidences, dim = 0)
        
        subject_correct = torch.cat(subject_correct, dim = 0)
        object_correct = torch.cat(object_correct, dim = 0)
        verb_correct = torch.cat(verb_correct, dim = 0)

        uncalibrated_subject_ece = self.ece(uncalibrated_subject_confidences, subject_correct)
        uncalibrated_object_ece = self.ece(uncalibrated_object_confidences, object_correct)
        uncalibrated_verb_ece = self.ece(uncalibrated_verb_confidences, verb_correct)

        calibrated_subject_ece = self.ece(calibrated_subject_confidences, subject_correct)
        calibrated_object_ece = self.ece(calibrated_object_confidences, object_correct)
        calibrated_verb_ece = self.ece(calibrated_verb_confidences, verb_correct)

        print(f'Uncalibrated subject ECE: {uncalibrated_subject_ece}')
        print(f'Calibrated subject ECE: {calibrated_subject_ece}')
        print(f'Uncalibrated object ECE: {uncalibrated_object_ece}')
        print(f'Calibrated object ECE: {calibrated_object_ece}')
        print(f'Uncalibrated verb ECE: {uncalibrated_verb_ece}')
        print(f'Calibrated verb ECE: {calibrated_verb_ece}')
        
if __name__ == '__main__':
    unittest.main()
