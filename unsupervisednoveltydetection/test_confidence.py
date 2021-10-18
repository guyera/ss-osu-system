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
                csv_path = 'Custom/annotations/val_dataset_v1_val.csv',
                num_subj_cls = 6,
                num_obj_cls = 9,
                num_action_cls = 8,
                training = False,
                image_batch_size = 16,
                feature_extraction_device = self.device
            )
        )

        # The remaining datasets are used for evaluating AUC of subject, object,
        # and verb classifiers in the open set setting.
        
        # Construct data loader
        self.testing_loader = torch.utils.data.DataLoader(
            dataset = testing_set,
            batch_size = 128,
            shuffle = True
        )
        
        # Get example features for shape information to construct classifiers
        appearance_features, _, verb_features, _, _, _ = testing_set[0]
        
        # Create classifier
        classifier = unsupervisednoveltydetection.common.Classifier(
            len(appearance_features),
            len(verb_features),
            1024,
            6,
            9,
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
        for subject_features,\
                object_features,\
                verb_features,\
                subject_labels,\
                object_labels,\
                verb_labels in self.testing_loader:
            subject_features = subject_features.to(self.device)
            object_features = object_features.to(self.device)
            verb_features = verb_features.to(self.device)
            subject_labels = subject_labels.to(self.device)
            object_labels = object_labels.to(self.device)
            verb_labels = verb_labels.to(self.device)
        
            subject_logits, object_logits, verb_logits = self.classifier.predict(
                subject_features,
                object_features,
                verb_features
            )
            
            uncalibrated_subject_probabilities = torch.nn.functional.softmax(subject_logits, dim = 1)
            uncalibrated_object_probabilities = torch.nn.functional.softmax(object_logits, dim = 1)
            uncalibrated_verb_probabilities = torch.nn.functional.softmax(verb_logits, dim = 1)

            calibrated_subject_probabilities, calibrated_object_probabilities, calibrated_verb_probabilities =\
                self.calibrator.calibrate(
                    subject_logits,
                    object_logits,
                    verb_logits
                )
            
            batch_uncalibrated_subject_confidences, subject_predictions =\
                torch.max(uncalibrated_subject_probabilities, dim = 1)
            batch_uncalibrated_object_confidences, object_predictions =\
                torch.max(uncalibrated_object_probabilities, dim = 1)
            batch_uncalibrated_verb_confidences, verb_predictions =\
                torch.max(uncalibrated_verb_probabilities, dim = 1)

            batch_calibrated_subject_confidences, _ =\
                torch.max(calibrated_subject_probabilities, dim = 1)
            batch_calibrated_object_confidences, _ =\
                torch.max(calibrated_object_probabilities, dim = 1)
            batch_calibrated_verb_confidences, _ =\
                torch.max(calibrated_verb_probabilities, dim = 1)
            
            uncalibrated_subject_confidences.append(batch_uncalibrated_subject_confidences)
            uncalibrated_object_confidences.append(batch_uncalibrated_object_confidences)
            uncalibrated_verb_confidences.append(batch_uncalibrated_verb_confidences)

            calibrated_subject_confidences.append(batch_calibrated_subject_confidences)
            calibrated_object_confidences.append(batch_calibrated_object_confidences)
            calibrated_verb_confidences.append(batch_calibrated_verb_confidences)
            
            batch_subject_correct = subject_predictions == subject_labels
            batch_object_correct = object_predictions == object_labels
            batch_verb_correct = verb_predictions == verb_labels
            
            subject_correct.append(batch_subject_correct)
            object_correct.append(batch_object_correct)
            verb_correct.append(batch_verb_correct)
        
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

        self.assertTrue(calibrated_subject_ece <= uncalibrated_subject_ece)
        self.assertTrue(calibrated_object_ece <= uncalibrated_object_ece)
        self.assertTrue(calibrated_verb_ece <= uncalibrated_verb_ece)
        
if __name__ == '__main__':
    unittest.main()
