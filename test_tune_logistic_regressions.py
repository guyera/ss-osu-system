import torch
import argparse
import pickle
import os
import math
import random

import noveltydetectionfeatures
import noveltydetection

import unittest

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.num_bins = 15
        
        # Create logistic regressions
        case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(self.device)
        case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(self.device)
        case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(self.device)
        
        # Load data from unsupervised_novelty_detection_module.pth
        state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
        case_1_logistic_regression.load_state_dict(state_dict['case_1_logistic_regression'])
        case_2_logistic_regression.load_state_dict(state_dict['case_2_logistic_regression'])
        case_3_logistic_regression.load_state_dict(state_dict['case_3_logistic_regression'])
        self.case_1_scores = state_dict['case_1_scores'].to(self.device)
        self.case_2_scores = state_dict['case_2_scores'].to(self.device)
        self.case_3_scores = state_dict['case_3_scores'].to(self.device)
        self.case_1_labels = state_dict['case_1_labels'].to(self.device)
        self.case_2_labels = state_dict['case_2_labels'].to(self.device)
        self.case_3_labels = state_dict['case_3_labels'].to(self.device)

        # Move logistic regressions to GPU
        self.case_1_logistic_regression = case_1_logistic_regression.to(self.device)
        self.case_2_logistic_regression = case_2_logistic_regression.to(self.device)
        self.case_3_logistic_regression = case_3_logistic_regression.to(self.device)

        # Generate 60 random novelty scores for subject, verb, and object, of
        # random cases. Ordinarily, these would be output by the
        # UnsupervisedNoveltyDetectionModule over the first 60 trial images
        self.first_60_subject_novelty_scores = []
        self.first_60_verb_novelty_scores = []
        self.first_60_object_novelty_scores = []
        for _ in range(60):
            case = random.randint(1, 3)
            if case == 1:
                self.first_60_subject_novelty_scores.append(torch.randn(1, device = self.device).squeeze(0))
                self.first_60_verb_novelty_scores.append(torch.randn(1, device = self.device).squeeze(0))
                self.first_60_object_novelty_scores.append(torch.randn(1, device = self.device).squeeze(0))
            elif case == 2:
                self.first_60_subject_novelty_scores.append(torch.randn(1, device = self.device).squeeze(0))
                self.first_60_verb_novelty_scores.append(torch.randn(1, device = self.device).squeeze(0))
                self.first_60_object_novelty_scores.append(None)
            else:
                self.first_60_subject_novelty_scores.append(None)
                self.first_60_verb_novelty_scores.append(None)
                self.first_60_object_novelty_scores.append(torch.randn(1, device = self.device).squeeze(0))
    
    def test_runs(self):
        noveltydetection.utils.tune_logistic_regressions(
            self.case_1_logistic_regression,
            self.case_2_logistic_regression,
            self.case_3_logistic_regression,
            self.case_1_scores,
            self.case_2_scores,
            self.case_3_scores,
            self.case_1_labels,
            self.case_2_labels,
            self.case_3_labels,
            self.first_60_subject_novelty_scores,
            self.first_60_verb_novelty_scores,
            self.first_60_object_novelty_scores,
            epochs = 3000,
            quiet = True
        )

if __name__ == '__main__':
    unittest.main()
