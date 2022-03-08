import os

import torch

import unsupervisednoveltydetection
import noveltydetectionfeatures
import noveltydetection

import unittest

class TestScoreCorrelationLogger(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 5, 12, 8)
        self.detector = detector.to(self.device)

        state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
        self.detector.load_state_dict(state_dict['module'])
        
        self.testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = 'Custom',
            csv_path = 'Custom/annotations/dataset_v4_val.csv',
            training = False,
            image_batch_size = 16,
            feature_extraction_device = self.device
        )
        
        self.logger = unsupervisednoveltydetection.UnsupervisedNoveltyDetectorLogger()

    def test(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label in self.testing_set:
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
        
        results = self.detector.score(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features)
        self.logger.log_novelty_scores(results['subject_novelty_score'], results['verb_novelty_score'], results['object_novelty_score'])
        figure_dir = 'unit_test_figures/score_correlation_logger'
        if not os.path.exists(figure_dir):
            os.makedirs(figure_dir)
        self.logger.save_figures(figure_dir)
        assert os.path.exists(os.path.join(figure_dir, 'subject_verb_scatter.jpg'))
        assert os.path.exists(os.path.join(figure_dir, 'subject_object_scatter.jpg'))
        assert os.path.exists(os.path.join(figure_dir, 'verb_object_scatter.jpg'))

if __name__ == '__main__':
    unittest.main()
