import torch

import unsupervisednoveltydetection
import noveltydetectionfeatures
import noveltydetection

import unittest

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 5, 12, 8)
        self.detector = detector.to(self.device)

        state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
        self.detector.load_state_dict(state_dict['module'])

        self.testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = 'Custom',
            csv_path = 'Custom/annotations/dataset_v3_val.csv',
            training = False,
            image_batch_size = 16,
            feature_extraction_device = self.device
        )

    def test_case_1(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None or\
                    subject_label <= 0 or\
                    object_label <= 0 or\
                    verb_label <= 0:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
        
        predictions = self.detector.known_top3(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features)
        for top3 in predictions:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                self.assertTrue(prediction[0][0] != 0 and prediction[0][1] != 0 and prediction[0][2] != 0)

    def test_case_2(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None or\
                    subject_label <= 0 or\
                    object_label <= 0 or\
                    verb_label <= 0:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(None)
            verb_appearance_features.append(example_verb_appearance_features)
        
        predictions = self.detector.known_top3(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features)
        for top3 in predictions:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] == -1)
                self.assertTrue(prediction[0][0] != 0 and prediction[0][1] != 0 and prediction[0][2] != 0)

    def test_case_3(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None or\
                    subject_label <= 0 or\
                    object_label <= 0 or\
                    verb_label <= 0:
                continue
            spatial_features.append(None)
            subject_appearance_features.append(None)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(None)
        
        predictions = self.detector.known_top3(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features)
        for top3 in predictions:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] == -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                self.assertTrue(prediction[0][0] != 0 and prediction[0][1] == 0 and prediction[0][2] != 0)
        
if __name__ == '__main__':
    unittest.main()
