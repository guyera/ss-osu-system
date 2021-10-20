import torch

import unsupervisednoveltydetection
import noveltydetectionfeatures
import noveltydetection

import unittest

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 6, 9, 8)
        self.detector = detector.to(self.device)

        state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
        self.detector.load_state_dict(state_dict['module'])

        self.testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
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

    def test_case_1_t1(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
        
        results = self.detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device = self.device))
        for top3 in results['top3']:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] is not None and prediction[0][1] is not None and prediction[0][2] is not None)
                self.assertTrue(prediction[0][0] == 0 and prediction[0][1] != 0 and prediction[0][2] != 0)

    def test_case_1_t2(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
        
        results = self.detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = self.device))
        for top3 in results['top3']:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] is not None and prediction[0][1] is not None and prediction[0][2] is not None)
                self.assertTrue(prediction[0][0] != 0 and prediction[0][1] == 0 and prediction[0][2] != 0)

    def test_case_1_t3(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
        
        results = self.detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device = self.device))
        for top3 in results['top3']:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] is not None and prediction[0][1] is not None and prediction[0][2] is not None)
                self.assertTrue(prediction[0][0] != 0 and prediction[0][1] != 0 and prediction[0][2] == 0)

    def test_case_1_t4(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
        
        results = self.detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], device = self.device))
        for top3 in results['top3']:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] is not None and prediction[0][1] is not None and prediction[0][2] is not None)
                self.assertTrue(prediction[0][0] != 0 and prediction[0][1] != 0 and prediction[0][2] != 0)
    
    def test_case_1_not_t4(self):
        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
        
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in self.testing_set:
            if example_spatial_features is None or\
                    example_subject_appearance_features is None or\
                    example_object_appearance_features is None or\
                    example_verb_appearance_features is None:
                continue
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
            
        results = self.detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.25, 0.25, 0.25, 0.0, 0.25], device = self.device))
        for top3 in results['top3']:
            previous_confidence = 1.0
            for prediction in top3:
                self.assertTrue(prediction[1] <= previous_confidence)
                previous_confidence = prediction[1]
                self.assertTrue(prediction[0][0] is not None and prediction[0][1] is not None and prediction[0][2] is not None)
                self.assertTrue(prediction[0][0] == 0 or prediction[0][1] == 0 or prediction[0][2] == 0)
        
if __name__ == '__main__':
    unittest.main()
