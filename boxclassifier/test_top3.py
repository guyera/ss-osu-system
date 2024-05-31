# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import os
import unittest

import torch

import boxclassifier
from boximagedataset import BoxImageDataset
from backbone import Backbone
from tupleprediction import TuplePredictor

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'

        architecture = Backbone.Architecture.swin_t
        backbone = Backbone(architecture)
        pretrained_models_dir = os.path.join(
            'pretrained-models',
            architecture.value['name']
        )
        backbone_state_dict = torch.load(
            os.path.join(pretrained_models_dir, 'backbone.pth')
        )
        backbone.load_state_dict(backbone_state_dict)
        backbone = backbone.to(self.device)
        backbone.eval()
        self.backbone = backbone
        
        classifier = boxclassifier.ClassifierV2(256, 5, 12, 8, 72)
        classifier_state_dict = torch.load(os.path.join(
            pretrained_models_dir,
            'classifier.pth'
        ))
        classifier.load_state_dict(classifier_state_dict)
        self.classifier = classifier.to(self.device)

        detector = TuplePredictor(5, 12, 8)
        tuple_prediction_state_dict = torch.load(os.path.join(
            pretrained_models_dir,
            'tuple-prediction.pth'
        ))
        detector.load_state_dict(tuple_prediction_state_dict)
        self.detector = detector.to(self.device)

        self.testing_set = BoxImageDataset(
            name = 'Custom',
            data_root = './',
            csv_path = 'dataset_v4/dataset_v4_2_val.csv',
            training = False,
            image_batch_size = 16,
            feature_extraction_device = self.device
        )

    def test_case_1_t1(self):
        with torch.no_grad():
            spatial_features = []
            subject_box_features = []
            object_box_features = []
            verb_box_features = []
            
            for example_spatial_features, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
                if example_spatial_features is None or\
                        example_subject_images is None or\
                        example_object_images is None or\
                        example_verb_images is None or\
                        subject_label <= 0 or\
                        object_label <= 0 or\
                        verb_label <= 0:
                    continue
                spatial_features.append(example_spatial_features)
                subject_box_features.append(self.backbone(example_subject_images.unsqueeze(0)).squeeze(0))
                object_box_features.append(self.backbone(example_object_images.unsqueeze(0)).squeeze(0))
                verb_box_features.append(self.backbone(example_verb_images.unsqueeze(0)).squeeze(0))
            
            subject_probs, _, verb_probs, _, object_probs, _ =\
                self.classifier.predict_score(
                    spatial_features,
                    subject_box_features,
                    verb_box_features,
                    object_box_features
                )
            results = self.detector.top3(subject_probs, verb_probs, object_probs, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
            for top3 in results['top3']:
                previous_confidence = 1.0
                for prediction in top3:
                    self.assertTrue(prediction[1] <= previous_confidence)
                    previous_confidence = prediction[1]
                    self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                    self.assertTrue((prediction[0][0] == 0 and prediction[0][1] != 0 and prediction[0][2] != 0) or prediction[1] == 0)

    def test_case_1_t2(self):
        with torch.no_grad():
            spatial_features = []
            subject_box_features = []
            object_box_features = []
            verb_box_features = []
            
            for example_spatial_features, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
                if example_spatial_features is None or\
                        example_subject_images is None or\
                        example_object_images is None or\
                        example_verb_images is None or\
                        subject_label <= 0 or\
                        object_label <= 0 or\
                        verb_label <= 0:
                    continue
                spatial_features.append(example_spatial_features)
                subject_box_features.append(self.backbone(example_subject_images.unsqueeze(0)).squeeze(0))
                object_box_features.append(self.backbone(example_object_images.unsqueeze(0)).squeeze(0))
                verb_box_features.append(self.backbone(example_verb_images.unsqueeze(0)).squeeze(0))
            
            subject_probs, _, verb_probs, _, object_probs, _ =\
                self.classifier.predict_score(
                    spatial_features,
                    subject_box_features,
                    verb_box_features,
                    object_box_features
                )
            results = self.detector.top3(subject_probs, verb_probs, object_probs, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
            for top3_idx, top3 in enumerate(results['top3']):
                previous_confidence = 1.0
                for prediction in top3:
                    self.assertTrue(prediction[1] <= previous_confidence)
                    previous_confidence = prediction[1]
                    self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                    self.assertTrue((prediction[0][0] != 0 and prediction[0][1] == 0 and prediction[0][2] != 0) or prediction[1] == 0)

    def test_case_1_t3(self):
        with torch.no_grad():
            spatial_features = []
            subject_box_features = []
            object_box_features = []
            verb_box_features = []
            
            for example_spatial_features, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
                if example_spatial_features is None or\
                        example_subject_images is None or\
                        example_object_images is None or\
                        example_verb_images is None or\
                        subject_label <= 0 or\
                        object_label <= 0 or\
                        verb_label <= 0:
                    continue
                spatial_features.append(example_spatial_features)
                subject_box_features.append(self.backbone(example_subject_images.unsqueeze(0)).squeeze(0))
                object_box_features.append(self.backbone(example_object_images.unsqueeze(0)).squeeze(0))
                verb_box_features.append(self.backbone(example_verb_images.unsqueeze(0)).squeeze(0))
            
            subject_probs, _, verb_probs, _, object_probs, _ =\
                self.classifier.predict_score(
                    spatial_features,
                    subject_box_features,
                    verb_box_features,
                    object_box_features
                )
            results = self.detector.top3(subject_probs, verb_probs, object_probs, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
            for top3 in results['top3']:
                previous_confidence = 1.0
                for prediction in top3:
                    self.assertTrue(prediction[1] <= previous_confidence)
                    previous_confidence = prediction[1]
                    self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                    self.assertTrue((prediction[0][0] != 0 and prediction[0][1] != 0 and prediction[0][2] == 0) or prediction[1] == 0)

    def test_case_1_t4(self):
        with torch.no_grad():
            spatial_features = []
            subject_box_features = []
            object_box_features = []
            verb_box_features = []
            
            for example_spatial_features, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
                if example_spatial_features is None or\
                        example_subject_images is None or\
                        example_object_images is None or\
                        example_verb_images is None or\
                        subject_label <= 0 or\
                        object_label <= 0 or\
                        verb_label <= 0:
                    continue
                spatial_features.append(example_spatial_features)
                subject_box_features.append(self.backbone(example_subject_images.unsqueeze(0)).squeeze(0))
                object_box_features.append(self.backbone(example_object_images.unsqueeze(0)).squeeze(0))
                verb_box_features.append(self.backbone(example_verb_images.unsqueeze(0)).squeeze(0))
            
            subject_probs, _, verb_probs, _, object_probs, _ =\
                self.classifier.predict_score(
                    spatial_features,
                    subject_box_features,
                    verb_box_features,
                    object_box_features
                )
            results = self.detector.top3(subject_probs, verb_probs, object_probs, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
            for top3 in results['top3']:
                previous_confidence = 1.0
                for prediction in top3:
                    self.assertTrue(prediction[1] <= previous_confidence)
                    previous_confidence = prediction[1]
                    self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                    self.assertTrue((prediction[0][0] != 0 and prediction[0][1] != 0 and prediction[0][2] != 0) or prediction[1] == 0)
    
    def test_case_1_novel_box(self):
        with torch.no_grad():
            spatial_features = []
            subject_box_features = []
            object_box_features = []
            verb_box_features = []
            
            for example_spatial_features, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
                if example_spatial_features is None or\
                        example_subject_images is None or\
                        example_object_images is None or\
                        example_verb_images is None or\
                        subject_label <= 0 or\
                        object_label <= 0 or\
                        verb_label <= 0:
                    continue
                spatial_features.append(example_spatial_features)
                subject_box_features.append(self.backbone(example_subject_images.unsqueeze(0)).squeeze(0))
                object_box_features.append(self.backbone(example_object_images.unsqueeze(0)).squeeze(0))
                verb_box_features.append(self.backbone(example_verb_images.unsqueeze(0)).squeeze(0))
                
            subject_probs, _, verb_probs, _, object_probs, _ =\
                self.classifier.predict_score(
                    spatial_features,
                    subject_box_features,
                    verb_box_features,
                    object_box_features
                )
            results = self.detector.top3(subject_probs, verb_probs, object_probs, torch.tensor([0.33, 0.33, 0.34, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
            for top3 in results['top3']:
                previous_confidence = 1.0
                for prediction in top3:
                    self.assertTrue(prediction[1] <= previous_confidence)
                    previous_confidence = prediction[1]
                    self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                    self.assertTrue((prediction[0][0] == 0 or prediction[0][1] == 0 or prediction[0][2] == 0) or prediction[1] == 0)
        
if __name__ == '__main__':
    unittest.main()
