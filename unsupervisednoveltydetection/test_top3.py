import torch
from torchvision.models import resnet50, swin_t, swin_b

import unsupervisednoveltydetection
import noveltydetectionfeatures
import noveltydetection

import unittest

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'

        model_ = 'swin_t' # 'swin_t' 'swin_b'  'resnet'

        if model_ == 'resnet': 
            backbone = resnet50(pretrained = False)
            backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
        if model_ == 'swin_t': 
            backbone = swin_t() 
            backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
        if model_ == 'swin_b': 
            backbone = swin_b() 
            backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
        
        backbone_state_dict = torch.load('unsupervisednoveltydetection/' +model_ +'_backbone_2.pth')
        backbone.load_state_dict(backbone_state_dict)
        backbone = backbone.to(self.device)
        backbone.eval()
        self.backbone = backbone
        
        classifier = unsupervisednoveltydetection.ClassifierV2(256, 5, 12, 8, 72)
        detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(classifier, 5, 12, 8)
        self.detector = detector.to(self.device)
        state_dict = torch.load('unsupervisednoveltydetection/' +model_ +'_unsupervised_novelty_detection_module_2.pth')
        self.detector.load_state_dict(state_dict['module'])

        self.testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = 'Custom',
            csv_path = 'Custom/annotations/dataset_v4_val.csv',
            training = False,
            image_batch_size = 16,
            backbone = backbone,
            feature_extraction_device = self.device
        )

    def test_case_1_t1(self):
        with torch.no_grad():
            spatial_features = []
            subject_box_features = []
            object_box_features = []
            verb_box_features = []
            
            for example_spatial_features, _, _, _, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
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
            
            results = self.detector.top3(spatial_features, subject_box_features, verb_box_features, object_box_features, torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
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
            
            for example_spatial_features, _, _, _, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
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
            
            results = self.detector.top3(spatial_features, subject_box_features, verb_box_features, object_box_features, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
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
            
            for example_spatial_features, _, _, _, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
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
            
            results = self.detector.top3(spatial_features, subject_box_features, verb_box_features, object_box_features, torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
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
            
            for example_spatial_features, _, _, _, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
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
            
            results = self.detector.top3(spatial_features, subject_box_features, verb_box_features, object_box_features, torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
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
            
            for example_spatial_features, _, _, _, subject_label, object_label, verb_label, example_subject_images, example_object_images, example_verb_images, _ in self.testing_set:
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
                
            results = self.detector.top3(spatial_features, subject_box_features, verb_box_features, object_box_features, torch.tensor([0.33, 0.33, 0.34, 0.0, 0.0], device = self.device).unsqueeze(0).repeat(len(spatial_features), 1))
            for top3 in results['top3']:
                previous_confidence = 1.0
                for prediction in top3:
                    self.assertTrue(prediction[1] <= previous_confidence)
                    previous_confidence = prediction[1]
                    self.assertTrue(prediction[0][0] != -1 and prediction[0][1] != -1 and prediction[0][2] != -1)
                    self.assertTrue((prediction[0][0] == 0 or prediction[0][1] == 0 or prediction[0][2] == 0) or prediction[1] == 0)
        
if __name__ == '__main__':
    unittest.main()
