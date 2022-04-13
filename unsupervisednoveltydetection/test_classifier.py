import torch
import argparse
import pickle
import os
import math

import noveltydetectionfeatures
import unsupervisednoveltydetection.common

from torchvision.models import resnet50

import unittest

class SubjectImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        super().__init__()
        self.novelty_feature_dataset = novelty_feature_dataset
    
    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        _, _, _, _, labels, _, _, images, _, _ = self.novelty_feature_dataset[idx]
        return images, labels

class VerbImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        super().__init__()
        self.novelty_feature_dataset = novelty_feature_dataset
    
    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        spatial_encodings, _, _, _, _, _, labels, _, _, images = self.novelty_feature_dataset[idx]
        return images, spatial_encodings, labels
        #return images, None, labels

class ObjectImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        super().__init__()
        self.novelty_feature_dataset = novelty_feature_dataset
    
    def __len__(self):
        return len(self.novelty_feature_dataset)
    
    def __getitem__(self, idx):
        _, _, _, _, _, labels, _, _, images, _ = self.novelty_feature_dataset[idx]
        return images, labels

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.num_bins = 15
        backbone = resnet50(pretrained = False)
        backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
        backbone_state_dict = torch.load('unsupervisednoveltydetection/backbone_2.pth')
        backbone.load_state_dict(backbone_state_dict)
        backbone = backbone.to(self.device)
        backbone.eval()
        self.backbone = backbone

        full_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = 'Custom',
            csv_path = 'Custom/annotations/dataset_v4_val.csv',
            training = False,
            image_batch_size = 16,
            backbone = backbone,
            feature_extraction_device = self.device
        )

        subject_indices = []
        verb_indices = []
        object_indices = []
        for idx, (_, _, _, _, subject_label, object_label, verb_label, _, _, _) in enumerate(full_dataset):
            # Remove novel examples
            if (subject_label is not None and subject_label.item() == 0) or (verb_label is not None and verb_label.item() == 0) or (object_label is not None and object_label.item() == 0):
                continue
            
            # Find non-null examples
            if subject_label is not None:
                subject_indices.append(idx)
            if verb_label is not None:
                verb_indices.append(idx)
            if object_label is not None:
                object_indices.append(idx)
        
        subject_set = SubjectImageDataset(torch.utils.data.Subset(full_dataset, subject_indices))
        verb_set = VerbImageDataset(torch.utils.data.Subset(full_dataset, verb_indices))
        object_set = ObjectImageDataset(torch.utils.data.Subset(full_dataset, object_indices))
    
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
        classifier = unsupervisednoveltydetection.common.ClassifierV2(256, 5, 12, 8, 72)
        classifier_state_dict = torch.load('unsupervisednoveltydetection/classifier_2.pth')
        classifier.load_state_dict(classifier_state_dict)
        self.classifier = classifier.to(self.device)
    
    def test_accuracy_acceptable(self):
        num_subject_correct = 0
        num_subject = 0
        num_object_correct = 0
        num_object = 0
        num_verb_correct = 0
        num_verb = 0
        for images, labels in self.subject_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.backbone(images)
            logits = self.classifier.predict_subject(features)
            
            predictions = torch.argmax(logits, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            batch_correct = predictions == labels
            num_subject_correct += batch_correct.to(torch.int).sum().cpu().item()
            num_subject += images.shape[0]
        
        for images, labels in self.object_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            features = self.backbone(images)
            logits = self.classifier.predict_object(features)
            
            predictions = torch.argmax(logits, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            batch_correct = predictions == labels
            num_object_correct += batch_correct.to(torch.int).sum().cpu().item()
            num_object += images.shape[0]
        
        for images, spatial_encodings, labels in self.verb_loader:
            images = images.to(self.device)
            spatial_encodings = spatial_encodings.to(self.device)
            labels = labels.to(self.device)
            
            features = self.backbone(images)
            features = torch.cat((torch.flatten(spatial_encodings, start_dim = 1), features), dim = 1)
            logits = self.classifier.predict_verb(features)
            
            predictions = torch.argmax(logits, dim = 1)

            # Shift predictions forward to allow for anomaly label = 0
            predictions += 1
            
            batch_correct = predictions == labels
            num_verb_correct += batch_correct.to(torch.int).sum().cpu().item()
            num_verb += images.shape[0]

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
