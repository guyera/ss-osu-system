import torch
import argparse
import pickle
import os
import unittest

from torchvision.models import resnet50

from boximagedataset import BoxImageDataset
import boxclassifier
from backbone import Backbone

class SubjectImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        super().__init__()
        self.novelty_feature_dataset = novelty_feature_dataset
    
    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        _, labels, _, _, images, _, _, _ = self.novelty_feature_dataset[idx]
        return images, labels

class VerbImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        super().__init__()
        self.novelty_feature_dataset = novelty_feature_dataset
    
    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        spatial_encodings, _, _, labels, _, _, images, _ = self.novelty_feature_dataset[idx]
        return images, spatial_encodings, labels
        #return images, None, labels

class ObjectImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        super().__init__()
        self.novelty_feature_dataset = novelty_feature_dataset
    
    def __len__(self):
        return len(self.novelty_feature_dataset)
    
    def __getitem__(self, idx):
        _, _, labels, _, _, images, _, _ = self.novelty_feature_dataset[idx]
        return images, labels

class TestConfidenceCalibrationMethods(unittest.TestCase):
    def setUp(self):
        self.device = 'cuda:0'
        self.num_bins = 15
        
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
        
        full_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = './',
            csv_path = 'dataset_v4/dataset_v4_2_val.csv',
            training = False,
            image_batch_size = 16
        )

        subject_indices = []
        verb_indices = []
        object_indices = []
        for idx, (_, subject_label, object_label, verb_label, _, _, _, _) in enumerate(full_dataset):
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
        
        # Construct data loader
        self.subject_loader = torch.utils.data.DataLoader(
            dataset = subject_set,
            batch_size = 32,
            shuffle = False
        )
        self.object_loader = torch.utils.data.DataLoader(
            dataset = object_set,
            batch_size = 32,
            shuffle = False
        )
        self.verb_loader = torch.utils.data.DataLoader(
            dataset = verb_set,
            batch_size = 32,
            shuffle = False
        )
        
        module_state_dict = torch.load(os.path.join(
            pretrained_models_dir,
            'unsupervised_novelty_detection_module.pth'
        ))
        # Create classifier
        classifier = boxclassifier.ClassifierV2(256, 5, 12, 8, 72)
        classifier.load_state_dict(module_state_dict['module']['classifier'])
        self.classifier = classifier.to(self.device)
    
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
        with torch.no_grad():
            uncalibrated_subject_confidences = []
            uncalibrated_object_confidences = []
            uncalibrated_verb_confidences = []
            calibrated_subject_confidences = []
            calibrated_object_confidences = []
            calibrated_verb_confidences = []
            subject_correct = []
            object_correct = []
            verb_correct = []
            for images, labels in self.subject_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
            
                features = self.backbone(images)
                logits = self.classifier.predict_subject(features)
                
                uncalibrated_probabilities = torch.nn.functional.softmax(logits, dim = 1)
                calibrated_probabilities = self.classifier.confidence_calibrator.calibrate_subject(logits)
                
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

            for images, labels in self.object_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
            
                features = self.backbone(images)
                logits = self.classifier.predict_object(features)
                
                uncalibrated_probabilities = torch.nn.functional.softmax(logits, dim = 1)
                calibrated_probabilities = self.classifier.confidence_calibrator.calibrate_object(logits)
                
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

            for images, spatial_encodings, labels in self.verb_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
            
                features = self.backbone(images)
                if spatial_encodings is not None:
                    spatial_encodings = torch.flatten(spatial_encodings.to(self.device), start_dim = 1)
                    features = torch.cat((spatial_encodings, features), dim = 1)
                logits = self.classifier.predict_verb(features)
                
                uncalibrated_probabilities = torch.nn.functional.softmax(logits, dim = 1)
                calibrated_probabilities = self.classifier.confidence_calibrator.calibrate_verb(logits)
                
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
