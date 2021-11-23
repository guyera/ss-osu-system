import torch
import argparse
import pickle
import os
import math
import pandas

import noveltydetectionfeatures
import unsupervisednoveltydetection.common

import unittest

device = 'cuda:0'

# Create classifier
classifier = unsupervisednoveltydetection.common.Classifier(
    12544,
    12616,
    1024,
    5,
    12,
    8
)
        
classifier_state_dict = torch.load('unsupervisednoveltydetection/classifier.pth')
classifier.load_state_dict(classifier_state_dict)

classifier = classifier.to(device)

detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 5, 12, 8)
detector = detector.to(device)

detector_state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
detector.load_state_dict(detector_state_dict['module'])
    
testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = 'Custom',
    #csv_path = 'Custom/annotations/dataset_v3_val_temp.csv',
    csv_path = 'Custom/annotations/dataset_v3_val.csv',
    num_subj_cls = 5,
    num_obj_cls = 12,
    num_action_cls = 8,
    training = False,
    image_batch_size = 16,
    feature_extraction_device = device
)

idx=673 # Image ID = 00467
spatial_features, subject_appearance_features, object_appearance_features, verb_appearance_features, subject_label, object_label, verb_label = testing_set[idx]

spatial_features = spatial_features.to(device)
subject_appearance_features = subject_appearance_features.to(device)
object_appearance_features = object_appearance_features.to(device)
verb_appearance_features = verb_appearance_features.to(device)

subject_features = torch.flatten(subject_appearance_features).unsqueeze(0)
object_features = torch.flatten(object_appearance_features).unsqueeze(0)
verb_features = torch.cat((torch.flatten(spatial_features), torch.flatten(verb_appearance_features)), dim = 0).unsqueeze(0)

subject_label = subject_label.to(device)
object_label = object_label.to(device)
verb_label = verb_label.to(device)

print(subject_features.shape)

print(subject_label)
print(object_label)
print(verb_label)

subject_classifier_predictions = classifier.predict_subject(subject_features)
object_classifier_predictions = classifier.predict_object(object_features)
verb_classifier_predictions = classifier.predict_verb(verb_features)

print(subject_classifier_predictions)
print(object_classifier_predictions)
print(verb_classifier_predictions)

spatial_features = [spatial_features.squeeze(0)]
subject_appearance_features = [subject_appearance_features.squeeze(0)]
object_appearance_features = [object_appearance_features.squeeze(0)]
verb_appearance_features = [verb_appearance_features.squeeze(0)]

p_type = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], device = device)
detector_results = detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, p_type)
print(detector_results['top3'])
