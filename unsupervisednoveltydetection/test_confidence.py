import torch
import argparse
import pickle
import os

import noveltydetectionfeatures
import unsupervisednoveltydetection.common
from auc import compute_auc

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Max Classifier Logits for Anomaly Detection'
    )
    
    # Device parameters
    parser.add_argument(
        '--device',
        type = str,
        default = 'cpu'
    )

    # Data loading parameters
    parser.add_argument(
        '--dataset-name',
        type = str,
        default = 'Custom'
    )
    parser.add_argument(
        '--data-root',
        type = str,
        default = 'Custom'
    )
    parser.add_argument(
        '--training-csv-path',
        type = str,
        default = 'Custom/annotations/val_dataset_v1_train.csv'
    )
    parser.add_argument(
        '--testing-csv-path',
        type = str,
        default = 'Custom/annotations/val_dataset_v1_val.csv'
    )
    parser.add_argument(
        '--num-subj-cls',
        type = int,
        default = 6
    )
    parser.add_argument(
        '--num-obj-cls',
        type = int,
        default = 9
    )
    parser.add_argument(
        '--num-action-cls',
        type = int,
        default = 8
    )
    parser.add_argument(
        '--image-batch-size',
        type = int,
        default = 16
    )
    parser.add_argument(
        '--batch-size',
        type = int,
        default = 128
    )

    # Model persistence parameters
    parser.add_argument(
        '--anomaly-detector-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--calibrator-load-file',
        type = str,
        default = None
    )

    # Architectural parameters
    parser.add_argument(
        '--hidden-nodes',
        type = int,
        default = 1024
    )

    # Evaluation parameters
    parser.add_argument(
        '--num-bins',
        type = int,
        default = 15
    )

    args = parser.parse_args()

    return args

def _test(
        device,
        anomaly_detector,
        calibrator,
        testing_loader,
        num_bins):
    subject_confidences = []
    object_confidences = []
    verb_confidences = []
    subject_correct = []
    object_correct = []
    verb_correct = []
    for subject_features,\
            object_features,\
            verb_features,\
            subject_labels,\
            object_labels,\
            verb_labels in testing_loader:
        subject_features = subject_features.to(device)
        object_features = object_features.to(device)
        verb_features = verb_features.to(device)
        subject_labels = subject_labels.to(device)
        object_labels = object_labels.to(device)
        verb_labels = verb_labels.to(device)
        
        subject_logits, object_logits, verb_logits = anomaly_detector.predict(
            subject_features,
            object_features,
            verb_features
        )
        subject_probabilities, object_probabilities, verb_probabilities =\
            calibrator.calibrate(
                subject_logits,
                object_logits,
                verb_logits
            )
        batch_subject_confidences, subject_predictions =\
            torch.max(subject_probabilities, dim = 1)
        batch_object_confidences, object_predictions =\
            torch.max(object_probabilities, dim = 1)
        batch_verb_confidences, verb_predictions =\
            torch.max(verb_probabilities, dim = 1)
        
        subject_confidences.append(batch_subject_confidences)
        object_confidences.append(batch_object_confidences)
        verb_confidences.append(batch_verb_confidences)

        batch_subject_correct = subject_predictions == subject_labels
        batch_object_correct = object_predictions == object_labels
        batch_verb_correct = verb_predictions == verb_labels
        
        subject_correct.append(batch_subject_correct)
        object_correct.append(batch_object_correct)
        verb_correct.append(batch_verb_correct)

    subject_confidences = torch.cat(subject_confidences, dim = 0)
    object_confidences = torch.cat(object_confidences, dim = 0)
    verb_confidences = torch.cat(verb_confidences, dim = 0)
    
    subject_correct = torch.cat(subject_correct, dim = 0)
    object_correct = torch.cat(object_correct, dim = 0)
    verb_correct = torch.cat(verb_correct, dim = 0)

    # TODO Compute ECE
    bin_starts = torch.arange(num_bins, device = device, dtype = torch.float) / num_bins
    bin_ends = (torch.arange(num_bins, device = device, dtype = torch.float) + 1.0) / num_bins
    bin_ends[-1] += 0.000001
    subject_bin_memberships = torch.logical_and(
        subject_confidences.unsqueeze(1) >= bin_starts,
        subject_confidences.unsqueeze(1) < bin_ends
    ).to(torch.float)
    object_bin_memberships = torch.logical_and(
        object_confidences.unsqueeze(1) >= bin_starts,
        object_confidences.unsqueeze(1) < bin_ends
    ).to(torch.float)
    verb_bin_memberships = torch.logical_and(
        verb_confidences.unsqueeze(1) >= bin_starts,
        verb_confidences.unsqueeze(1) < bin_ends
    ).to(torch.float)
    
    subject_bin_counts = subject_bin_memberships.sum(dim = 0)
    object_bin_counts = object_bin_memberships.sum(dim = 0)
    verb_bin_counts = verb_bin_memberships.sum(dim = 0)
    
    subject_bin_weights = subject_bin_counts / (torch.sum(subject_bin_counts) + 0.000001)
    object_bin_weights = object_bin_counts / (torch.sum(object_bin_counts) + 0.000001)
    verb_bin_weights = verb_bin_counts / (torch.sum(verb_bin_counts) + 0.000001)

    subject_bin_correct = (subject_bin_memberships * subject_correct.unsqueeze(1)).sum(dim = 0)
    object_bin_correct = (object_bin_memberships * object_correct.unsqueeze(1)).sum(dim = 0)
    verb_bin_correct = (verb_bin_memberships * verb_correct.unsqueeze(1)).sum(dim = 0)

    subject_bin_accuracies = subject_bin_correct / (subject_bin_counts + 0.000001)
    object_bin_accuracies = object_bin_correct / (object_bin_counts + 0.000001)
    verb_bin_accuracies = verb_bin_correct / (verb_bin_counts + 0.000001)
    
    subject_bin_confidences = (subject_bin_memberships * subject_confidences.unsqueeze(1)).sum(dim = 0) / (subject_bin_counts + 0.000001)
    object_bin_confidences = (object_bin_memberships * object_confidences.unsqueeze(1)).sum(dim = 0) / (object_bin_counts + 0.000001)
    verb_bin_confidences = (verb_bin_memberships * verb_confidences.unsqueeze(1)).sum(dim = 0) / (verb_bin_counts + 0.000001)
    
    subject_bin_abs_errors = torch.abs(subject_bin_accuracies - subject_bin_confidences)
    object_bin_abs_errors = torch.abs(object_bin_accuracies - object_bin_confidences)
    verb_bin_abs_errors = torch.abs(verb_bin_accuracies - verb_bin_confidences)
    
    subject_bin_weighted_errors = subject_bin_abs_errors * subject_bin_weights
    object_bin_weighted_errors = object_bin_abs_errors * object_bin_weights
    verb_bin_weighted_errors = verb_bin_abs_errors * verb_bin_weights

    subject_ece = subject_bin_weighted_errors.sum()
    object_ece = object_bin_weighted_errors.sum()
    verb_ece = verb_bin_weighted_errors.sum()
    
    print(f'Subject ECE: {subject_ece}')
    print(f'Object ECE: {object_ece}')
    print(f'Verb ECE: {verb_ece}')

def main():
    args = parse_args()
    testing_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
        noveltydetectionfeatures.NoveltyFeatureDataset(
            name = args.dataset_name,
            data_root = args.data_root,
            csv_path = args.testing_csv_path,
            num_subj_cls = args.num_subj_cls,
            num_obj_cls = args.num_obj_cls,
            num_action_cls = args.num_action_cls,
            training = False,
            image_batch_size = args.image_batch_size,
            feature_extraction_device = args.device
        )
    )

    # The remaining datasets are used for evaluating AUC of subject, object,
    # and verb classifiers in the open set setting.
    
    # Construct data loader
    testing_loader = torch.utils.data.DataLoader(
        dataset = testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    
    # Get example features for shape information to construct classifiers
    appearance_features, _, verb_features, _, _, _ = testing_set[0]
    print(len(appearance_features))
    print(len(verb_features))

    # Create anomaly detector
    anomaly_detector = unsupervisednoveltydetection.common.AnomalyDetector(
        len(appearance_features),
        len(verb_features),
        args.hidden_nodes,
        args.num_subj_cls,
        args.num_obj_cls,
        args.num_action_cls
    )
    
    anomaly_detector_state_dict = torch.load(args.anomaly_detector_load_file)
    anomaly_detector.load_state_dict(anomaly_detector_state_dict)
    
    anomaly_detector = anomaly_detector.to(args.device)
    
    calibrator = unsupervisednoveltydetection.common.ConfidenceCalibrator().to(args.device)
    
    if args.calibrator_load_file is not None:
        calibrator_state_dict = torch.load(args.calibrator_load_file)
        calibrator.load_state_dict(calibrator_state_dict)
    
    with torch.no_grad():
        _test(
            args.device,
            anomaly_detector,
            calibrator,
            testing_loader,
            args.num_bins
        )

if __name__ == '__main__':
    main()
