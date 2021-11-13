import argparse
import pickle
import os

import torch
from matplotlib import pyplot as plt
import numpy as np
import scipy.interpolate
import sklearn.metrics

import noveltydetectionfeatures
import noveltydetection
import unsupervisednoveltydetection.common

def plot_score_distributions(
        scores,
        labels,
        bins,
        save_path,
        plot_title,
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0):
    plt.figure()

    if normalize:
        plt.suptitle('Anomaly score distributions')
    else:
        plt.suptitle('Anomaly score histogram')
    plt.title(plot_title)
    plt.xlabel(plot_xlabel)

    for group_idx, score_group in enumerate(scores):
        p, x = np.histogram(score_group.detach().cpu().numpy(), bins = bins)
        x = x[:-1] + (x[1] - x[0]) / 2
        if normalize:
            trapezoid_widths = x[1] - x[0]
            trapezoid_left_heights = p[:-1]
            trapezoid_right_heights = p[1:]
            trapezoid_areas = (trapezoid_left_heights + trapezoid_right_heights) * trapezoid_widths / 2
            p = p / np.sum(trapezoid_areas)
        f = scipy.interpolate.UnivariateSpline(x, p, s = smoothing_factor)
        plt.plot(x, f(x), label = labels[group_idx])

    plt.legend()
    plt.savefig(save_path)

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
        default = 'Custom/annotations/dataset_v3_train.csv'
    )
    parser.add_argument(
        '--testing-csv-path',
        type = str,
        default = 'Custom/annotations/dataset_v3_val.csv'
    )
    parser.add_argument(
        '--num-subj-cls',
        type = int,
        default = 5
    )
    parser.add_argument(
        '--num-obj-cls',
        type = int,
        default = 12
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

    # Architectural parameters
    parser.add_argument(
        '--hidden-nodes',
        type = int,
        default = 1024
    )
    
    # Training parameters
    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01
    )
    parser.add_argument(
        '--weight-decay',
        type = float,
        default = 0.0001
    )
    parser.add_argument(
        '--epochs',
        type = int,
        default = 300
    )

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # Construct training dataset
    training_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
        noveltydetectionfeatures.NoveltyFeatureDataset(
            name = args.dataset_name,
            data_root = args.data_root,
            csv_path = args.training_csv_path,
            num_subj_cls = args.num_subj_cls,
            num_obj_cls = args.num_obj_cls,
            num_action_cls = args.num_action_cls,
            training = True,
            image_batch_size = args.image_batch_size,
            feature_extraction_device = args.device
        )
    )
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
    
    ### Start with held-out split with fewer ID classes

    # Hard-code held-out novel tuples
    
    '''
    # Cat wearing tie, cat wearing bag, cat catching frisbee, cat riding bike,
    # horse wearing tie, horse pulling carriage
    ood_subject_tuples = [
        (3, 2, 4),
        (3, 2, 2),
        (3, 4, 1),
        (3, 3, 3),
        (4, 2, 4),
        (4, 5, 7)
    ]
    
    # Cat wearing bag, dog wearing bag, person carrying bag, horse pulling hay,
    # person pulling hay
    ood_object_tuples = [
        (3, 2, 2),
        (2, 2, 2),
        (1, 1, 2),
        (4, 5, 8),
        (1, 5, 8)
    ]
    
    # Dog carrying frisbee, person carrying frisbee, horse carrying hay,
    # horse eating hay
    ood_verb_tuples = [
        (2, 1, 1),
        (1, 1, 1),
        (4, 1, 8),
        (4, 6, 8)
    ]
    '''

    # horse wearing tie, horse pulling carriage
    ood_subject_tuples = [
        (4, 2, 4),
        (4, 5, 7)
    ]
    
    # horse pulling hay, person pulling hay
    ood_object_tuples = [
        (4, 5, 8),
        (1, 5, 8)
    ]
    
    # horse eating hay
    ood_verb_tuples = [
        (4, 6, 8)
    ]
    
    # Get tuples of held-out OOD tuples
    ood_subject_tuple_indices = unsupervisednoveltydetection.common.get_indices_of_tuples(training_set, ood_subject_tuples)
    ood_object_tuple_indices = unsupervisednoveltydetection.common.get_indices_of_tuples(training_set, ood_object_tuples)
    ood_verb_tuple_indices = unsupervisednoveltydetection.common.get_indices_of_tuples(training_set, ood_verb_tuples)
    
    # Construct held-out OOD sets
    ood_subject_tuple_set = torch.utils.data.Subset(training_set, ood_subject_tuple_indices)
    ood_object_tuple_set = torch.utils.data.Subset(training_set, ood_object_tuple_indices)
    ood_verb_tuple_set = torch.utils.data.Subset(training_set, ood_verb_tuple_indices)
    ood_subject_set = unsupervisednoveltydetection.common.SubjectDataset(ood_subject_tuple_set, train = True)
    ood_object_set = unsupervisednoveltydetection.common.ObjectDataset(ood_object_tuple_set, train = True)
    ood_verb_set = unsupervisednoveltydetection.common.VerbDataset(ood_verb_tuple_set, train = True)
    
    # Construct individual S/V/O sets of all training and testing data
    subject_training_set = unsupervisednoveltydetection.common.SubjectDataset(training_set, train = True)
    object_training_set = unsupervisednoveltydetection.common.ObjectDataset(training_set, train = True)
    verb_training_set = unsupervisednoveltydetection.common.VerbDataset(training_set, train = True)
    subject_testing_set = unsupervisednoveltydetection.common.SubjectDataset(testing_set, train = True)
    object_testing_set = unsupervisednoveltydetection.common.ObjectDataset(testing_set, train = True)
    verb_testing_set = unsupervisednoveltydetection.common.VerbDataset(testing_set, train = True)
    
    # Init individual ID S/V/O labels
    id_subject_labels = [x for x in range(1, args.num_subj_cls) if x not in [3, 4]]
    id_object_labels = [x for x in range(1, args.num_obj_cls) if x not in [2, 8]]
    id_verb_labels = [x for x in range(1, args.num_action_cls) if x not in [1, 6]]

    # Get indices of ID individual boxes in training and testing sets
    id_subject_training_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_training_set, id_subject_labels)
    id_object_training_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_training_set, id_object_labels)
    id_verb_training_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_training_set, id_verb_labels)
    id_subject_testing_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_testing_set, id_subject_labels)
    id_object_testing_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_testing_set, id_object_labels)
    id_verb_testing_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_testing_set, id_verb_labels)
    
    # Construct ID individual box S/V/O training and testing sets
    id_subject_training_set = torch.utils.data.Subset(subject_training_set, id_subject_training_indices)
    id_object_training_set = torch.utils.data.Subset(object_training_set, id_object_training_indices)
    id_verb_training_set = torch.utils.data.Subset(verb_training_set, id_verb_training_indices)
    id_subject_testing_set = torch.utils.data.Subset(subject_testing_set, id_subject_testing_indices)
    id_object_testing_set = torch.utils.data.Subset(object_testing_set, id_object_testing_indices)
    id_verb_testing_set = torch.utils.data.Subset(verb_testing_set, id_verb_testing_indices)
    
    # Remap labels in ID data
    id_subject_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_subject_training_set, id_subject_labels)
    id_object_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_object_training_set, id_object_labels)
    id_verb_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_verb_training_set, id_verb_labels)
    id_subject_testing_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_subject_testing_set, id_subject_labels)
    id_object_testing_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_object_testing_set, id_object_labels)
    id_verb_testing_set = unsupervisednoveltydetection.common.LabelMappingDataset(id_verb_testing_set, id_verb_labels)
    
    # Shift labels up 1; classifier assumes label 0 is for anomalies
    target_transform = unsupervisednoveltydetection.common.UpshiftTargetTransform()
    id_subject_training_set = unsupervisednoveltydetection.common.TransformingDataset(id_subject_training_set, target_transform = target_transform)
    id_object_training_set = unsupervisednoveltydetection.common.TransformingDataset(id_object_training_set, target_transform = target_transform)
    id_verb_training_set = unsupervisednoveltydetection.common.TransformingDataset(id_verb_training_set, target_transform = target_transform)
    id_subject_testing_set = unsupervisednoveltydetection.common.TransformingDataset(id_subject_testing_set, target_transform = target_transform)
    id_object_testing_set = unsupervisednoveltydetection.common.TransformingDataset(id_object_testing_set, target_transform = target_transform)
    id_verb_testing_set = unsupervisednoveltydetection.common.TransformingDataset(id_verb_testing_set, target_transform = target_transform)

    # And now repeat with the held-out split with more ID classes
    
    # Hard-code held-out novel tuples
    
    # Cat wearing tie, cat wearing bag, cat catching frisbee, cat riding bike
    super_ood_subject_tuples = [
        (3, 2, 4),
        (3, 2, 2),
        (3, 4, 1),
        (3, 3, 3)
    ]
    
    # Cat wearing bag, dog wearing bag, person carrying bag
    super_ood_object_tuples = [
        (3, 2, 2),
        (2, 2, 2),
        (1, 1, 2)
    ]
    
    # Dog carrying frisbee, person carrying frisbee, horse carrying hay
    super_ood_verb_tuples = [
        (2, 1, 1),
        (1, 1, 1),
        (4, 1, 8)
    ]
    
    # Get tuples of held-out OOD tuples
    super_ood_subject_tuple_indices = unsupervisednoveltydetection.common.get_indices_of_tuples(training_set, super_ood_subject_tuples)
    super_ood_object_tuple_indices = unsupervisednoveltydetection.common.get_indices_of_tuples(training_set, super_ood_object_tuples)
    super_ood_verb_tuple_indices = unsupervisednoveltydetection.common.get_indices_of_tuples(training_set, super_ood_verb_tuples)
    
    # Construct held-out OOD sets
    super_ood_subject_tuple_set = torch.utils.data.Subset(training_set, super_ood_subject_tuple_indices)
    super_ood_object_tuple_set = torch.utils.data.Subset(training_set, super_ood_object_tuple_indices)
    super_ood_verb_tuple_set = torch.utils.data.Subset(training_set, super_ood_verb_tuple_indices)
    super_ood_subject_set = unsupervisednoveltydetection.common.SubjectDataset(super_ood_subject_tuple_set, train = True)
    super_ood_object_set = unsupervisednoveltydetection.common.ObjectDataset(super_ood_object_tuple_set, train = True)
    super_ood_verb_set = unsupervisednoveltydetection.common.VerbDataset(super_ood_verb_tuple_set, train = True)
    
    # Init individual ID S/V/O labels
    super_id_subject_labels = [x for x in range(1, args.num_subj_cls) if x not in [3]]
    super_id_object_labels = [x for x in range(1, args.num_obj_cls) if x not in [2]]
    super_id_verb_labels = [x for x in range(1, args.num_action_cls) if x not in [1]]

    # Get indices of ID individual boxes in training and testing sets
    super_id_subject_training_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_training_set, super_id_subject_labels)
    super_id_object_training_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_training_set, super_id_object_labels)
    super_id_verb_training_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_training_set, super_id_verb_labels)
    super_id_subject_testing_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_testing_set, super_id_subject_labels)
    super_id_object_testing_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_testing_set, super_id_object_labels)
    super_id_verb_testing_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_testing_set, super_id_verb_labels)
    
    # Construct ID individual box S/V/O training and testing sets
    super_id_subject_training_set = torch.utils.data.Subset(subject_training_set, super_id_subject_training_indices)
    super_id_object_training_set = torch.utils.data.Subset(object_training_set, super_id_object_training_indices)
    super_id_verb_training_set = torch.utils.data.Subset(verb_training_set, super_id_verb_training_indices)
    super_id_subject_testing_set = torch.utils.data.Subset(subject_testing_set, super_id_subject_testing_indices)
    super_id_object_testing_set = torch.utils.data.Subset(object_testing_set, super_id_object_testing_indices)
    super_id_verb_testing_set = torch.utils.data.Subset(verb_testing_set, super_id_verb_testing_indices)
    
    # Remap labels in ID data
    super_id_subject_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(super_id_subject_training_set, super_id_subject_labels)
    super_id_object_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(super_id_object_training_set, super_id_object_labels)
    super_id_verb_training_set = unsupervisednoveltydetection.common.LabelMappingDataset(super_id_verb_training_set, super_id_verb_labels)
    super_id_subject_testing_set = unsupervisednoveltydetection.common.LabelMappingDataset(super_id_subject_testing_set, super_id_subject_labels)
    super_id_object_testing_set = unsupervisednoveltydetection.common.LabelMappingDataset(super_id_object_testing_set, super_id_object_labels)
    super_id_verb_testing_set = unsupervisednoveltydetection.common.LabelMappingDataset(super_id_verb_testing_set, super_id_verb_labels)
    
    # Shift labels up 1; classifier assumes label 0 is for anomalies
    super_id_subject_training_set = unsupervisednoveltydetection.common.TransformingDataset(super_id_subject_training_set, target_transform = target_transform)
    super_id_object_training_set = unsupervisednoveltydetection.common.TransformingDataset(super_id_object_training_set, target_transform = target_transform)
    super_id_verb_training_set = unsupervisednoveltydetection.common.TransformingDataset(super_id_verb_training_set, target_transform = target_transform)
    super_id_subject_testing_set = unsupervisednoveltydetection.common.TransformingDataset(super_id_subject_testing_set, target_transform = target_transform)
    super_id_object_testing_set = unsupervisednoveltydetection.common.TransformingDataset(super_id_object_testing_set, target_transform = target_transform)
    super_id_verb_testing_set = unsupervisednoveltydetection.common.TransformingDataset(super_id_verb_testing_set, target_transform = target_transform)

    # Construct data loaders
    id_subject_training_loader = torch.utils.data.DataLoader(
        dataset = id_subject_training_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    id_object_training_loader = torch.utils.data.DataLoader(
        dataset = id_object_training_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    id_verb_training_loader = torch.utils.data.DataLoader(
        dataset = id_verb_training_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    id_subject_testing_loader = torch.utils.data.DataLoader(
        dataset = id_subject_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    id_object_testing_loader = torch.utils.data.DataLoader(
        dataset = id_object_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    id_verb_testing_loader = torch.utils.data.DataLoader(
        dataset = id_verb_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    ood_subject_loader = torch.utils.data.DataLoader(
        dataset = ood_subject_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    ood_object_loader = torch.utils.data.DataLoader(
        dataset = ood_object_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    ood_verb_loader = torch.utils.data.DataLoader(
        dataset = ood_verb_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_id_subject_training_loader = torch.utils.data.DataLoader(
        dataset = super_id_subject_training_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_id_object_training_loader = torch.utils.data.DataLoader(
        dataset = super_id_object_training_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_id_verb_training_loader = torch.utils.data.DataLoader(
        dataset = super_id_verb_training_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_id_subject_testing_loader = torch.utils.data.DataLoader(
        dataset = super_id_subject_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_id_object_testing_loader = torch.utils.data.DataLoader(
        dataset = super_id_object_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_id_verb_testing_loader = torch.utils.data.DataLoader(
        dataset = super_id_verb_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_ood_subject_loader = torch.utils.data.DataLoader(
        dataset = super_ood_subject_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_ood_object_loader = torch.utils.data.DataLoader(
        dataset = super_ood_object_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    super_ood_verb_loader = torch.utils.data.DataLoader(
        dataset = super_ood_verb_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    
    # Create classifier for split with fewer ID classes
    classifier = unsupervisednoveltydetection.common.Classifier(
        12544,
        12616,
        args.hidden_nodes,
        len(id_subject_labels) + 1, # + 1 for novel label 0
        len(id_object_labels) + 1, # + 1 for novel label 0
        len(id_verb_labels) + 1 # + 1 for novel label 0
    ).to(args.device)
    
    # Relocate classifier to given device
    classifier = classifier.to(args.device)
    
    # Fit classifier to ID training data
    classifier.fit(
        args.lr,
        args.weight_decay,
        args.epochs,
        id_subject_training_loader,
        id_object_training_loader,
        id_verb_training_loader,
    )

    num_correct = None
    num = 0
    for features, targets in id_subject_testing_loader:
        features = features.to(args.device)
        targets = targets.to(args.device)
        logits = classifier.predict_subject(features)
        predictions = torch.argmax(logits, dim = 1) + 1
        cur_num_correct = (predictions == targets).int().sum()
        if num_correct is None:
            num_correct = cur_num_correct
        else:
            num_correct += cur_num_correct
        num += len(targets)
    accuracy = num_correct / float(num)
    print(f'Subject testing accuracy: {accuracy}')

    # Create classifier for split with more ID classes
    super_classifier = unsupervisednoveltydetection.common.Classifier(
        12544,
        12616,
        args.hidden_nodes,
        len(super_id_subject_labels) + 1, # + 1 for novel label 0
        len(super_id_object_labels) + 1, # + 1 for novel label 0
        len(super_id_verb_labels) + 1 # + 1 for novel label 0
    ).to(args.device)
    
    # Relocate classifier to given device
    super_classifier = super_classifier.to(args.device)
    
    # Fit classifier to ID training data
    super_classifier.fit(
        args.lr,
        args.weight_decay,
        args.epochs,
        super_id_subject_training_loader,
        super_id_object_training_loader,
        super_id_verb_training_loader,
    )
    
    # We need to compute score distributions for both splits, and then use
    # the ID / OOD score distribution separation for the split with fewer
    # ID classes to try to predict the OOD score distribution statistics for the
    # split with more classes. We'll use the REAL OOD score distribution
    # statistics to measure our success. However, these scores have to be
    # computed over the same data, or else the distributions and their
    # statistics won't be comparable. So we'll use the ID classes from the
    # split with fewer ID classes, and the OOD classes from the split with
    # more ID classes (i.e. we'll leave out the classes which change roles 
    # between splits). That way, we're constructing shift and scale statistics
    # from some ID / OOD score distributions and applying them to an ID
    # score distribution from the same ID classes, and comparing our results
    # to an OOD score distribution from the same OOD classes.

    # Compute ID scores from classifier trained on split with fewer ID classes
    id_subject_scores = []
    id_object_scores = []
    id_verb_scores = []
    for features, targets in id_subject_testing_loader:
        features = features.to(args.device)
        id_subject_scores.append(classifier.score_subject(features))
    for features, _ in id_object_testing_loader:
        features = features.to(args.device)
        id_object_scores.append(classifier.score_object(features))
    for features, _ in id_verb_testing_loader:
        features = features.to(args.device)
        id_verb_scores.append(classifier.score_verb(features))
    id_subject_scores = torch.cat(id_subject_scores, dim = 0)
    id_object_scores = torch.cat(id_object_scores, dim = 0)
    id_verb_scores = torch.cat(id_verb_scores, dim = 0)

    # Compute OOD scores from classifier trained on split with fewer ID classes
    ood_subject_scores = []
    ood_object_scores = []
    ood_verb_scores = []
    for features, targets in ood_subject_loader:
        features = features.to(args.device)
        ood_subject_scores.append(classifier.score_subject(features))
    for features, _ in ood_object_loader:
        features = features.to(args.device)
        ood_object_scores.append(classifier.score_object(features))
    for features, _ in ood_verb_loader:
        features = features.to(args.device)
        ood_verb_scores.append(classifier.score_verb(features))
    ood_subject_scores = torch.cat(ood_subject_scores, dim = 0)
    ood_object_scores = torch.cat(ood_object_scores, dim = 0)
    ood_verb_scores = torch.cat(ood_verb_scores, dim = 0)

    # Compute ID scores from classifier trained on split with more ID classes
    super_id_subject_scores = []
    super_id_object_scores = []
    super_id_verb_scores = []
    for features, _ in super_id_subject_testing_loader:
        features = features.to(args.device)
        super_id_subject_scores.append(super_classifier.score_subject(features))
    for features, _ in super_id_object_testing_loader:
        features = features.to(args.device)
        super_id_object_scores.append(super_classifier.score_object(features))
    for features, _ in super_id_verb_testing_loader:
        features = features.to(args.device)
        super_id_verb_scores.append(super_classifier.score_verb(features))
    super_id_subject_scores = torch.cat(super_id_subject_scores, dim = 0)
    super_id_object_scores = torch.cat(super_id_object_scores, dim = 0)
    super_id_verb_scores = torch.cat(super_id_verb_scores, dim = 0)
    
    # Compute OOD scores from classifier trained on split with more ID classes
    super_ood_subject_scores = []
    super_ood_object_scores = []
    super_ood_verb_scores = []
    for features, _ in super_ood_subject_loader:
        features = features.to(args.device)
        super_ood_subject_scores.append(super_classifier.score_subject(features))
    for features, _ in super_ood_object_loader:
        features = features.to(args.device)
        super_ood_object_scores.append(super_classifier.score_object(features))
    for features, _ in super_ood_verb_loader:
        features = features.to(args.device)
        super_ood_verb_scores.append(super_classifier.score_verb(features))
    super_ood_subject_scores = torch.cat(super_ood_subject_scores, dim = 0)
    super_ood_object_scores = torch.cat(super_ood_object_scores, dim = 0)
    super_ood_verb_scores = torch.cat(super_ood_verb_scores, dim = 0)
    
    # Compute score distribution statistics for split with fewer ID classes
    id_subject_score_mean = torch.mean(id_subject_scores)
    id_object_score_mean = torch.mean(id_object_scores)
    id_verb_score_mean = torch.mean(id_verb_scores)
    id_subject_score_std = torch.std(id_subject_scores, unbiased = True)
    id_object_score_std = torch.std(id_object_scores, unbiased = True)
    id_verb_score_std = torch.std(id_verb_scores, unbiased = True)
    ood_subject_score_mean = torch.mean(ood_subject_scores)
    ood_object_score_mean = torch.mean(ood_object_scores)
    ood_verb_score_mean = torch.mean(ood_verb_scores)
    ood_subject_score_std = torch.std(ood_subject_scores, unbiased = True)
    ood_object_score_std = torch.std(ood_object_scores, unbiased = True)
    ood_verb_score_std = torch.std(ood_verb_scores, unbiased = True)
    
    # Compute score distribution statistics for split with more ID classes
    super_id_subject_score_mean = torch.mean(super_id_subject_scores)
    super_id_object_score_mean = torch.mean(super_id_object_scores)
    super_id_verb_score_mean = torch.mean(super_id_verb_scores)
    super_id_subject_score_std = torch.std(super_id_subject_scores, unbiased = True)
    super_id_object_score_std = torch.std(super_id_object_scores, unbiased = True)
    super_id_verb_score_std = torch.std(super_id_verb_scores, unbiased = True)
    super_ood_subject_score_mean = torch.mean(super_ood_subject_scores)
    super_ood_object_score_mean = torch.mean(super_ood_object_scores)
    super_ood_verb_score_mean = torch.mean(super_ood_verb_scores)
    super_ood_subject_score_std = torch.std(super_ood_subject_scores, unbiased = True)
    super_ood_object_score_std = torch.std(super_ood_object_scores, unbiased = True)
    super_ood_verb_score_std = torch.std(super_ood_verb_scores, unbiased = True)

    # Compute shift of ood distribution mean from id distribution mean
    # normalized by id distribution standard deviation, for split with fewer ID
    # classes
    subject_score_ood_shift = (ood_subject_score_mean - id_subject_score_mean) / id_subject_score_std
    object_score_ood_shift = (ood_object_score_mean - id_object_score_mean) / id_object_score_std
    verb_score_ood_shift = (ood_verb_score_mean - id_verb_score_mean) / id_verb_score_std

    # Compute scale of ood distribution standard deviation from id distribution
    # standard deviation, for split with fewer ID classes
    subject_score_ood_scale = ood_subject_score_std / id_subject_score_std
    object_score_ood_scale = ood_object_score_std / id_object_score_std
    verb_score_ood_scale = ood_verb_score_std / id_verb_score_std

    # Shift and scale ID distribution from the split with more ID classes using
    # the shift and scale statistics from the split with fewer ID classes to get
    # supposed OOD score distribution statistics for the split with more ID
    # classes
    predicted_super_ood_subject_score_mean = super_id_subject_score_mean + subject_score_ood_shift * super_id_subject_score_std
    predicted_super_ood_object_score_mean = super_id_object_score_mean + object_score_ood_shift * super_id_object_score_std
    predicted_super_ood_verb_score_mean = super_id_verb_score_mean + verb_score_ood_shift * super_id_verb_score_std
    predicted_super_ood_subject_score_std = super_id_subject_score_std * subject_score_ood_scale
    predicted_super_ood_object_score_std = super_id_object_score_std * object_score_ood_scale
    predicted_super_ood_verb_score_std = super_id_verb_score_std * verb_score_ood_scale

    # Now to see if the predicted statistics align with the actual statistics:
    print(f'OOD subject mean: {ood_subject_score_mean.cpu().item()} | OOD subject std: {ood_subject_score_std.cpu().item()}')
    print(f'Predicted super OOD subject mean: {predicted_super_ood_subject_score_mean.cpu().item()} | Predicted super OOD subject std: {predicted_super_ood_subject_score_std.cpu().item()}')
    print(f'Super OOD subject mean: {super_ood_subject_score_mean.cpu().item()} | Super OOD subject std: {super_ood_subject_score_std.cpu().item()}')
    print()
    print(f'OOD object mean: {ood_object_score_mean.cpu().item()} | OOD object std: {ood_object_score_std.cpu().item()}')
    print(f'Predicted super OOD object mean: {predicted_super_ood_object_score_mean.cpu().item()} | Predicted super OOD object std: {predicted_super_ood_object_score_std.cpu().item()}')
    print(f'Super OOD object mean: {super_ood_object_score_mean.cpu().item()} | Super OOD object std: {super_ood_object_score_std.cpu().item()}')
    print()
    print(f'OOD verb mean: {ood_verb_score_mean.cpu().item()} | OOD verb std: {ood_verb_score_std.cpu().item()}')
    print(f'Predicted super OOD verb mean: {predicted_super_ood_verb_score_mean.cpu().item()} | Predicted super OOD verb std: {predicted_super_ood_verb_score_std.cpu().item()}')
    print(f'Super OOD verb mean: {super_ood_verb_score_mean.cpu().item()} | Super OOD verb std: {super_ood_verb_score_std.cpu().item()}')
    
    # Shift and rescale OOD scores from the split with fewer ID classes to match
    # the predicted mean and standard deviation of the OOD score distribution
    # for the split with more ID classes
    centered_ood_subject_scores = (ood_subject_scores - ood_subject_score_mean) / ood_subject_score_std
    adjusted_ood_subject_scores = centered_ood_subject_scores * predicted_super_ood_subject_score_std + predicted_super_ood_subject_score_mean
    centered_ood_object_scores = (ood_object_scores - ood_object_score_mean) / ood_object_score_std
    adjusted_ood_object_scores = centered_ood_object_scores * predicted_super_ood_object_score_std + predicted_super_ood_object_score_mean
    centered_ood_verb_scores = (ood_verb_scores - ood_verb_score_mean) / ood_verb_score_std
    adjusted_ood_verb_scores = centered_ood_verb_scores * predicted_super_ood_verb_score_std + predicted_super_ood_verb_score_mean

    # Plot distributions and save to PNG files for visualization
    plot_scores = (ood_subject_scores, super_ood_subject_scores, adjusted_ood_subject_scores)
    plot_labels = ('Held-out OOD Subject Scores', 'Target OOD Subject Scores', 'Predicted OOD Subject Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/subject_score_distributions_1.png',
        f'Subject Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (ood_object_scores, super_ood_object_scores, adjusted_ood_object_scores)
    plot_labels = ('Held-out OOD Object Scores', 'Target OOD Object Scores', 'Predicted OOD Object Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/object_score_distributions_1.png',
        f'Object Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (ood_verb_scores, super_ood_verb_scores, adjusted_ood_verb_scores)
    plot_labels = ('Held-out OOD Verb Scores', 'Target OOD Verb Scores', 'Predicted OOD Verb Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/verb_score_distributions_1.png',
        f'Verb Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )
    
    centered_super_id_subject_scores = (super_id_subject_scores - super_id_subject_score_mean) / super_id_subject_score_std
    adjusted_super_id_subject_scores = centered_super_id_subject_scores * predicted_super_ood_subject_score_std + predicted_super_ood_subject_score_mean
    centered_super_id_object_scores = (super_id_object_scores - super_id_object_score_mean) / super_id_object_score_std
    adjusted_super_id_object_scores = centered_super_id_object_scores * predicted_super_ood_object_score_std + predicted_super_ood_object_score_mean
    centered_super_id_verb_scores = (super_id_verb_scores - super_id_verb_score_mean) / super_id_verb_score_std
    adjusted_super_id_verb_scores = centered_super_id_verb_scores * predicted_super_ood_verb_score_std + predicted_super_ood_verb_score_mean
    
    # Plot distributions and save to PNG files for visualization
    plot_scores = (ood_subject_scores, super_ood_subject_scores, adjusted_super_id_subject_scores)
    plot_labels = ('Held-out OOD Subject Scores', 'Target OOD Subject Scores', 'Predicted OOD Subject Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/subject_score_distributions_2.png',
        f'Subject Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (ood_object_scores, super_ood_object_scores, adjusted_super_id_object_scores)
    plot_labels = ('Held-out OOD Object Scores', 'Target OOD Object Scores', 'Predicted OOD Object Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/object_score_distributions_2.png',
        f'Object Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (ood_verb_scores, super_ood_verb_scores, adjusted_super_id_verb_scores)
    plot_labels = ('Held-out OOD Verb Scores', 'Target OOD Verb Scores', 'Predicted OOD Verb Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/verb_score_distributions_2.png',
        f'Verb Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (id_subject_scores, ood_subject_scores)
    plot_labels = ('ID Subject scores', 'OOD Subject Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/held_out_subject_distributions.png',
        f'Held-out Subject Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (id_object_scores, ood_object_scores)
    plot_labels = ('ID Object scores', 'OOD Object Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/held_out_object_distributions.png',
        f'Held-out Object Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (id_verb_scores, ood_verb_scores)
    plot_labels = ('ID Verb scores', 'OOD Verb Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/held_out_verb_distributions.png',
        f'Held-out Verb Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (super_id_subject_scores, super_ood_subject_scores)
    plot_labels = ('ID Subject scores', 'OOD Subject Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/non_held_out_subject_distributions.png',
        f'Non-held-out Subject Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (super_id_object_scores, super_ood_object_scores)
    plot_labels = ('ID Object scores', 'OOD Object Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/non_held_out_object_distributions.png',
        f'Non-held-out Object Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )

    plot_scores = (super_id_verb_scores, super_ood_verb_scores)
    plot_labels = ('ID Verb scores', 'OOD Verb Scores')
    plot_score_distributions(
        plot_scores,
        plot_labels,
        20,
        f'unsupervisednoveltydetection/non_held_out_verb_distributions.png',
        f'Non-held-out Verb Scores',
        plot_xlabel = 'Anomaly score',
        normalize = True,
        smoothing_factor = 0
    )
    
    subject_scores = torch.cat((id_subject_scores, ood_subject_scores), dim = 0)
    subject_trues = torch.cat((torch.zeros_like(id_subject_scores), torch.ones_like(ood_subject_scores)), dim = 0)
    subject_auc = sklearn.metrics.roc_auc_score(subject_trues.detach().cpu().numpy(), subject_scores.detach().cpu().numpy())
    print(f'Subject AUC: {subject_auc}')
    object_scores = torch.cat((id_object_scores, ood_object_scores), dim = 0)
    object_trues = torch.cat((torch.zeros_like(id_object_scores), torch.ones_like(ood_object_scores)), dim = 0)
    object_auc = sklearn.metrics.roc_auc_score(object_trues.detach().cpu().numpy(), object_scores.detach().cpu().numpy())
    print(f'Object AUC: {object_auc}')
    verb_scores = torch.cat((id_verb_scores, ood_verb_scores), dim = 0)
    verb_trues = torch.cat((torch.zeros_like(id_verb_scores), torch.ones_like(ood_verb_scores)), dim = 0)
    verb_auc = sklearn.metrics.roc_auc_score(verb_trues.detach().cpu().numpy(), verb_scores.detach().cpu().numpy())
    print(f'Verb AUC: {verb_auc}')

    super_subject_scores = torch.cat((super_id_subject_scores, super_ood_subject_scores), dim = 0)
    super_subject_trues = torch.cat((torch.zeros_like(super_id_subject_scores), torch.ones_like(super_ood_subject_scores)), dim = 0)
    super_subject_auc = sklearn.metrics.roc_auc_score(super_subject_trues.detach().cpu().numpy(), super_subject_scores.detach().cpu().numpy())
    print(f'Super Subject AUC: {super_subject_auc}')
    super_object_scores = torch.cat((super_id_object_scores, super_ood_object_scores), dim = 0)
    super_object_trues = torch.cat((torch.zeros_like(super_id_object_scores), torch.ones_like(super_ood_object_scores)), dim = 0)
    super_object_auc = sklearn.metrics.roc_auc_score(super_object_trues.detach().cpu().numpy(), super_object_scores.detach().cpu().numpy())
    print(f'Super Object AUC: {super_object_auc}')
    super_verb_scores = torch.cat((super_id_verb_scores, super_ood_verb_scores), dim = 0)
    super_verb_trues = torch.cat((torch.zeros_like(super_id_verb_scores), torch.ones_like(super_ood_verb_scores)), dim = 0)
    super_verb_auc = sklearn.metrics.roc_auc_score(super_verb_trues.detach().cpu().numpy(), super_verb_scores.detach().cpu().numpy())
    print(f'Super Verb AUC: {super_verb_auc}')

    # Construct score contexts using the super id scores and the adjusted
    # ood scores
    subject_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = super_id_subject_scores,
        novel_scores = adjusted_ood_subject_scores
    )
    object_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = super_id_object_scores,
        novel_scores = adjusted_ood_object_scores
    )
    verb_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = super_id_verb_scores,
        novel_scores = adjusted_ood_verb_scores
    )

if __name__ == '__main__':
    main()
