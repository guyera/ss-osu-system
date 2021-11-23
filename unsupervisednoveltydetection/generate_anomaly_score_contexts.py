import torch
import argparse
import pickle
import os

import noveltydetectionfeatures
import noveltydetection
import unsupervisednoveltydetection.common

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

    # Model persistence parameters
    parser.add_argument(
        '--classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--score-contexts-save-file',
        type = str,
        required = True
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

    # Hard-code held-out novel tuples
    
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

    # The s/v/o_testing_sets will be used for computing ID scores from the
    # full pre-trained box classifiers. However, the novel 0 labels must be
    # removed. TODO Check if novel combinations need to be removed as well
    pretrained_id_subject_labels = list(range(1, args.num_subj_cls))
    pretrained_id_object_labels = list(range(1, args.num_obj_cls))
    pretrained_id_verb_labels = list(range(1, args.num_action_cls))
    pretrained_id_subject_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_testing_set, pretrained_id_subject_labels)
    pretrained_id_object_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_testing_set, pretrained_id_object_labels)
    pretrained_id_verb_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_testing_set, pretrained_id_verb_labels)
    pretrained_id_subject_testing_set = torch.utils.data.Subset(subject_testing_set, pretrained_id_subject_indices)
    pretrained_id_object_testing_set = torch.utils.data.Subset(object_testing_set, pretrained_id_object_indices)
    pretrained_id_verb_testing_set = torch.utils.data.Subset(verb_testing_set, pretrained_id_verb_indices)

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
    pretrained_id_subject_testing_loader = torch.utils.data.DataLoader(
        dataset = pretrained_id_subject_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    pretrained_id_object_testing_loader = torch.utils.data.DataLoader(
        dataset = pretrained_id_object_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    pretrained_id_verb_testing_loader = torch.utils.data.DataLoader(
        dataset = pretrained_id_verb_testing_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    
    # Create classifier
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

    # Compute ID scores
    id_subject_scores = []
    id_object_scores = []
    id_verb_scores = []
    for features, _ in id_subject_testing_loader:
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
    
    # Compute OOD scores
    ood_subject_scores = []
    ood_object_scores = []
    ood_verb_scores = []
    for features, _ in ood_subject_loader:
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
    
    # Compute score distribution statistics
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

    # Compute shift of ood distribution mean from id distribution mean
    # normalized by id distribution standard deviation
    subject_score_ood_shift = (ood_subject_score_mean - id_subject_score_mean) / id_subject_score_std
    object_score_ood_shift = (ood_object_score_mean - id_object_score_mean) / id_object_score_std
    verb_score_ood_shift = (ood_verb_score_mean - id_verb_score_mean) / id_verb_score_std

    # Compute scale of ood distribution standard deviation from id distribution
    # standard deviation
    subject_score_ood_scale = ood_subject_score_std / id_subject_score_std
    object_score_ood_scale = ood_object_score_std / id_object_score_std
    verb_score_ood_scale = ood_verb_score_std / id_verb_score_std

    # Load classifier pre-trained on full training data
    pretrained_classifier = unsupervisednoveltydetection.common.Classifier(
        12544,
        12616,
        args.hidden_nodes,
        args.num_subj_cls,
        args.num_obj_cls,
        args.num_action_cls
    ).to(args.device)
    
    # Load classifier parameters and relocate to given device
    state_dict = torch.load(args.classifier_load_file)
    pretrained_classifier.load_state_dict(state_dict)
    pretrained_classifier = pretrained_classifier.to(args.device)

    # Compute ID scores from pretrained classifier
    pretrained_id_subject_scores = []
    pretrained_id_object_scores = []
    pretrained_id_verb_scores = []
    for features, _ in pretrained_id_subject_testing_loader:
        features = features.to(args.device)
        pretrained_id_subject_scores.append(classifier.score_subject(features))
    for features, _ in pretrained_id_object_testing_loader:
        features = features.to(args.device)
        pretrained_id_object_scores.append(classifier.score_object(features))
    for features, _ in pretrained_id_verb_testing_loader:
        features = features.to(args.device)
        pretrained_id_verb_scores.append(classifier.score_verb(features))
    pretrained_id_subject_scores = torch.cat(pretrained_id_subject_scores, dim = 0)
    pretrained_id_object_scores = torch.cat(pretrained_id_object_scores, dim = 0)
    pretrained_id_verb_scores = torch.cat(pretrained_id_verb_scores, dim = 0)
    
    # Compute pretrained id score distribution statistics
    pretrained_id_subject_score_mean = torch.mean(pretrained_id_subject_scores)
    pretrained_id_object_score_mean = torch.mean(pretrained_id_object_scores)
    pretrained_id_verb_score_mean = torch.mean(pretrained_id_verb_scores)
    pretrained_id_subject_score_std = torch.std(pretrained_id_subject_scores, unbiased = True)
    pretrained_id_object_score_std = torch.std(pretrained_id_object_scores, unbiased = True)
    pretrained_id_verb_score_std = torch.std(pretrained_id_verb_scores, unbiased = True)

    # Shift and scale ID distribution to get supposed OOD score distribution
    # for the pretrained classifier
    pretrained_ood_subject_score_mean = pretrained_id_subject_score_mean + subject_score_ood_shift * pretrained_id_subject_score_std
    pretrained_ood_object_score_mean = pretrained_id_object_score_mean + object_score_ood_shift * pretrained_id_object_score_std
    pretrained_ood_verb_score_mean = pretrained_id_verb_score_mean + verb_score_ood_shift * pretrained_id_verb_score_std
    pretrained_ood_subject_score_std = pretrained_id_subject_score_std * subject_score_ood_scale
    pretrained_ood_object_score_std = pretrained_id_object_score_std * object_score_ood_scale
    pretrained_ood_verb_score_std = pretrained_id_verb_score_std * verb_score_ood_scale
    
    # Shift and rescale OOD scores (from held-out setting) to match the supposed
    # mean and standard deviation of the novel score distribution for the full
    # pretrained classifier.
    centered_ood_subject_scores = (ood_subject_scores - ood_subject_score_mean) / ood_subject_score_std
    adjusted_ood_subject_scores = centered_ood_subject_scores * pretrained_ood_subject_score_std + pretrained_ood_subject_score_mean
    centered_ood_object_scores = (ood_object_scores - ood_object_score_mean) / ood_object_score_std
    adjusted_ood_object_scores = centered_ood_object_scores * pretrained_ood_object_score_std + pretrained_ood_object_score_mean
    centered_ood_verb_scores = (ood_verb_scores - ood_verb_score_mean) / ood_verb_score_std
    adjusted_ood_verb_scores = centered_ood_verb_scores * pretrained_ood_verb_score_std + pretrained_ood_verb_score_mean

    # Construct score contexts using the pretrained id scores and the adjusted
    # ood scores
    subject_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = pretrained_id_subject_scores,
        novel_scores = adjusted_ood_subject_scores
    )
    object_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = pretrained_id_object_scores,
        novel_scores = adjusted_ood_object_scores
    )
    verb_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = pretrained_id_verb_scores,
        novel_scores = adjusted_ood_verb_scores
    )
    
    # Save the score contexts
    state = {}
    state['subject_score_context'] = subject_score_context.state_dict()
    state['object_score_context'] = object_score_context.state_dict()
    state['verb_score_context'] = verb_score_context.state_dict()
    torch.save(
        state,
        args.score_contexts_save_file
    )

if __name__ == '__main__':
    main()
