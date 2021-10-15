import torch
import argparse
import pickle
import os

import noveltydetectionfeatures
import noveltydetection
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
        '--csv-path',
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
        '--classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--score-context-save-file',
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
    # Once we receive more data, this is where we'll construct held-out sets,
    # training the classifiers on some sets and computing novelty score
    # distributions from other sets which include examples of novelty. For now,
    # load the trained classifiers and evaluate just the novelty score
    # distributions over nominal data pulled from the validation set.
    testing_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
        noveltydetectionfeatures.NoveltyFeatureDataset(
            name = args.dataset_name,
            data_root = args.data_root,
            csv_path = args.csv_path,
            num_subj_cls = args.num_subj_cls,
            num_obj_cls = args.num_obj_cls,
            num_action_cls = args.num_action_cls,
            training = False,
            image_batch_size = args.image_batch_size,
            feature_extraction_device = args.device
        )
    )

    subject_set = unsupervisednoveltydetection.common.SubjectDataset(testing_set, train = True)
    object_set = unsupervisednoveltydetection.common.ObjectDataset(testing_set, train = True)
    verb_set = unsupervisednoveltydetection.common.VerbDataset(testing_set, train = True)
    
    # Construct testing loaders
    subject_loader = torch.utils.data.DataLoader(
        dataset = subject_set,
        batch_size = args.batch_size,
        shuffle = True
    )

    object_loader = torch.utils.data.DataLoader(
        dataset = object_set,
        batch_size = args.batch_size,
        shuffle = True
    )

    verb_loader = torch.utils.data.DataLoader(
        dataset = verb_set,
        batch_size = args.batch_size,
        shuffle = True
    )
    
    # Create classifier
    classifier = unsupervisednoveltydetection.common.Classifier(
        12544,
        12616,
        args.hidden_nodes,
        args.num_subj_cls,
        args.num_obj_cls,
        args.num_action_cls
    ).to(args.device)
    
    # Load classifier parameters and relocate to given device
    state_dict = torch.load(args.classifier_load_file)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(args.device)
    
    # Compute scores
    subject_scores = []
    object_scores = []
    verb_scores = []
    for features, _ in subject_loader:
        features = features.to(args.device)
        subject_scores.append(classifier.score_subject(features))
    for features, _ in object_loader:
        features = features.to(args.device)
        object_scores.append(classifier.score_object(features))
    for features, _ in verb_loader:
        features = features.to(args.device)
        verb_scores.append(classifier.score_verb(features))
    subject_scores = torch.cat(subject_scores, dim = 0)
    object_scores = torch.cat(object_scores, dim = 0)
    verb_scores = torch.cat(verb_scores, dim = 0)
    
    # Construct score contexts
    subject_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = subject_scores
    )
    object_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = object_scores
    )
    verb_score_context = noveltydetection.utils.ScoreContext(
        noveltydetection.utils.ScoreContext.Source.UNSUPERVISED,
        nominal_scores = verb_scores
    )
    
    # Save the score contexts
    state = {}
    state['subject_score_context'] = subject_score_context.state_dict()
    state['object_score_context'] = object_score_context.state_dict()
    state['verb_score_context'] = verb_score_context.state_dict()
    torch.save(
        state,
        args.score_context_save_file
    )

if __name__ == '__main__':
    main()
