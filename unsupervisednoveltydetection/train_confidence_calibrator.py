import torch
import argparse
import pickle
import os

import noveltydetectionfeatures
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
        '--csv-path',
        type = str,
        default = 'Custom/annotations/val_dataset_v1_train.csv'
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
    parser.add_argument(
        '--split-dir',
        type = str,
        default = 'splits'
    )
    parser.add_argument(
        '--split-name',
        type = str,
        default = 'split_1'
    )

    # Model persistence parameters
    parser.add_argument(
        '--classifier-load-file',
        type = str,
        required = True
    )
    parser.add_argument(
        '--calibrator-save-file',
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
        default = 0.001
    )
    parser.add_argument(
        '--weight-decay',
        type = float,
        default = 0.0
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
    # Construct dataset. If the features haven't already been computed and
    # stored in disk, this will do that as well (so it might take awhile),
    # loading image_batch_size images at a time and computing features from
    # them.
    training_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
        noveltydetectionfeatures.NoveltyFeatureDataset(
            name = args.dataset_name,
            data_root = args.data_root,
            csv_path = args.csv_path,
            num_subj_cls = args.num_subj_cls,
            num_obj_cls = args.num_obj_cls,
            num_action_cls = args.num_action_cls,
            training = True,
            image_batch_size = args.image_batch_size,
            feature_extraction_device = args.device
        )
    )

    subject_set = unsupervisednoveltydetection.common.SubjectDataset(training_set, train = True)
    object_set = unsupervisednoveltydetection.common.ObjectDataset(training_set, train = True)
    verb_set = unsupervisednoveltydetection.common.VerbDataset(training_set, train = True)
    
    # Split the training sets into a training and calibration sets (discarding
    # the latter; it's only used in train_confidence_calibrator.py).
    _, subject_set = unsupervisednoveltydetection.common.bipartition_dataset(subject_set, 0.7)
    _, object_set = unsupervisednoveltydetection.common.bipartition_dataset(object_set, 0.7)
    _, verb_set = unsupervisednoveltydetection.common.bipartition_dataset(verb_set, 0.7)
    
    # Construct calibration loaders
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
    )
    
    state_dict = torch.load(args.classifier_load_file)
    classifier.load_state_dict(state_dict)
    classifier = classifier.to(args.device)
    
    calibrator = unsupervisednoveltydetection.common.ConfidenceCalibrator().to(args.device)
    calibrator.fit(
        classifier,
        args.lr,
        args.weight_decay,
        args.epochs,
        subject_loader,
        object_loader,
        verb_loader
    )

    torch.save(
        calibrator.state_dict(),
        args.calibrator_save_file
    )

if __name__ == '__main__':
    main()
