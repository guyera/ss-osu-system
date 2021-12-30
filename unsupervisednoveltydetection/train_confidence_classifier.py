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
        default = 'Custom/annotations/dataset_v4_train.csv'
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
        '--classifier-save-file',
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
    # Construct dataset. If the features haven't already been computed and
    # stored in disk, this will do that as well (so it might take awhile),
    # loading image_batch_size images at a time and computing features from
    # them.
    training_set = unsupervisednoveltydetection.common.ReshapedNoveltyFeatureDataset(
        noveltydetectionfeatures.NoveltyFeatureDataset(
            name = args.dataset_name,
            data_root = args.data_root,
            csv_path = args.csv_path,
            training = True,
            image_batch_size = args.image_batch_size,
            feature_extraction_device = args.device
        )
    )

    subject_set = unsupervisednoveltydetection.common.SubjectDataset(training_set, train = True)
    object_set = unsupervisednoveltydetection.common.ObjectDataset(training_set, train = True)
    verb_set = unsupervisednoveltydetection.common.VerbDataset(training_set, train = True)
    
    # Remove novel 0 labels (yes, there are novel labels in the v3 train.csv
    # file. Don't ask my why.)
    id_subject_labels = list(range(1, args.num_subj_cls))
    id_object_labels = list(range(1, args.num_obj_cls))
    id_verb_labels = list(range(1, args.num_action_cls))
    id_subject_indices = unsupervisednoveltydetection.common.get_indices_of_labels(subject_set, id_subject_labels)
    id_object_indices = unsupervisednoveltydetection.common.get_indices_of_labels(object_set, id_object_labels)
    id_verb_indices = unsupervisednoveltydetection.common.get_indices_of_labels(verb_set, id_verb_labels)
    subject_set = torch.utils.data.Subset(subject_set, id_subject_indices)
    object_set = torch.utils.data.Subset(object_set, id_object_indices)
    verb_set = torch.utils.data.Subset(verb_set, id_verb_indices)
    
    # Split the training sets into a training and calibration sets (discarding
    # the latter; it's only used in train_confidence_calibrator.py).
    subject_set, _ = unsupervisednoveltydetection.common.bipartition_dataset(subject_set, 0.7)
    object_set, _ = unsupervisednoveltydetection.common.bipartition_dataset(object_set, 0.7)
    verb_set, _ = unsupervisednoveltydetection.common.bipartition_dataset(verb_set, 0.7)
    
    # Construct training loaders
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
    
    classifier.fit(
        args.lr,
        args.weight_decay,
        args.epochs,
        subject_loader,
        object_loader,
        verb_loader
    )

    torch.save(
        classifier.state_dict(),
        args.classifier_save_file
    )

if __name__ == '__main__':
    main()
