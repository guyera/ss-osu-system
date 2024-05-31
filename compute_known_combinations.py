# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import argparse
import pickle

from torchvision.models import resnet50

from boximagedataset import BoxImageDataset

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
        default = './'
    )
    parser.add_argument(
        '--csv-path',
        type = str,
        default = 'dataset_v4/dataset_v4_2_train.csv'
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
        default = 64
    )
    parser.add_argument(
        '--batch-size',
        type = int,
        default = 128
    )

    # Persistence arguments
    parser.add_argument(
        '--known-combinations-save-file',
        type = str,
        default='known_combinations.pth'
    )
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    # Construct dataset. If the features haven't already been computed and
    # stored in disk, this will do that as well (so it might take awhile),
    # loading image_batch_size images at a time and computing features from
    # them.
    training_set = BoxImageDataset(
        name = args.dataset_name,
        data_root = args.data_root,
        csv_path = args.csv_path,
        training = True,
        image_batch_size = args.image_batch_size,
        feature_extraction_device = args.device
    )
    
    known_svo = set()
    known_sv = set()
    known_so = set()
    known_vo = set()

    # import ipdb; ipdb.set_trace

    for _, subject_label, object_label, verb_label, _, _, _, _ in training_set:
        if subject_label is not None and int(subject_label.item()) > 0:
            known_subject = True
        else:
            known_subject = False
        
        if verb_label is not None and int(verb_label.item()) > 0:
            known_verb = True
        else:
            known_verb = False

        if object_label is not None and int(object_label.item()) > 0:
            known_object = True
        else:
            known_object = False
        
        # Shift labels back 1, since 0 is for anomaly
        if known_subject and known_verb and known_object:
            joint_label = (int(subject_label.item()) - 1, int(verb_label.item()) - 1, int(object_label.item()) - 1)
            known_svo.add(joint_label)
            known_sv.add((joint_label[0], joint_label[1]))
            known_so.add((joint_label[0], joint_label[2]))
            known_vo.add((joint_label[1], joint_label[2]))
        elif known_subject and known_verb:
            joint_label = (int(subject_label.item()) - 1, int(verb_label.item()) - 1)
            known_sv.add(joint_label)
        elif known_subject and known_object:
            joint_label = (int(subject_label.item()) - 1, int(object_label.item()) - 1)
            known_so.add(joint_label)
        elif known_verb and known_object:
            joint_label = (int(verb_label.item() - 1), int(object_label.item()) - 1)
            known_vo.add(joint_label)

    joint_labels = {
        'svo': known_svo,
        'sv': known_sv,
        'so': known_so,
        'vo': known_vo
    }

    with open(args.known_combinations_save_file, 'wb') as f:
        pickle.dump(joint_labels, f)

if __name__ == '__main__':
    main()
