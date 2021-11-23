import argparse
import pickle

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
        default = 'Custom/annotations/dataset_v3_train.csv'
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
        required = True
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
    
    known_svo = set()
    known_sv = set()
    known_so = set()
    known_vo = set()

    for _, _, _, subject_label, object_label, verb_label in training_set:
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
