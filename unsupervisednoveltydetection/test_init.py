import torch

import noveltydetectionfeatures
import unsupervisednoveltydetection

def main():
    detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 6, 9, 8)
    detector = detector.to('cuda:0')
    state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    detector.load_state_dict(state_dict)

    testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
        name = 'Custom',
        data_root = 'Custom',
        csv_path = 'Custom/annotations/val_dataset_v1_val.csv',
        num_subj_cls = 6,
        num_obj_cls = 9,
        num_action_cls = 8,
        training = False,
        image_batch_size = 16,
        feature_extraction_device = 'cuda:0'
    )

    spatial_features = []
    subject_appearance_features = []
    object_appearance_features = []
    verb_appearance_features = []
    
    for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in testing_set:
        
        # For testing case 1:
        spatial_features.append(example_spatial_features)
        subject_appearance_features.append(example_subject_appearance_features)
        object_appearance_features.append(example_object_appearance_features)
        verb_appearance_features.append(example_verb_appearance_features)
        
        # For testing case 2:
        # spatial_features.append(example_spatial_features)
        # subject_appearance_features.append(example_subject_appearance_features)
        # object_appearance_features.append(None)
        # verb_appearance_features.append(example_verb_appearance_features)

        # For testing case 3:
        # spatial_features.append(None)
        # subject_appearance_features.append(None)
        # object_appearance_features.append(example_object_appearance_features)
        # verb_appearance_features.append(None)

    results = detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.full((5,), 0.2, device = 'cuda:0'))
    
    for i in range(10):
        print(results['top3'][i])
        print()

    lowest_confidence_1 = 1.0
    lowest_confidence_3 = 1.0
    for top3 in results['top3']:
        confidence_1 = top3[0][1].item()
        if confidence_1 < lowest_confidence_1:
            lowest_confidence_1 = confidence_1

        confidence_3 = top3[2][1].item()
        if confidence_3 < lowest_confidence_3:
            lowest_confidence_3 = confidence_3

    print(f'Lowest #1 confidence: {lowest_confidence_1}')
    print(f'Lowest #3 confidence: {lowest_confidence_3}')

if __name__ == '__main__':
    main()
