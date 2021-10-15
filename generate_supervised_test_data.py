import torch

import noveltydetectionfeatures
import unsupervisednoveltydetection

def main():
    detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 6, 9, 8)
    detector = detector.to('cuda:0')
    state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    detector.load_state_dict(state_dict)

    known_set = noveltydetectionfeatures.NoveltyFeatureDataset(
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

    unknown_set = noveltydetectionfeatures.NoveltyFeatureDataset(
        name = 'Custom',
        data_root = 'Custom',
        csv_path = 'Custom/annotations/val_dataset_v1_novel_val.csv',
        num_subj_cls = 6,
        num_obj_cls = 9,
        num_action_cls = 8,
        training = False,
        image_batch_size = 16,
        feature_extraction_device = 'cuda:0'
    )
    
    S_X_kn = torch.flatten(known_set.__dict__['subject_appearance_features'], start_dim=1, end_dim=3)
    S_y_kn = torch.tensor([1 for _ in range(len(S_X_kn))]) 
   
    unkn_features = torch.stack(unknown_set.__dict__['subject_appearance_features'])  
    unkn_features = torch.squeeze(unkn_features)

    S_X_unkn = torch.flatten(unkn_features, start_dim=1, end_dim=3)
    S_y_unkn = torch.tensor([0 for _ in range(len(S_X_unkn))]) 

    S_X = torch.vstack((S_X_kn, S_X_unkn))
    S_y = torch.vstack((torch.atleast_2d(S_y_kn).T, torch.atleast_2d(S_y_unkn).T))

    S_shuffle_idx = torch.randint(0, len(S_y), (len(S_y),))
    
    S_X = S_X[S_shuffle_idx]
    S_y = S_y[S_shuffle_idx]

    import pdb; pdb.set_trace()

    # TODO: 
    #    -> Work out kinks (i.e. repeat what you did above on lines 36 to 51)
    #       for V and O.
    #    
    #    -> Extract anomaly scores for each
    #    
    #    -> Test out supervised methods
 
    V_X_kn = torch.flatten(torch.tensor(known_set.__dict__['verb_appearance_features']), start_dim=1, end_dim=3)
    V_y_kn = torch.tensor([1 for _ in range(len(V_X_kn))]) 
    
    V_X_unkn = torch.flatten(torch.tensor(unknown_set.__dict__['verb_appearance_features']), start_dim=1, end_dim=3)
    V_y_unkn = torch.tensor([0 for _ in range(len(V_X_unkn))]) 
    
    V_X_kn = torch.flatten(torch.tensor(known_set.__dict__['object_appearance_features']), start_dim=1, end_dim=3)
    V_y_kn = torch.tensor([1 for _ in range(len(O_X_kn))]) 
    
    V_X_unkn = torch.flatten(torch.tensor(unknown_set.__dict__['object_appearance_features']), start_dim=1, end_dim=3)
    V_y_unkn = torch.tensor([0 for _ in range(len(O_X_unkn))]) 
   
 
    import pdb; pdb.set_trace()

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
