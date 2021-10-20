import torch
import adaptation
import noveltydetection
import noveltydetectionfeatures
import unsupervisednoveltydetection


def main():
    detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 6, 9, 8)
    detector = detector.to('cuda:0')
    subject_score_context = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
    object_score_context = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
    verb_score_context = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
    
    state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    detector.load_state_dict(state_dict['module'])
    subject_score_context.load_state_dict(state_dict['subject_score_context'])
    object_score_context.load_state_dict(state_dict['object_score_context'])
    verb_score_context.load_state_dict(state_dict['verb_score_context'])

    #detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 6, 9, 8)
    #detector = detector.to('cuda:0')
    #state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    #detector.load_state_dict(state_dict)
    
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
    
    # Extracting supervised subject features and labels
    S_X_kn = torch.flatten(known_set.__dict__['subject_appearance_features'], start_dim=1, end_dim=3)
    S_y_kn = torch.tensor([1 for _ in range(len(S_X_kn))]) 
   
    S_unkn_features = torch.stack(unknown_set.__dict__['subject_appearance_features'])  
    S_unkn_features = torch.squeeze(S_unkn_features)

    S_X_unkn = torch.flatten(S_unkn_features, start_dim=1, end_dim=3)
    S_y_unkn = torch.tensor([0 for _ in range(len(S_X_unkn))]) 

    S_X = torch.vstack((S_X_kn, S_X_unkn))
    S_y = torch.vstack((torch.atleast_2d(S_y_kn).T, torch.atleast_2d(S_y_unkn).T))

    S_shuffle_idx = torch.randint(0, len(S_y), (len(S_y),))
    
    S_X = S_X[S_shuffle_idx]
    S_y = S_y[S_shuffle_idx]

   
    # Extracting supervised verb features and labels 
    V_X_kn = torch.flatten(torch.tensor(known_set.__dict__['verb_appearance_features']), start_dim=1, end_dim=3)
    V_y_kn = torch.tensor([1 for _ in range(len(V_X_kn))]) 
    
    V_unkn_features = torch.stack(unknown_set.__dict__['verb_appearance_features'])
    V_unkn_features = torch.squeeze(V_unkn_features)

    V_X_unkn = torch.flatten(V_unkn_features, start_dim=1, end_dim=3)
    V_y_unkn = torch.tensor([0 for _ in range(len(V_X_unkn))])

    V_X = torch.vstack((V_X_kn, V_X_unkn))
    V_y = torch.vstack((torch.atleast_2d(V_y_kn).T, torch.atleast_2d(V_y_unkn).T)) 
    
    V_shuffle_idx = torch.randint(0, len(V_y), (len(V_y),))

    V_X = V_X[V_shuffle_idx]
    V_y = V_y[V_shuffle_idx]


    # Extracting supervised object features and labels
    O_X_kn = torch.flatten(torch.tensor(known_set.__dict__['object_appearance_features']), start_dim=1, end_dim=3)
    O_y_kn = torch.tensor([1 for _ in range(len(O_X_kn))]) 
    
    O_unkn_features = torch.stack(unknown_set.__dict__['object_appearance_features'])
    O_unkn_features = torch.squeeze(O_unkn_features)

    O_X_unkn = torch.flatten(O_unkn_features, start_dim=1, end_dim=3)
    O_y_unkn = torch.tensor([0 for _ in range(len(O_X_unkn))])

    O_X = torch.vstack((O_X_kn, O_X_unkn))
    O_y = torch.vstack((torch.atleast_2d(O_y_kn).T, torch.atleast_2d(O_y_unkn).T)) 
    
    O_shuffle_idx = torch.randint(0, len(O_y), (len(O_y),))

    O_X = O_X[O_shuffle_idx]
    O_y = O_y[O_shuffle_idx]


    # TODO: 
    #    -> Extract anomaly scores for each
    #    
    #    -> Test out supervised methods
 
    spatial_features = []
    subject_appearance_features = []
    object_appearance_features = []
    verb_appearance_features = []
     
    unkn_spatial_features = []
    unkn_subject_appearance_features = []
    unkn_object_appearance_features = []
    unkn_verb_appearance_features = []

    # Extract anomaly scores for known examples
    for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in known_set:
        
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

    S_results = results['subject_novelty_score']
    V_results = results['verb_novelty_score']
    O_results = results['object_novelty_score']
   
    S_kn_scores = [float(i) for i in S_results]
    V_kn_scores = [float(i) for i in V_results]
    O_kn_scores = [float(i) for i in O_results]

    
    # Extract anomaly scores for unknown examples
    for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in unknown_set:
        
        # For testing case 1:
        unkn_spatial_features.append(example_spatial_features)
        unkn_subject_appearance_features.append(example_subject_appearance_features)
        unkn_object_appearance_features.append(example_object_appearance_features)
        unkn_verb_appearance_features.append(example_verb_appearance_features)
        
        # For testing case 2:
        # unkn_spatial_features.append(example_spatial_features)
        # unkn_subject_appearance_features.append(example_subject_appearance_features)
        # unkn_object_appearance_features.append(None)
        # unkn_verb_appearance_features.append(example_verb_appearance_features)

        # For testing case 3:
        # unkn_spatial_features.append(None)
        # unkn_subject_appearance_features.append(None)
        # unkn_object_appearance_features.append(example_object_appearance_features)
        # unkn_verb_appearance_features.append(None)

    results = detector(unkn_spatial_features, unkn_subject_appearance_features, unkn_verb_appearance_features, unkn_object_appearance_features, torch.full((5,), 0.2, device = 'cuda:0')) 

    S_results = results['subject_novelty_score']
    V_results = results['verb_novelty_score']
    O_results = results['object_novelty_score']
   
    S_unkn_scores = [float(i) for i in S_results]
    V_unkn_scores = [float(i) for i in V_results]
    O_unkn_scores = [float(i) for i in O_results]
   
    S_scores = torch.tensor(S_kn_scores + S_unkn_scores)
    V_scores = torch.tensor(V_kn_scores + V_unkn_scores)
    O_scores = torch.tensor(O_kn_scores + O_unkn_scores)

    S_a = S_scores[S_shuffle_idx]
    V_a = V_scores[V_shuffle_idx]
    O_a = O_scores[O_shuffle_idx]

    S_a = torch.reshape(S_a, (len(S_a),1)) 
    V_a = torch.reshape(V_a, (len(V_a),1)) 
    O_a = torch.reshape(O_a, (len(O_a),1))
   
    
 
    mean_AUC, models = adaptation.supervised_anomaly_detectors.train_supervised_models(S_X,V_X,O_X,S_a,V_a,O_a,S_y,V_y,O_y)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    main()
