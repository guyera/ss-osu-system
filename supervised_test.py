import torch
import numpy as np
import random
import adaptation
import noveltydetection
import noveltydetectionfeatures
import unsupervisednoveltydetection


def main():
    torch.random.manual_seed(42)
    random.seed(42)

    detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 6, 9, 8)
    detector = detector.to('cuda:1')
    subject_score_context = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
    object_score_context = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
    verb_score_context = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
    
    state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    detector.load_state_dict(state_dict['module'])
    subject_score_context.load_state_dict(state_dict['subject_score_context'])
    object_score_context.load_state_dict(state_dict['object_score_context'])
    verb_score_context.load_state_dict(state_dict['verb_score_context'])

    dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
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

    spat_f = dataset.__dict__['spatial_features']
    s_f = dataset.__dict__['subject_appearance_features']
    v_f = dataset.__dict__['verb_appearance_features']
    o_f = dataset.__dict__['object_appearance_features']

    mask = [i for i in range(len(spat_f)) if (s_f[i] != None and v_f[i] != None and o_f[i] != None)]

    dataset.__dict__['spatial_features'] = [dataset.__dict__['spatial_features'][i] for i in mask]  
    dataset.__dict__['subject_labels'] = [dataset.__dict__['subject_labels'][i] for i in mask] 
    dataset.__dict__['object_labels'] = [dataset.__dict__['object_labels'][i] for i in mask] 
    dataset.__dict__['verb_labels'] = [dataset.__dict__['verb_labels'][i] for i in mask] 
    dataset.__dict__['subject_appearance_features'] = [dataset.__dict__['subject_appearance_features'][i] for i in mask] 
    dataset.__dict__['object_appearance_features'] = [dataset.__dict__['object_appearance_features'][i] for i in mask] 
    dataset.__dict__['verb_appearance_features'] = [dataset.__dict__['verb_appearance_features'][i] for i in mask] 

    A_s_size = 2
    A_v_size = 3
    A_o_size = 3

    num_s_classes = 5
    num_v_classes = 7
    num_o_classes = 8

    num_random_splits = 30

    mean_aucs = []

    for i in range(num_random_splits):
        A_s = random.sample(range(1, num_s_classes), A_s_size)
        A_v = random.sample(range(1, num_v_classes), A_v_size)
        A_o = random.sample(range(1, num_o_classes), A_o_size)

        s_class_labels = list(dataset.__dict__['subject_labels'])
        v_class_labels = list(dataset.__dict__['verb_labels'])
        o_class_labels = list(dataset.__dict__['object_labels'])

        # 0 indicates anomaly status
        S_y = torch.tensor([0 if s_class_labels[i] in A_s else 1 for i in range(len(s_class_labels))])
        S_y = torch.atleast_2d(S_y).T

        V_y = torch.tensor([0 if v_class_labels[i] in A_v else 1 for i in range(len(v_class_labels))])
        V_y = torch.atleast_2d(V_y).T

        O_y = torch.tensor([0 if o_class_labels[i] in A_o else 1 for i in range(len(o_class_labels))])
        O_y = torch.atleast_2d(O_y).T
        
        # Extracting supervised subject features and labels
        
        # If NoneType, remove element from features and labels
        S_features = dataset.__dict__['subject_appearance_features']
        
        S_features = torch.stack(S_features)
        S_features = torch.squeeze(S_features)

        S_X = torch.flatten(S_features, start_dim=1, end_dim=3)   

        V_features = dataset.__dict__['verb_appearance_features']
        V_features = torch.stack(V_features)
        V_features = torch.squeeze(V_features)

        V_X = torch.flatten(V_features, start_dim=1, end_dim=3)   

        O_features = dataset.__dict__['object_appearance_features']
        O_features = torch.stack(O_features)
        O_features = torch.squeeze(O_features)

        O_X = torch.flatten(O_features, start_dim=1, end_dim=3)   


        # TODO: 
        #    -> Extract anomaly scores for each
        #    
        #    -> Test out supervised methods

        spatial_features = []
        subject_appearance_features = []
        object_appearance_features = []
        verb_appearance_features = []
             
        for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, _, _, _ in dataset:
                
            # For testing case 1:
            spatial_features.append(example_spatial_features)
            subject_appearance_features.append(example_subject_appearance_features)
            object_appearance_features.append(example_object_appearance_features)
            verb_appearance_features.append(example_verb_appearance_features)
                    
            # For testing case 2:
            ##if i == 1:
            ##    spatial_features.append(example_spatial_features)
            ##    subject_appearance_features.append(example_subject_appearance_features)
            ##    object_appearance_features.append(None)
            ##    verb_appearance_features.append(example_verb_appearance_features)

            # For testing case 3:
            ##if i == 2:
            ##    spatial_features.append(None)
            ##    subject_appearance_features.append(None)
            ##    object_appearance_features.append(example_object_appearance_features)
            ##    verb_appearance_features.append(None)
 
        results = detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.full((5,), 0.2, device = 'cuda:1')) 

        S_results = results['subject_novelty_score']
        V_results = results['verb_novelty_score']
        O_results = results['object_novelty_score']
            
        S_a = torch.tensor([float(i) for i in S_results])
        V_a = torch.tensor([float(i) for i in V_results])
        S_a = torch.reshape(S_a, (len(S_a),1)) 
        V_a = torch.reshape(V_a, (len(V_a),1)) 
        S_shuffle_idxs = random.sample(range(len(S_a)), len(S_a))
        V_shuffle_idxs = random.sample(range(len(V_a)), len(V_a))
        S_X = S_X[S_shuffle_idxs]
        S_y = S_y[S_shuffle_idxs]
        S_a = S_a[S_shuffle_idxs] 

        V_X = V_X[V_shuffle_idxs]
        V_y = V_y[V_shuffle_idxs]
        V_a = V_a[V_shuffle_idxs] 
            
        O_a = torch.tensor([float(i) for i in O_results])
        O_a = torch.reshape(O_a, (len(O_a),1))
        O_shuffle_idxs = random.sample(range(len(O_a)), len(O_a))
        O_X = O_X[O_shuffle_idxs]
        O_y = O_y[O_shuffle_idxs]
        O_a = O_a[O_shuffle_idxs] 
         
        mean_AUC, novelty_scores, models = adaptation.supervised_anomaly_detectors.train_supervised_models(S_X,V_X,O_X,S_a,V_a,O_a,S_y,V_y,O_y)
        mean_aucs.append(mean_AUC)
        print("For anomaly split {}: mean_AUC is {}".format(i, mean_AUC))

        with open('supervised_test_out.txt', 'a') as f:
            f.write("For anomaly split {}: mean_AUC is {}\n".format(i, mean_AUC))

    for i in range(3):
        accum = 0
        for j in range(len(mean_aucs)):
            accum += mean_aucs[j][i]
        means.append(accum / len(mean_aucs))

    means = (means[0], means[1], means[2])
    with open('supervised_test_out.txt', 'a') as f:
        f.write("Averaged over all random splits, AUC is {}\n\n".format(means))


if __name__ == '__main__':
    main()
