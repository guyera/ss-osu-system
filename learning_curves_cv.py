import numpy as np
import random
import adaptation
import noveltydetection
import noveltydetectionfeatures
import unsupervisednoveltydetection
import pickle
import torch
import matplotlib.pyplot as plt
from auc import compute_auc

from sklearn.metrics import roc_auc_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

examples_per_trial = 10

dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
        name = 'Custom',
        data_root = 'Custom',
        csv_path = 'Custom/csvs/dataset_v3_4_val.csv',
        #num_subj_cls = 5,
        #num_obj_cls = 12,
        #num_action_cls = 8,
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

ci_confidence = 0.99
num_splits = 1 #5
meta_partial_subj_aucs = []
meta_partial_verb_aucs = []
meta_partial_obj_aucs  = []
meta_subj_aucs = []
meta_verb_aucs = []
meta_obj_aucs  = []

meta_meta_subj_trues = []
meta_meta_verb_trues = []
meta_meta_obj_trues  = []
meta_meta_subj_scores = []
meta_meta_verb_scores = []
meta_meta_obj_scores  = []
for i in range(1,num_splits+1):
    with open('Custom/us_s_learning_curves/results{}.pkl'.format(i), 'rb') as f:
        unsupervised_results = pickle.load(f)

    classifier_state_dict = unsupervised_results['classifier']

    uns_subject_auc = unsupervised_results['subject_auc']
    uns_object_auc = unsupervised_results['object_auc']
    uns_verb_auc = unsupervised_results['verb_auc']

    uns_partial_subject_auc = unsupervised_results['partial_subject_auc']
    uns_partial_object_auc = unsupervised_results['partial_object_auc']
    uns_partial_verb_auc = unsupervised_results['partial_verb_auc']

    id_subject_labels = unsupervised_results['id_subject_labels']
    id_object_labels = unsupervised_results['id_object_labels']
    id_verb_labels = unsupervised_results['id_verb_labels']

    ood_subject_labels = unsupervised_results['ood_subject_labels']
    ood_object_labels = unsupervised_results['ood_object_labels']
    ood_verb_labels = unsupervised_results['ood_verb_labels']

    classifier = unsupervisednoveltydetection.common.Classifier(
        12544,
        12616,
        1024,
        len(id_subject_labels) + 1, # Add 1 for anomaly label = 0
        len(id_object_labels) + 1, # Add 1 for anomaly label = 0
        len(id_verb_labels) + 1 # Add 1 for anomaly label = 0
    )

    classifier = classifier.to(device)

    classifier.load_state_dict(classifier_state_dict)

    # NOTE: All feature data are lists of torch tensors 
    # currently.

    A_s = ood_subject_labels
    A_v = ood_verb_labels
    A_o = ood_object_labels

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

    # Making X_features into a tensor instead of a list 
    # of tensors

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

    # Continue to run loop over dataset, stitching
    # together pieces as specified by Alex in his
    # email. This should produce anomaly scores on an
    # element by element basis that can then be used
    # to help train the supervised anomaly detectors.

    S_a = []
    V_a = []
    O_a = []

    for example_spatial_features, \
        example_subject_appearance_features, \
        example_object_appearance_features, \
        example_verb_appearance_features, \
        _, _, _ in dataset:

        verb_features = torch.flatten(example_verb_appearance_features)
        spat_features = torch.flatten(example_spatial_features)

        subj_clf_features = torch.flatten(example_subject_appearance_features)
        obj_clf_features  = torch.flatten(example_object_appearance_features)
        verb_clf_features = torch.cat((spat_features, verb_features))

        # Unsqueeze along dimension 0 to generate batch tensors
        # of length 1
        subj_clf_features = torch.unsqueeze(subj_clf_features, 0)
        obj_clf_features  = torch.unsqueeze(obj_clf_features,  0)
        verb_clf_features = torch.unsqueeze(verb_clf_features, 0)

        S_batch_scores = classifier.score_subject(subj_clf_features.to(device))
        V_batch_scores = classifier.score_verb(verb_clf_features.to(device))
        O_batch_scores = classifier.score_object(obj_clf_features.to(device))

        # Assert batch size is 1, an assumption we're making here
        assert(S_batch_scores.shape[0] == 1)
        S_a.append(torch.squeeze(S_batch_scores, 0))

        assert(V_batch_scores.shape[0] == 1)
        V_a.append(torch.squeeze(V_batch_scores, 0))

        assert(O_batch_scores.shape[0] == 1)
        O_a.append(torch.squeeze(O_batch_scores, 0))

    S_a = torch.atleast_2d(torch.stack(S_a)).T
    V_a = torch.atleast_2d(torch.stack(V_a)).T
    O_a = torch.atleast_2d(torch.stack(O_a)).T

    # BREAK HERE FOR BAR CHARTS

    # Need to utilize max anomaly score querying strategy
    # for feedback selection in trials. Could do this by
    # sorting the anomaly scores for each appearance 
    # feature type and imposing the sorted order on 
    # the appearance features and the labels as well.
    #
    # This would effectively require 3 orderings:
    #    i) Subject Feature Ordering
    #   ii) Verb Feature Ordering
    #  iii) Object Feature Ordering

    subj_vals, subj_idxs = torch.sort(S_a, dim=0, descending=True)
    verb_vals, verb_idxs = torch.sort(V_a, dim=0, descending=True)
    obj_vals, obj_idxs = torch.sort(O_a, dim=0, descending=True)

    subj_idxs = torch.squeeze(subj_idxs)
    verb_idxs = torch.squeeze(verb_idxs)
    obj_idxs  = torch.squeeze(obj_idxs)

    # Sort {}_a {}_X and {}_y according to {}_idxs order.
    # This imposes our query selection ordering, so now
    # we can simply run batches through in this order
    # to simulate.
    #
    # NOTE: This approach will provide a lower bound on when the 
    # supervised detector will supercede the unsupervised detector
    # since we are ranking according to anomaly scores over the 
    # WHOLE validation set and then training in batches accordingly.
    S_a = S_a[subj_idxs]
    S_X = S_X[subj_idxs]
    S_y = S_y[subj_idxs]

    V_a = V_a[verb_idxs]
    V_X = V_X[verb_idxs]
    V_y = V_y[verb_idxs]

    O_a = O_a[obj_idxs]
    O_X = O_X[obj_idxs]
    O_y = O_y[obj_idxs]

    # We have floor(len(train dataset) / examples_per_trial) rounds here
    # start with examples_per_trial examples being sent to the super-
    # vised anomaly detectors for training

    # ISSUE: If we rely on the mean_AUC returned during
    # training, this will be uninformative for small batch
    # sizes because each ensemble member will only be trained
    # on 1 or two examples. Potential solution, could ignore 
    # them and hold out half of data for eval during each round.

    # Note: Trials are data-cumulative. i.e. If we are
    # on round 5, we are utilizing feedback data introduced
    # in round 5 as well as the feedback data provided in 
    # rounds 4,3,2,1,0.

    subj_aucs   = []
    subj_partial_aucs = []
    verb_aucs   = []
    verb_partial_aucs = []
    object_aucs = []
    object_partial_aucs = []
    
    meta_subj_trues = []
    meta_verb_trues = []
    meta_obj_trues  = []
    meta_subj_scores = []
    meta_verb_scores = []
    meta_obj_scores  = []
    
    subj_cis   = []
    verb_cis  = []
    obj_cis = []
    
    num_rounds = S_X.shape[0] // examples_per_trial

    ### Let the rounds commence!
    for j in range(num_rounds):
        round_idxs = [i for i in range((j+1)*examples_per_trial)]

        S_examples      = S_X[round_idxs].to(device)
        S_labels              = S_y[round_idxs].to(device)
        S_unsupervised_scores = S_a[round_idxs].to(device)

        V_examples      = V_X[round_idxs].to(device)
        V_labels              = V_y[round_idxs].to(device)
        V_unsupervised_scores = V_a[round_idxs].to(device)

        O_examples      = O_X[round_idxs].to(device)
        O_labels              = O_y[round_idxs].to(device)
        O_unsupervised_scores = O_a[round_idxs].to(device)

        ### BREAK HERE AND CHECK {}_labels
        ## import pdb; pdb.set_trace()
        
        aucs, scores, models = \
            adaptation.supervised_anomaly_detectors.train_supervised_models_nom_anom(
                S_examples, V_examples, O_examples,
                S_unsupervised_scores, V_unsupervised_scores, O_unsupervised_scores,
                S_labels, V_labels, O_labels
            )
        
        S_scores, V_scores, O_scores = scores
        
        S_nom_scores  = S_scores[0]
        S_nom_trues   = torch.ones_like(S_nom_scores)
        S_anom_scores = S_scores[1]
        S_anom_trues  = torch.zeros_like(S_anom_scores)
        subj_trues = torch.vstack((S_nom_trues.cpu(), S_anom_trues.cpu()))
        S_scores = torch.vstack((S_nom_scores.cpu(), S_anom_scores.cpu()))
        
        V_nom_scores  = V_scores[0]
        V_nom_trues   = torch.ones_like(V_nom_scores)
        V_anom_scores = V_scores[1]
        V_anom_trues  = torch.zeros_like(V_anom_scores)
        verb_trues = torch.vstack((V_nom_trues.cpu(), V_anom_trues.cpu()))
        V_scores = torch.vstack((V_nom_scores.cpu(), V_anom_scores.cpu()))
        
        O_nom_scores  = O_scores[0]
        O_nom_trues   = torch.ones_like(O_nom_scores)
        O_anom_scores = O_scores[1]
        O_anom_trues  = torch.zeros_like(O_anom_scores)
        obj_trues = torch.vstack((O_nom_trues.cpu(), O_anom_trues.cpu()))
        O_scores = torch.vstack((O_nom_scores.cpu(), O_anom_scores.cpu())) 
        
        subj_partial_auc, verb_partial_auc, obj_partial_auc = aucs

        #subj_trues  = torch.flatten(torch.stack(S_y)).data.cpu().numpy()
        subj_scores = torch.squeeze(S_scores).detach().numpy()
        subj_auc, subj_cov, subj_ci = \
            compute_auc(torch.squeeze(subj_trues), subj_scores, ci_confidence=ci_confidence)

        #verb_trues  = torch.flatten(torch.stack(V_y_holdout)).data.cpu().numpy()
        verb_scores = torch.squeeze(V_scores).detach().numpy()
        verb_auc, verb_cov, verb_ci = \
            compute_auc(torch.squeeze(verb_trues), verb_scores, ci_confidence=ci_confidence)

        #obj_trues  = torch.flatten(torch.stack(O_y_holdout)).data.cpu().numpy()
        obj_scores = torch.squeeze(O_scores).detach().numpy()
        obj_auc, obj_cov, obj_ci = \
            compute_auc(torch.squeeze(obj_trues), obj_scores, ci_confidence=ci_confidence)
        
        meta_subj_trues.append(subj_trues)
        meta_verb_trues.append(verb_trues)
        meta_obj_trues.append(obj_trues)
        
        meta_subj_scores.append(subj_scores)
        meta_verb_scores.append(verb_scores)
        meta_obj_scores.append(obj_scores)
        
        subj_partial_aucs.append(subj_partial_auc)
        verb_partial_aucs.append(verb_partial_auc)
        object_partial_aucs.append(obj_partial_auc)

        subj_aucs.append(subj_auc)
        verb_aucs.append(verb_auc)
        object_aucs.append(obj_auc)
        
        subj_cis.append(subj_ci)
        verb_cis.append(verb_ci)
        obj_cis.append(obj_ci)

        # Need to compute partial and full AUC for each S,V,O
        print(f'\nRound {j} Complete!')
        print(f'Subject Partial AUC is {subj_partial_auc}')
        print(f'This is {subj_partial_auc-uns_partial_subject_auc} better than unsupervised')
        print(f'Verb Partial AUC is {verb_partial_auc}')
        print(f'This is {verb_partial_auc-uns_partial_verb_auc} better than unsupervised')
        print(f'Object Partial AUC is {obj_partial_auc}')
        print(f'This is {obj_partial_auc-uns_partial_object_auc} better than unsupervised')
        print(f'Subject AUC is {subj_auc}')
        print(f'This is {subj_auc-uns_subject_auc} better than unsupervised')
        print(f'Verb AUC is {verb_auc}')
        print(f'This is {verb_auc-uns_verb_auc} better than unsupervised')
        print(f'Object AUC is {obj_auc}')
        print(f'This is {obj_auc-uns_object_auc} better than unsupervised')

    meta_partial_subj_aucs.append(subj_partial_aucs)
    meta_partial_verb_aucs.append(verb_partial_aucs)
    meta_partial_obj_aucs.append(object_partial_aucs)
    meta_subj_aucs.append(subj_aucs)
    meta_verb_aucs.append(verb_aucs)
    meta_obj_aucs.append(object_aucs)
    
    meta_meta_subj_trues.append(meta_subj_trues)
    meta_meta_verb_trues.append(meta_verb_trues)
    meta_meta_obj_trues.append(meta_obj_trues)
    meta_meta_subj_scores.append(meta_subj_scores)
    meta_meta_verb_scores.append(meta_verb_scores)
    meta_meta_obj_scores.append(meta_obj_scores)
    
    assert(len(meta_meta_subj_trues) != 0)
        
    # First row of plots:  partial AUROCs
    # Second row of plots: complete AUROCS
    #%matplotlib notebook
    fig_ps, ax_ps = plt.subplots()
    fig_pv, ax_pv = plt.subplots()
    fig_po, ax_po = plt.subplots()
    fig_fs, ax_fs = plt.subplots()
    fig_fv, ax_fv = plt.subplots()
    fig_fo, ax_fo = plt.subplots()

    x = [k for k in range(num_rounds)]
    uns_partial_subj = uns_partial_subject_auc * np.ones(len(subj_partial_aucs))
    uns_partial_verb = uns_partial_verb_auc * np.ones(len(verb_partial_aucs))
    uns_partial_obj  = uns_partial_object_auc * np.ones(len(object_partial_aucs))
    uns_subj = uns_subject_auc * np.ones(len(subj_aucs))
    uns_verb = uns_verb_auc * np.ones(len(verb_aucs))
    uns_obj  = uns_object_auc * np.ones(len(object_aucs))

    subj_cl = np.array(subj_cis)[:,0]
    subj_cu = np.array(subj_cis)[:,1]

    verb_cl = np.array(verb_cis)[:,0]
    verb_cu = np.array(verb_cis)[:,1]

    obj_cl = np.array(obj_cis)[:,0]
    obj_cu = np.array(obj_cis)[:,1]

    # Look at AUC confidence here to determine how best to plot it
    # then make appropriate changes in the block below to save the plots

    ax_ps.plot(x, uns_partial_subj, label='unsupervised')
    ax_ps.plot(x, subj_partial_aucs, label='supervised')
    ax_ps.set_xlabel('Rounds')
    ax_ps.set_ylabel('AUROC')
    ax_ps.set_title(f'Subject Partial AUC Split {i}')
    ax_ps.legend()

    fig_ps.savefig(f'learning_curves/cv_ps{i}.png', format='png')

    ax_pv.plot(x, uns_partial_verb, label='unsupervised')
    ax_pv.plot(x, verb_partial_aucs, label='supervised')
    ax_pv.set_xlabel('Rounds')
    ax_pv.set_ylabel('AUROC')
    ax_pv.set_title(f'Verb Partial AUC Split {i}')
    ax_pv.legend()

    fig_pv.savefig(f'learning_curves/cv_pv{i}.png', format='png')

    ax_po.plot(x, uns_partial_obj, label='unsupervised')
    ax_po.plot(x, object_partial_aucs, label='supervised')
    ax_po.set_xlabel('Rounds')
    ax_po.set_ylabel('AUROC')
    ax_po.set_title(f'Object Partial AUC Split {i}')
    ax_po.legend()

    fig_po.savefig(f'learning_curves/cv_po{i}.png', format='png')

    ax_fs.plot(x, uns_subj, label='unsupervised')
    ax_fs.plot(x, subj_aucs, label='supervised')
    ax_fs.plot(x, subj_cl, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence lower bound')
    ax_fs.plot(x, subj_cu, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence upper bound')
    ax_fs.set_xlabel('Rounds')
    ax_fs.set_ylabel('AUROC')
    ax_fs.set_title(f'Subject AUC Split {i}')
    ax_fs.legend()

    fig_fs.savefig(f'learning_curves/cv_fs{i}.png', format='png')

    ax_fv.plot(x, uns_verb, label='unsupervised')
    ax_fv.plot(x, verb_aucs, label='supervised')
    ax_fv.plot(x, verb_cl, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence lower bound')
    ax_fv.plot(x, verb_cu, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence upper bound')
    ax_fv.set_xlabel('Rounds')
    ax_fv.set_ylabel('AUROC')
    ax_fv.set_title(f'Verb AUC Split {i}')
    ax_fv.legend()

    fig_fv.savefig(f'learning_curves/cv_fv{i}.png', format='png')

    ax_fo.plot(x, uns_obj, label='unsupervised')
    ax_fo.plot(x, object_aucs, label='supervised')
    ax_fo.plot(x, obj_cl, linestyle='dashed', label=f'{int(ci_confidence*100)}% confidence lower bound')
    ax_fo.plot(x, obj_cu, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence upper bound')
    ax_fo.set_xlabel('Rounds')
    ax_fo.set_ylabel('AUROC')
    ax_fo.set_title(f'Object AUC Split {i}')
    ax_fo.legend()

    fig_fo.savefig(f'learning_curves/cv_fo{i}.png', format='png')

# Compute Confidence Intervals over full AUCs
s_true_flat = np.array(meta_meta_subj_trues).swapaxes(0,1)
s_true_flat = s_true_flat.reshape(s_true_flat.shape[0],-1)
v_true_flat = np.array(meta_meta_verb_trues).swapaxes(0,1)
v_true_flat = v_true_flat.reshape(v_true_flat.shape[0],-1)
o_true_flat = np.array(meta_meta_obj_trues).swapaxes(0,1)
o_true_flat = o_true_flat.reshape(o_true_flat.shape[0],-1)

s_score_flat = np.array(meta_meta_subj_scores).swapaxes(0,1)
s_score_flat = s_score_flat.reshape(s_score_flat.shape[0],-1)
v_score_flat = np.array(meta_meta_verb_trues).swapaxes(0,1)
v_score_flat = v_score_flat.reshape(v_score_flat.shape[0],-1)
o_score_flat = np.array(meta_meta_obj_scores).swapaxes(0,1)
o_score_flat = o_score_flat.reshape(o_score_flat.shape[0],-1)

agg_s_auc = []
agg_s_ci  = []
agg_v_auc = []
agg_v_ci  = []
agg_o_auc = []
agg_o_ci  = []

agg_sp_auc = []
agg_vp_auc = []
agg_op_auc = []

round_count = len(s_true_flat)

for i in range(round_count):
    sti = np.array(s_true_flat[i])
    ssi = np.array(s_score_flat[i])
    vti = np.array(v_true_flat[i])
    vsi = np.array(v_score_flat[i])
    oti = np.array(o_true_flat[i])
    osi = np.array(o_score_flat[i])
    s_auc, s_cov, s_ci = \
                compute_auc(sti, ssi, ci_confidence=ci_confidence)
    v_auc, v_cov, v_ci = \
                compute_auc(vti, vsi, ci_confidence=ci_confidence)
    o_auc, o_cov, o_ci = \
                compute_auc(oti, osi, ci_confidence=ci_confidence)
    
    sp_auc = roc_auc_score(sti, ssi, max_fpr = 0.25)
    vp_auc = roc_auc_score(vti, vsi, max_fpr = 0.25)
    op_auc = roc_auc_score(oti, osi, max_fpr = 0.25)
    
    agg_s_auc.append(s_auc)
    agg_v_auc.append(v_auc)
    agg_o_auc.append(o_auc)
    agg_s_ci.append(s_ci)
    agg_v_ci.append(v_ci)
    agg_o_ci.append(o_ci)
    
    agg_sp_auc.append(sp_auc)
    agg_vp_auc.append(vp_auc)
    agg_op_auc.append(op_auc)

fig_agg_ps, ax_agg_ps = plt.subplots()
fig_agg_pv, ax_agg_pv = plt.subplots()
fig_agg_po, ax_agg_po = plt.subplots()
fig_agg_fs, ax_agg_fs = plt.subplots()
fig_agg_fv, ax_agg_fv = plt.subplots()
fig_agg_fo, ax_agg_fo = plt.subplots()

x = [k for k in range(round_count)]
uns_partial_subj = uns_partial_subject_auc * np.ones(len(subj_partial_aucs))
uns_partial_verb = uns_partial_verb_auc * np.ones(len(verb_partial_aucs))
uns_partial_obj  = uns_partial_object_auc * np.ones(len(object_partial_aucs))
uns_subj = uns_subject_auc * np.ones(round_count)
uns_verb = uns_verb_auc * np.ones(round_count)
uns_obj  = uns_object_auc * np.ones(round_count)

agg_subj_cl = np.array(agg_s_ci)[:,0]
agg_subj_cu = np.array(agg_s_ci)[:,1]

agg_verb_cl = np.array(agg_v_ci)[:,0]
agg_verb_cu = np.array(agg_v_ci)[:,1]

agg_obj_cl = np.array(agg_o_ci)[:,0]
agg_obj_cu = np.array(agg_o_ci)[:,1]


# Look at AUC confidence here to determine how best to plot it
# then make appropriate changes in the block below to save the plots

ax_agg_ps.plot(x, uns_partial_subj, label='unsupervised')
ax_agg_ps.plot(x, agg_sp_auc, label='supervised')
ax_agg_ps.set_xlabel('Rounds')
ax_agg_ps.set_ylabel('AUROC')
ax_agg_ps.set_title(f'Aggregated Subject Partial AUROC')
ax_agg_ps.legend()

fig_agg_ps.savefig(f'learning_curves/cv_agg_ps.png', format='png')

ax_agg_pv.plot(x, uns_partial_verb, label='unsupervised')
ax_agg_pv.plot(x, agg_vp_auc, label='supervised')
ax_agg_pv.set_xlabel('Rounds')
ax_agg_pv.set_ylabel('AUROC')
ax_agg_pv.set_title(f'Aggregated Verb Partial AUROC')
ax_agg_pv.legend()

fig_agg_pv.savefig(f'learning_curves/cv_agg_pv.png', format='png')

ax_agg_po.plot(x, uns_partial_obj, label='unsupervised')
ax_agg_po.plot(x, agg_op_auc, label='supervised')
ax_agg_po.set_xlabel('Rounds')
ax_agg_po.set_ylabel('AUROC')
ax_agg_po.set_title(f'Aggregated Object Partial AUROC')
ax_agg_po.legend()

fig_agg_po.savefig(f'learning_curves/cv_agg_po.png', format='png')

ax_agg_fs.plot(x, uns_subj, label='unsupervised')
ax_agg_fs.plot(x, agg_s_auc, label='supervised')
ax_agg_fs.plot(x, agg_subj_cl, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence lower bound')
ax_agg_fs.plot(x, agg_subj_cu, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence upper bound')
ax_agg_fs.set_xlabel('Rounds')
ax_agg_fs.set_ylabel('AUROC')
ax_agg_fs.set_title(f'Aggregated Subject AUROC')
ax_agg_fs.legend()

fig_agg_fs.savefig(f'learning_curves/cv_agg_fs.png', format='png')

ax_agg_fv.plot(x, uns_verb, label='unsupervised')
ax_agg_fv.plot(x, agg_v_auc, label='supervised')
ax_agg_fv.plot(x, agg_verb_cl, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence lower bound')
ax_agg_fv.plot(x, agg_verb_cu, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence upper bound')
ax_agg_fv.set_xlabel('Rounds')
ax_agg_fv.set_ylabel('AUROC')
ax_agg_fv.set_title(f'Aggregated Verb AUROC')
ax_agg_fv.legend()

fig_agg_fv.savefig(f'learning_curves/cv_agg_fv.png', format='png')

ax_agg_fo.plot(x, uns_obj, label='unsupervised')
ax_agg_fo.plot(x, agg_o_auc, label='supervised')
ax_agg_fo.plot(x, agg_obj_cl, linestyle='dashed', label=f'{int(ci_confidence*100)}% confidence lower bound')
ax_agg_fo.plot(x, agg_obj_cu, linestyle='dashed',label=f'{int(ci_confidence*100)}% confidence upper bound')
ax_agg_fo.set_xlabel('Rounds')
ax_agg_fo.set_ylabel('AUROC')
ax_agg_fo.set_title(f'Aggregated Object AUROC')
ax_agg_fo.legend()

fig_agg_fo.savefig(f'learning_curves/cv_agg_fo.png', format='png')    

