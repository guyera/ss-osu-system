import torch
import unsupervisednoveltydetection
import noveltydetectionfeatures
import sklearn.metrics
import noveltydetection
import matplotlib.pyplot as plt

device = 'cuda:0'
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 5, 12, 8)
detector = detector.to(device)

state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
detector.load_state_dict(state_dict['module'])

testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = 'Custom',
    csv_path = 'Custom/annotations/dataset_v3_val.csv',
    training = False,
    image_batch_size = 16,
    feature_extraction_device = device
)

spatial_features = []
subject_appearance_features = []
object_appearance_features = []
verb_appearance_features = []
subject_labels = []
object_labels = []
verb_labels = []
        
for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label in testing_set:
    spatial_features.append(example_spatial_features)
    subject_appearance_features.append(example_subject_appearance_features)
    object_appearance_features.append(example_object_appearance_features)
    verb_appearance_features.append(example_verb_appearance_features)
    subject_labels.append(subject_label)
    object_labels.append(object_label)
    verb_labels.append(verb_label)

results = detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = device))

subject_scores = results['subject_novelty_score']
object_scores = results['object_novelty_score']
verb_scores = results['verb_novelty_score']
p_known_svo = results['p_known_svo']
p_known_sv = results['p_known_sv']
p_known_so = results['p_known_so']
p_known_vo = results['p_known_vo']

filtered_subject_scores = []
filtered_subject_labels = []
for idx, subject_label in enumerate(subject_labels):
    if subject_label is None:
        continue
    filtered_subject_labels.append(subject_label)
    filtered_subject_scores.append(subject_scores[idx])
    
filtered_subject_scores = torch.stack(filtered_subject_scores, dim = 0)
filtered_subject_labels = torch.stack(filtered_subject_labels, dim = 0)

filtered_object_scores = []
filtered_object_labels = []
for idx, object_label in enumerate(object_labels):
    if object_label is None:
        continue
    filtered_object_labels.append(object_label)
    filtered_object_scores.append(object_scores[idx])
filtered_object_scores = torch.stack(filtered_object_scores, dim = 0)
filtered_object_labels = torch.stack(filtered_object_labels, dim = 0)

filtered_verb_scores = []
filtered_verb_labels = []
for idx, verb_label in enumerate(verb_labels):
    if verb_label is None:
        continue
    filtered_verb_labels.append(verb_label)
    filtered_verb_scores.append(verb_scores[idx])
filtered_verb_scores = torch.stack(filtered_verb_scores, dim = 0)
filtered_verb_labels = torch.stack(filtered_verb_labels, dim = 0)

novel_subject_mask = filtered_subject_labels == 0
novel_subject_scores = filtered_subject_scores[novel_subject_mask]
nominal_subject_scores = filtered_subject_scores[~novel_subject_mask]
auc_subject_scores = torch.cat((nominal_subject_scores, novel_subject_scores), dim = 0)
auc_subject_trues = torch.cat((torch.zeros_like(nominal_subject_scores), torch.ones_like(novel_subject_scores)), dim = 0)
subject_auc = sklearn.metrics.roc_auc_score(auc_subject_trues.detach().cpu().numpy(), auc_subject_scores.detach().cpu().numpy())
print(f'Subject AUC: {subject_auc}')

novel_object_mask = filtered_object_labels == 0
novel_object_scores = filtered_object_scores[novel_object_mask]
nominal_object_scores = filtered_object_scores[~novel_object_mask]
auc_object_scores = torch.cat((nominal_object_scores, novel_object_scores), dim = 0)
auc_object_trues = torch.cat((torch.zeros_like(nominal_object_scores), torch.ones_like(novel_object_scores)), dim = 0)
object_auc = sklearn.metrics.roc_auc_score(auc_object_trues.detach().cpu().numpy(), auc_object_scores.detach().cpu().numpy())
print(f'Object AUC: {object_auc}')

novel_verb_mask = filtered_verb_labels == 0
novel_verb_scores = filtered_verb_scores[novel_verb_mask]
nominal_verb_scores = filtered_verb_scores[~novel_verb_mask]
auc_verb_scores = torch.cat((nominal_verb_scores, novel_verb_scores), dim = 0)
auc_verb_trues = torch.cat((torch.zeros_like(nominal_verb_scores), torch.ones_like(novel_verb_scores)), dim = 0)
verb_auc = sklearn.metrics.roc_auc_score(auc_verb_trues.detach().cpu().numpy(), auc_verb_scores.detach().cpu().numpy())
print(f'Verb AUC: {verb_auc}')

subject_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
subject_score_ctx.load_state_dict(state_dict['subject_score_context'])

object_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
object_score_ctx.load_state_dict(state_dict['object_score_context'])

verb_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
verb_score_ctx.load_state_dict(state_dict['verb_score_context'])

p_type, p_n = noveltydetection.utils.compute_probability_novelty(subject_scores, verb_scores, object_scores, p_known_svo, p_known_sv, p_known_so, p_known_vo, subject_score_ctx, verb_score_ctx, object_score_ctx)

type_1_p_n = []
type_3_p_n = []
type_2_p_n = []
nominal_p_n = []
for idx in range(len(p_n)):
    if subject_labels[idx] == 0 and object_labels[idx] != 0 and verb_labels[idx] != 0:
        type_1_p_n.append(p_n[idx])
    elif object_labels[idx] == 0 and subject_labels[idx] != 0 and (verb_labels[idx] != 0 or subject_labels[idx] == -1):
        type_3_p_n.append(p_n[idx])
    elif subject_labels[idx] is not None and verb_labels[idx] == 0 and subject_labels[idx] != 0 and object_labels[idx] != 0:
        type_2_p_n.append(p_n[idx])
    elif subject_labels[idx] != 0 and object_labels[idx] != 0 and (verb_labels[idx] != 0 or subject_labels[idx] == -1):
        nominal_p_n.append(p_n[idx])
type_1_p_n = torch.stack(type_1_p_n, dim = 0)
type_3_p_n = torch.stack(type_3_p_n, dim = 0)
type_2_p_n = torch.stack(type_2_p_n, dim = 0)
nominal_p_n = torch.stack(nominal_p_n, dim = 0)

nominal_x = torch.ones_like(nominal_p_n) * 0
type_1_x = torch.ones_like(type_1_p_n) * 1
type_2_x = torch.ones_like(type_2_p_n) * 2
type_3_x = torch.ones_like(type_3_p_n) * 3

print(f'Average P(N_i) over nominals: {nominal_p_n.mean()}')
print(f'Average P(N_i) over type 1 novelty: {type_1_p_n.mean()}')
print(f'Average P(N_i) over type 2 novelty: {type_2_p_n.mean()}')
print(f'Average P(N_i) over type 3 novelty: {type_3_p_n.mean()}')
print(f'Std P(N_i) over nominals: {nominal_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 1 novelty: {type_1_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 2 novelty: {type_2_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 3 novelty: {type_3_p_n.std(unbiased = True)}')
print(f'Number of nominal examples: {len(nominal_p_n)}')
print(f'Number of type 1 examples: {len(type_1_p_n)}')
print(f'Number of type 2 examples: {len(type_2_p_n)}')
print(f'Number of type 3 examples: {len(type_3_p_n)}')

print()
print(f'p_type: {p_type}')

plt.scatter(nominal_x.detach().cpu().numpy(), nominal_p_n.detach().cpu().numpy(), label = 'No novelty', alpha = 0.2)
plt.scatter(type_1_x.detach().cpu().numpy(), type_1_p_n.detach().cpu().numpy(), label = 'Type 1 novelty', alpha = 0.2)
plt.scatter(type_2_x.detach().cpu().numpy(), type_2_p_n.detach().cpu().numpy(), label = 'Type 2 novelty', alpha = 0.2)
plt.scatter(type_3_x.detach().cpu().numpy(), type_3_p_n.detach().cpu().numpy(), label = 'Type 3 novelty', alpha = 0.2)
plt.ylabel('P(N_i)')
plt.legend()
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
plt.savefig('plt.jpg')
