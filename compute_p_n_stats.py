import sys
import argparse

from torchvision.models import resnet50, swin_t, swin_b
import torch
import unsupervisednoveltydetection
import noveltydetectionfeatures
import sklearn.metrics
import noveltydetection
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()

parser.add_argument('--hint-a', type=int, default=None)
parser.add_argument('--hint-b', action='store_true')

args = parser.parse_args()

device = 'cuda:0'

model_ = 'swin_t' # 'swin_t' 'swin_b'  'resnet'

if model_ == 'resnet': 
    backbone = resnet50(pretrained = False)
    backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
if model_ == 'swin_t': 
    backbone = swin_t() 
    backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
if model_ == 'swin_b': 
    backbone = swin_b() 
    backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)

backbone_state_dict = torch.load('unsupervisednoveltydetection/' +model_ +'_backbone_2.pth')
backbone.load_state_dict(backbone_state_dict)
backbone = backbone.to(device)
backbone.eval()
backbone = backbone

classifier = unsupervisednoveltydetection.common.ClassifierV2(256, 5, 12, 8, 72)
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(classifier, 5, 12, 8)
detector = detector.to(device)
state_dict = torch.load('unsupervisednoveltydetection/' +model_ +'_unsupervised_novelty_detection_module_2.pth')
detector.load_state_dict(state_dict['module'])

activation_statistical_model = unsupervisednoveltydetection.common.ActivationStatisticalModel(model_).to(device)
activation_statistical_model.load_state_dict(state_dict['activation_statistical_model'])

testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = './',
    csv_path = 'dataset_v4/dataset_v4_2_val.csv',
    training = False,
    image_batch_size = 16,
    backbone = backbone,
    feature_extraction_device = device
)

incident_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = './',
    csv_path = 'dataset_v4/dataset_v4_2_cal_incident.csv',
    training = False,
    image_batch_size = 16,
    backbone = backbone,
    feature_extraction_device = device
)

spatial_features = []
subject_roi_features = []
object_roi_features = []
verb_roi_features = []
subject_labels = []
object_labels = []
verb_labels = []
subject_box_features = []
object_box_features = []
verb_box_features = []
activation_statistical_scores = []
hint_b = []

with torch.no_grad():
    for example_spatial_features, example_subject_roi_features, example_object_roi_features, example_verb_roi_features, subject_label, object_label, verb_label, example_subject_box_image, example_object_box_image, example_verb_box_image, example_whole_image in testing_set:
        spatial_features.append(example_spatial_features)
        subject_roi_features.append(example_subject_roi_features)
        object_roi_features.append(example_object_roi_features)
        verb_roi_features.append(example_verb_roi_features)
        subject_labels.append(subject_label)
        object_labels.append(object_label)
        verb_labels.append(verb_label)
        subject_box_features.append(backbone(example_subject_box_image.unsqueeze(0)).squeeze(0) if example_subject_box_image is not None else None)
        object_box_features.append(backbone(example_object_box_image.unsqueeze(0)).squeeze(0) if example_object_box_image is not None else None)
        verb_box_features.append(backbone(example_verb_box_image.unsqueeze(0)).squeeze(0) if example_verb_box_image is not None else None)
        whole_image_features = activation_statistical_model.compute_features(backbone, example_whole_image.unsqueeze(0))
        activation_statistical_scores.append(activation_statistical_model.score(whole_image_features).squeeze(0))
        if subject_label == 0 or (subject_label is not None and verb_label == 0) or object_label == 0:
            hint_b.append(True)
        else:
            hint_b.append(False)

    results = detector.scores_and_p_t4(spatial_features, subject_box_features, verb_box_features, object_box_features)
    hint_b = torch.tensor(hint_b, dtype=torch.bool, device=device) if args.hint_b else None

subject_scores = results['subject_novelty_score']
object_scores = results['object_novelty_score']
verb_scores = results['verb_novelty_score']
p_t4 = results['p_t4']

incident_spatial_features = []
incident_subject_roi_features = []
incident_object_roi_features = []
incident_verb_roi_features = []
incident_subject_labels = []
incident_object_labels = []
incident_verb_labels = []
incident_subject_box_features = []
incident_object_box_features = []
incident_verb_box_features = []
incident_activation_statistical_scores = []
incident_hint_b = []

with torch.no_grad():
    for example_spatial_features, example_subject_roi_features, example_object_roi_features, example_verb_roi_features, subject_label, object_label, verb_label, example_subject_box_image, example_object_box_image, example_verb_box_image, example_whole_image in incident_set:
        incident_spatial_features.append(example_spatial_features)
        incident_subject_roi_features.append(example_subject_roi_features)
        incident_object_roi_features.append(example_object_roi_features)
        incident_verb_roi_features.append(example_verb_roi_features)
        incident_subject_labels.append(subject_label)
        incident_object_labels.append(object_label)
        incident_verb_labels.append(verb_label)
        incident_subject_box_features.append(backbone(example_subject_box_image.unsqueeze(0)).squeeze(0) if example_subject_box_image is not None else None)
        incident_object_box_features.append(backbone(example_object_box_image.unsqueeze(0)).squeeze(0) if example_object_box_image is not None else None)
        incident_verb_box_features.append(backbone(example_verb_box_image.unsqueeze(0)).squeeze(0) if example_verb_box_image is not None else None)
        whole_image_features = activation_statistical_model.compute_features(backbone, example_whole_image.unsqueeze(0))
        incident_activation_statistical_scores.append(activation_statistical_model.score(whole_image_features).squeeze(0))
        incident_hint_b.append(True)

    results = detector.scores_and_p_t4(incident_spatial_features, incident_subject_box_features, incident_verb_box_features, incident_object_box_features)
    incident_hint_b = torch.tensor(incident_hint_b, dtype=torch.bool, device=device) if args.hint_b else None

incident_subject_scores = results['subject_novelty_score']
incident_object_scores = results['object_novelty_score']
incident_verb_scores = results['verb_novelty_score']
incident_p_t4 = results['p_t4']

filtered_subject_scores = []
filtered_subject_labels = []
sv_filtered_subject_scores = []
sv_filtered_subject_labels = []
so_filtered_subject_scores = []
so_filtered_subject_labels = []

filtered_verb_scores = []
filtered_verb_labels = []
sv_filtered_verb_scores = []
sv_filtered_verb_labels = []
vo_filtered_verb_scores = []
vo_filtered_verb_labels = []

filtered_object_scores = []
filtered_object_labels = []
so_filtered_object_scores = []
so_filtered_object_labels = []
vo_filtered_object_scores = []
vo_filtered_object_labels = []
for idx in range(len(subject_labels)):
    subject_label = subject_labels[idx]
    verb_label = verb_labels[idx]
    object_label = object_labels[idx]

    subject_score = subject_scores[idx]
    verb_score = verb_scores[idx]
    object_score = object_scores[idx]

    if subject_label is not None:
        filtered_subject_labels.append(subject_label)
        filtered_subject_scores.append(subject_score)
        if verb_label is not None:
            sv_filtered_subject_scores.append(subject_score)
            sv_filtered_subject_labels.append(subject_label)
        if object_label is not None:
            so_filtered_subject_scores.append(subject_score)
            so_filtered_subject_labels.append(subject_label)

    if verb_label is not None:
        filtered_verb_labels.append(verb_label)
        filtered_verb_scores.append(verb_score)
        if subject_label is not None:
            sv_filtered_verb_scores.append(verb_score)
            sv_filtered_verb_labels.append(verb_label)
        if object_label is not None:
            vo_filtered_verb_scores.append(verb_score)
            vo_filtered_verb_labels.append(verb_label)

    if object_label is not None:
        filtered_object_labels.append(object_label)
        filtered_object_scores.append(object_score)
        if subject_label is not None:
            so_filtered_object_scores.append(object_score)
            so_filtered_object_labels.append(object_label)
        if verb_label is not None:
            vo_filtered_object_scores.append(object_score)
            vo_filtered_object_labels.append(object_label)

filtered_subject_scores = torch.stack(filtered_subject_scores, dim = 0)
filtered_subject_labels = torch.stack(filtered_subject_labels, dim = 0)
sv_filtered_subject_scores = torch.stack(sv_filtered_subject_scores, dim = 0)
sv_filtered_subject_labels = torch.stack(sv_filtered_subject_labels, dim = 0)
so_filtered_subject_scores = torch.stack(so_filtered_subject_scores, dim = 0)
so_filtered_subject_labels = torch.stack(so_filtered_subject_labels, dim = 0)

filtered_verb_scores = torch.stack(filtered_verb_scores, dim = 0)
filtered_verb_labels = torch.stack(filtered_verb_labels, dim = 0)
sv_filtered_verb_scores = torch.stack(sv_filtered_verb_scores, dim = 0)
sv_filtered_verb_labels = torch.stack(sv_filtered_verb_labels, dim = 0)
vo_filtered_verb_scores = torch.stack(vo_filtered_verb_scores, dim = 0)
vo_filtered_verb_labels = torch.stack(vo_filtered_verb_labels, dim = 0)

filtered_object_scores = torch.stack(filtered_object_scores, dim = 0)
filtered_object_labels = torch.stack(filtered_object_labels, dim = 0)
so_filtered_object_scores = torch.stack(so_filtered_object_scores, dim = 0)
so_filtered_object_labels = torch.stack(so_filtered_object_labels, dim = 0)
vo_filtered_object_scores = torch.stack(vo_filtered_object_scores, dim = 0)
vo_filtered_object_labels = torch.stack(vo_filtered_object_labels, dim = 0)

incident_filtered_subject_scores = []
incident_filtered_subject_labels = []

incident_filtered_verb_scores = []
incident_filtered_verb_labels = []

incident_filtered_object_scores = []
incident_filtered_object_labels = []
for idx in range(len(incident_subject_labels)):
    subject_label = incident_subject_labels[idx]
    verb_label = incident_verb_labels[idx]
    object_label = incident_object_labels[idx]

    subject_score = incident_subject_scores[idx]
    verb_score = incident_verb_scores[idx]
    object_score = incident_object_scores[idx]

    if subject_label is not None:
        incident_filtered_subject_labels.append(subject_label)
        incident_filtered_subject_scores.append(subject_score)

    if verb_label is not None:
        incident_filtered_verb_labels.append(verb_label)
        incident_filtered_verb_scores.append(verb_score)

    if object_label is not None:
        incident_filtered_object_labels.append(object_label)
        incident_filtered_object_scores.append(object_score)

incident_filtered_subject_scores = torch.stack(incident_filtered_subject_scores, dim = 0)
incident_filtered_subject_labels = torch.stack(incident_filtered_subject_labels, dim = 0)

incident_filtered_verb_scores = torch.stack(incident_filtered_verb_scores, dim = 0)
incident_filtered_verb_labels = torch.stack(incident_filtered_verb_labels, dim = 0)

incident_filtered_object_scores = torch.stack(incident_filtered_object_scores, dim = 0)
incident_filtered_object_labels = torch.stack(incident_filtered_object_labels, dim = 0)

novel_subject_mask = filtered_subject_labels == 0
novel_subject_scores = filtered_subject_scores[novel_subject_mask]
nominal_subject_scores = torch.cat((filtered_subject_scores[~novel_subject_mask], incident_filtered_subject_scores), dim=0)
auc_subject_scores = torch.cat((nominal_subject_scores, novel_subject_scores), dim = 0)
auc_subject_trues = torch.cat((torch.zeros_like(nominal_subject_scores), torch.ones_like(novel_subject_scores)), dim = 0)
subject_auc = sklearn.metrics.roc_auc_score(auc_subject_trues.detach().cpu().numpy(), auc_subject_scores.detach().cpu().numpy())
print(f'Subject AUC: {subject_auc}')
subject_auc = sklearn.metrics.roc_auc_score(auc_subject_trues.detach().cpu().numpy(), auc_subject_scores.detach().cpu().numpy(), max_fpr = 0.25)
print(f'Subject partial AUC: {subject_auc}')

novel_object_mask = filtered_object_labels == 0
novel_object_scores = filtered_object_scores[novel_object_mask]
nominal_object_scores = torch.cat((filtered_object_scores[~novel_object_mask], incident_filtered_object_scores), dim=0)
auc_object_scores = torch.cat((nominal_object_scores, novel_object_scores), dim = 0)
auc_object_trues = torch.cat((torch.zeros_like(nominal_object_scores), torch.ones_like(novel_object_scores)), dim = 0)
object_auc = sklearn.metrics.roc_auc_score(auc_object_trues.detach().cpu().numpy(), auc_object_scores.detach().cpu().numpy())
print(f'Object AUC: {object_auc}')
object_auc = sklearn.metrics.roc_auc_score(auc_object_trues.detach().cpu().numpy(), auc_object_scores.detach().cpu().numpy(), max_fpr = 0.25)
print(f'Object partial AUC: {object_auc}')

novel_verb_mask = filtered_verb_labels == 0
novel_verb_scores = filtered_verb_scores[novel_verb_mask]
nominal_verb_scores = torch.cat((filtered_verb_scores[~novel_verb_mask], incident_filtered_verb_scores), dim=0)
auc_verb_scores = torch.cat((nominal_verb_scores, novel_verb_scores), dim = 0)
auc_verb_trues = torch.cat((torch.zeros_like(nominal_verb_scores), torch.ones_like(novel_verb_scores)), dim = 0)
verb_auc = sklearn.metrics.roc_auc_score(auc_verb_trues.detach().cpu().numpy(), auc_verb_scores.detach().cpu().numpy())
print(f'Verb AUC: {verb_auc}')
verb_auc = sklearn.metrics.roc_auc_score(auc_verb_trues.detach().cpu().numpy(), auc_verb_scores.detach().cpu().numpy(), max_fpr = 0.25)
print(f'Verb partial AUC: {verb_auc}')

print()

fig, ax = plt.subplots()
ax.set_title('Verb novelty scores vs subject novelty scores')
ax.set_xlabel('Subject novelty scores')
ax.set_ylabel('Verb novelty scores')
ax.scatter(sv_filtered_subject_scores.detach().cpu().numpy(), sv_filtered_verb_scores.detach().cpu().numpy())
a = torch.stack((sv_filtered_subject_scores, torch.ones_like(sv_filtered_subject_scores)), dim = 1)
y = sv_filtered_verb_scores
x = torch.matmul(torch.matmul(torch.linalg.inv(torch.matmul(a.T, a)), a.T), sv_filtered_verb_scores)
y_hat = torch.matmul(a, x)
ax.plot(sv_filtered_subject_scores.detach().cpu().numpy(), y_hat.detach().cpu().numpy(), color = 'black')
r = torch.corrcoef(torch.stack((sv_filtered_subject_scores, sv_filtered_verb_scores), dim = 0))[0, 1]
text = f'r = {float(r):.2f}'
props = dict(boxstyle = 'round', facecolor = 'tab:orange', alpha = 0.85)
ax.text(0.95, 0.05, text, transform = ax.transAxes, fontsize = 14, verticalalignment = 'bottom', horizontalalignment = 'right', bbox = props)
fig.savefig('subject_verb_scatter.jpg')
plt.close(fig)

case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression()
case_1_logistic_regression.load_state_dict(state_dict['case_1_logistic_regression'])
case_1_logistic_regression = case_1_logistic_regression.to(device)

case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression()
case_2_logistic_regression.load_state_dict(state_dict['case_2_logistic_regression'])
case_2_logistic_regression = case_2_logistic_regression.to(device)

case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression()
case_3_logistic_regression.load_state_dict(state_dict['case_3_logistic_regression'])
case_3_logistic_regression = case_3_logistic_regression.to(device)

#p_type, p_n = noveltydetection.utils.compute_probability_novelty(subject_scores, verb_scores, object_scores, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, ignore_t2_in_pni = True, p_t4 = None)
#p_type, p_n = noveltydetection.utils.compute_probability_novelty(subject_scores, verb_scores, object_scores, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, ignore_t2_in_pni = True, p_t4 = p_t4)
#p_type, p_n = noveltydetection.utils.compute_probability_novelty(subject_scores, verb_scores, object_scores, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, ignore_t2_in_pni = False, p_t4 = None)
p_type, p_n, separated_p_type = noveltydetection.utils.compute_probability_novelty(subject_scores, verb_scores, object_scores, activation_statistical_scores, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, ignore_t2_in_pni = False, p_t4 = p_t4, hint_a=args.hint_a, hint_b=hint_b) # This gives the best results (thankfully)
incident_p_type, incident_p_n, incident_separated_p_type = noveltydetection.utils.compute_probability_novelty(incident_subject_scores, incident_verb_scores, incident_object_scores, incident_activation_statistical_scores, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, ignore_t2_in_pni = False, p_t4 = incident_p_t4, hint_a=args.hint_a, hint_b=incident_hint_b)
print(p_type)
print(incident_p_type)

type_1_p_n = []
type_1_p_type = []
type_1_subject_scores = []

type_3_p_n = []
type_3_p_type = []
type_3_object_scores = []

type_2_p_n = []
type_2_p_type = []
type_2_verb_scores = []

nominal_p_n = []
nominal_p_type = []
nominal_subject_scores = []
nominal_verb_scores = []
nominal_object_scores = []
for idx in range(len(p_n)):
    if subject_labels[idx] == 0 and object_labels[idx] != 0 and verb_labels[idx] != 0:
        type_1_p_n.append(p_n[idx])
        type_1_p_type.append(p_type[idx])
        type_1_subject_scores.append(subject_scores[idx])
    elif object_labels[idx] == 0 and subject_labels[idx] != 0 and (verb_labels[idx] != 0 or subject_labels[idx] is None):
        type_3_p_n.append(p_n[idx])
        type_3_p_type.append(p_type[idx])
        type_3_object_scores.append(object_scores[idx])
    elif subject_labels[idx] is not None and verb_labels[idx] == 0 and subject_labels[idx] != 0 and object_labels[idx] != 0:
        type_2_p_n.append(p_n[idx])
        type_2_p_type.append(p_type[idx])
        type_2_verb_scores.append(verb_scores[idx])
    elif subject_labels[idx] != 0 and object_labels[idx] != 0 and (verb_labels[idx] != 0 or subject_labels[idx] is None):
        nominal_p_n.append(p_n[idx])
        nominal_p_type.append(p_type[idx])
        if subject_labels[idx] is not None:
            nominal_subject_scores.append(subject_scores[idx])
            nominal_verb_scores.append(verb_scores[idx])
        if object_labels[idx] is not None:
            nominal_object_scores.append(object_scores[idx])

type_1_p_n = torch.stack(type_1_p_n, dim = 0)
type_1_p_type = torch.stack(type_1_p_type, dim = 0)
type_1_subject_scores = torch.stack(type_1_subject_scores, dim = 0)

type_3_p_n = torch.stack(type_3_p_n, dim = 0)
type_3_p_type = torch.stack(type_3_p_type, dim = 0)
type_3_object_scores = torch.stack(type_3_object_scores, dim = 0)

type_2_p_n = torch.stack(type_2_p_n, dim = 0)
type_2_p_type = torch.stack(type_2_p_type, dim = 0)
type_2_verb_scores = torch.stack(type_2_verb_scores, dim = 0)

type_67_p_n = incident_p_n
type_67_p_type = incident_p_type

nominal_p_n = torch.stack(nominal_p_n, dim = 0)
nominal_p_type = torch.stack(nominal_p_type, dim = 0)
nominal_subject_scores = torch.stack(nominal_subject_scores, dim = 0)
nominal_object_scores = torch.stack(nominal_object_scores, dim = 0)
nominal_verb_scores = torch.stack(nominal_verb_scores, dim = 0)

argmax_type_1_p_type = torch.argmax(type_1_p_type, dim = 1)
argmax_type_2_p_type = torch.argmax(type_2_p_type, dim = 1)
argmax_type_3_p_type = torch.argmax(type_3_p_type, dim = 1)
argmax_type_67_p_type = torch.argmax(type_67_p_type, dim=1)
argmax_nominal_p_type = torch.argmax(nominal_p_type, dim = 1)

print(f'Predicted novelty types for type 1 data: {argmax_type_1_p_type + 1}')
print(f'Predicted novelty types for type 2 data: {argmax_type_2_p_type + 1}')
print(f'Predicted novelty types for type 3 data: {argmax_type_3_p_type + 1}')
print(f'Predicted novelty types for type 6/7 data: {argmax_type_67_p_type + 1}')
print(f'Predicted novelty types for nominal data: {argmax_nominal_p_type + 1}')

accuracy_argmax_type_1_p_type = (argmax_type_1_p_type == 0).int().sum() / float(len(argmax_type_1_p_type))
accuracy_argmax_type_2_p_type = (argmax_type_2_p_type == 1).int().sum() / float(len(argmax_type_2_p_type))
accuracy_argmax_type_3_p_type = (argmax_type_3_p_type == 2).int().sum() / float(len(argmax_type_3_p_type))
accuracy_argmax_type_67_p_type = (argmax_type_67_p_type == 4).int().sum() / float(len(argmax_type_67_p_type))

print(f'Predicted novelty type accuracy for type 1 data: {accuracy_argmax_type_1_p_type}')
print(f'Predicted novelty type accuracy for type 2 data: {accuracy_argmax_type_2_p_type}')
print(f'Predicted novelty type accuracy for type 3 data: {accuracy_argmax_type_3_p_type}')
print(f'Predicted novelty type accuracy for type 6/7 data: {accuracy_argmax_type_67_p_type}')

nominal_x = torch.ones_like(nominal_p_n) * 0
type_1_x = torch.ones_like(type_1_p_n) * 1
type_2_x = torch.ones_like(type_2_p_n) * 2
type_3_x = torch.ones_like(type_3_p_n) * 3
type_67_x = torch.ones_like(type_67_p_n) * 4

scores = torch.cat((nominal_subject_scores, type_1_subject_scores), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_subject_scores), torch.ones_like(type_1_subject_scores)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 1 subject score AUC: {auc}')

scores = torch.cat((nominal_verb_scores, type_2_verb_scores), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_verb_scores), torch.ones_like(type_2_verb_scores)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 2 verb score AUC: {auc}')

scores = torch.cat((nominal_object_scores, type_3_object_scores), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_object_scores), torch.ones_like(type_3_object_scores)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 3 object score AUC: {auc}')

# TODO Type 6/7 activation statistical score AUC

print()

scores = torch.cat((nominal_p_n, type_1_p_n), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_p_n), torch.ones_like(type_1_p_n)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 1 P_N AUC: {auc}')
predictions = scores > 0.5
n_correct = (predictions == trues).to(torch.int).sum().detach().cpu().item()
accuracy = float(n_correct) / len(trues)
print(f'Type 1 P_N Accuracy: {accuracy}')

scores = torch.cat((nominal_p_n, type_2_p_n), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_p_n), torch.ones_like(type_2_p_n)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 2 P_N AUC: {auc}')
predictions = scores > 0.5
n_correct = (predictions == trues).to(torch.int).sum().detach().cpu().item()
accuracy = float(n_correct) / len(trues)
print(f'Type 2 P_N Accuracy: {accuracy}')

scores = torch.cat((nominal_p_n, type_3_p_n), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_p_n), torch.ones_like(type_3_p_n)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 3 P_N AUC: {auc}')
predictions = scores > 0.5
n_correct = (predictions == trues).to(torch.int).sum().detach().cpu().item()
accuracy = float(n_correct) / len(trues)
print(f'Type 3 P_N Accuracy: {accuracy}')

scores = torch.cat((nominal_p_n, type_67_p_n), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_p_n), torch.ones_like(type_67_p_n)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 6/7 P_N AUC: {auc}')
predictions = scores > 0.5
n_correct = (predictions == trues).to(torch.int).sum().detach().cpu().item()
accuracy = float(n_correct) / len(trues)
print(f'Type 6/7 P_N Accuracy: {accuracy}')

scores = torch.cat((nominal_p_n, type_1_p_n, type_2_p_n, type_3_p_n, type_67_p_n), dim = 0)
trues = torch.cat((torch.zeros_like(nominal_p_n), torch.ones_like(type_1_p_n), torch.ones_like(type_2_p_n), torch.ones_like(type_3_p_n), torch.ones_like(type_67_p_n)), dim = 0)
auc = sklearn.metrics.roc_auc_score(trues.detach().cpu().numpy(), scores.detach().cpu().numpy())
print(f'Type 0 vs Type Non-0 P_N AUC: {auc}')
predictions = scores > 0.5
n_correct = (predictions == trues).to(torch.int).sum().detach().cpu().item()
accuracy = float(n_correct) / len(trues)
print(f'Type 0 vs Type Non-0 P_N Accuracy: {accuracy}')

print()

print(f'Average P(N_i) over nominals: {nominal_p_n.mean()}')
print(f'Average P(N_i) over type 1 novelty: {type_1_p_n.mean()}')
print(f'Average P(N_i) over type 2 novelty: {type_2_p_n.mean()}')
print(f'Average P(N_i) over type 3 novelty: {type_3_p_n.mean()}')
print(f'Average P(N_i) over type 6/7 novelty: {type_67_p_n.mean()}')
print(f'Std P(N_i) over nominals: {nominal_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 1 novelty: {type_1_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 2 novelty: {type_2_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 3 novelty: {type_3_p_n.std(unbiased = True)}')
print(f'Std P(N_i) over type 6/7 novelty: {type_67_p_n.std(unbiased = True)}')
print(f'Number of nominal examples: {len(nominal_p_n)}')
print(f'Number of type 1 examples: {len(type_1_p_n)}')
print(f'Number of type 2 examples: {len(type_2_p_n)}')
print(f'Number of type 3 examples: {len(type_3_p_n)}')
print(f'Number of type 6/7 examples: {len(type_67_p_n)}')

print()

plt.scatter(nominal_x.detach().cpu().numpy(), nominal_p_n.detach().cpu().numpy(), label = 'No novelty', alpha = 0.2)
plt.scatter(type_1_x.detach().cpu().numpy(), type_1_p_n.detach().cpu().numpy(), label = 'Type 1 novelty', alpha = 0.2)
plt.scatter(type_2_x.detach().cpu().numpy(), type_2_p_n.detach().cpu().numpy(), label = 'Type 2 novelty', alpha = 0.2)
plt.scatter(type_3_x.detach().cpu().numpy(), type_3_p_n.detach().cpu().numpy(), label = 'Type 3 novelty', alpha = 0.2)
plt.scatter(type_67_x.detach().cpu().numpy(), type_67_p_n.detach().cpu().numpy(), label = 'Type 6/7 novelty', alpha = 0.2)
plt.ylabel('P(N_i)')
plt.legend()
ax = plt.gca()
ax.axes.xaxis.set_visible(False)
plt.savefig('plt.jpg')
