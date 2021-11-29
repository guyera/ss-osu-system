import unsupervisednoveltydetection
import pickle
import torch

with open('Custom/us_s_learning_curves/results1.pkl', 'rb') as f:
    unsupervised_results = pickle.load(f)

classifier_state_dict = unsupervised_results['classifier']

subject_auc = unsupervised_results['subject_auc']
object_auc = unsupervised_results['object_auc']
verb_auc = unsupervised_results['verb_auc']

partial_subject_auc = unsupervised_results['partial_subject_auc']
partial_object_auc = unsupervised_results['partial_object_auc']
partial_verb_auc = unsupervised_results['partial_verb_auc']

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

# classifier = classifier.to(<insert your device here, e.g. 'cuda:0'>)

classifier.load_state_dict(classifier_state_dict)

print(subject_auc)
print(partial_subject_auc)
print(id_subject_labels)
print(ood_subject_labels)
print(object_auc)
print(partial_object_auc)
print(id_object_labels)
print(ood_object_labels)
print(verb_auc)
print(partial_verb_auc)
print(id_verb_labels)
print(ood_verb_labels)

'''
# Notes:

# Note that unsupervisednoveltydetection.common.Classifier is a bit different
# from unsupervisednoveltydetection.UnsupervisedNoveltyDetector.

# First, you'll have to reshape (flatten) appearance features, and you'll also
# have to concatenate the spatial features and the verb appearance features
# together. Suppose you have feature tensors for a SINGLE image, i.e.
# XXX_appearance_features tensors of size [256, 7, 7] and a spatial_features
# tensor of size [2, 36] (or whatever it is...). Then, do the following:

unsupervised_subject_features = torch.flatten(
    subject_appearance_features
)

unsupervised_object_features = torch.flatten(
    object_appearance_features
)

flattened_spatial_features = torch.flatten(spatial_features)
flattened_verb_appearance_features = torch.flatten(
    verb_appearance_features
)
unsupervised_verb_features =\
    torch.cat((flattened_spatial_features, flattened_verb_appearance_features), dim = 0)

# Then, to get scores, do the following:
unsupervised_subject_scores = classifier.score_subject(unsupervised_subject_features.unsqueeze(0)).squeeze(0)
unsupervised_object_scores = classifier.score_object(unsupervised_object_features.unsqueeze(0)).squeeze(0)
unsupervised_verb_scores = classifier.score_verb(unsupervised_verb_features.unsqueeze(0)).squeeze(0)

# You have to unsqueeze the 0 dimension because the classifier works with
# batches, but all you have is a feature for a single image. You then squeeze
# out that extra dimension afterwards (if you want to). Internally, this is
# what the unsupervised novelty detection module does when you pass it the
# raw features.

# Also note that if an image has "None" as one or more of its features, the
# corresponding scores should also be "None". You'll have to set these manually.
'''
