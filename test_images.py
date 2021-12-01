import unsupervisednoveltydetection
import noveltydetectionfeatures
import noveltydetection
import torch

device = 'cuda:0'
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 5, 12, 8)
detector = detector.to(device)
subject_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
object_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)
verb_score_ctx = noveltydetection.utils.ScoreContext(noveltydetection.utils.ScoreContext.Source.UNSUPERVISED)

state_dict = torch.load('unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
detector.load_state_dict(state_dict['module'])
subject_score_ctx.load_state_dict(state_dict['subject_score_context'])
object_score_ctx.load_state_dict(state_dict['object_score_context'])
verb_score_ctx.load_state_dict(state_dict['verb_score_context'])

testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = 'Custom',
    csv_path = 'Custom/annotations/dataset_v3_val.csv',
    training = False,
    image_batch_size = 16,
    feature_extraction_device = device
)

image_ids = ['00164', '00339', '00680', '00694']
image_indices = [172, 98, 114, 150]
spatial_features = []
subject_appearance_features = []
object_appearance_features = []
verb_appearance_features = []
for idx, image_idx in enumerate(image_indices):
    example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label = testing_set[image_idx]
    print(f'Image id: {image_ids[idx]}')
    print(f'Label: {subject_label}/{verb_label}/{object_label}')
    spatial_features.append(example_spatial_features)
    subject_appearance_features.append(example_subject_appearance_features)
    object_appearance_features.append(example_object_appearance_features)
    verb_appearance_features.append(example_verb_appearance_features)

results = detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = device))

subject_scores = results['subject_novelty_score']
object_scores = results['object_novelty_score']
verb_scores = results['verb_novelty_score']
p_known_svo = results['p_known_svo']
p_known_sv = results['p_known_sv']
p_known_so = results['p_known_so']
p_known_vo = results['p_known_vo']
p_type, p_n, novel_subject_probs, novel_verb_probs, novel_object_probs, p_t4 = noveltydetection.utils.compute_probability_novelty_all(
    subject_scores,
    verb_scores,
    object_scores,
    p_known_svo,
    p_known_sv,
    p_known_so,
    p_known_vo,
    subject_score_ctx,
    verb_score_ctx,
    object_score_ctx
)

print(p_n)
print(novel_subject_probs)
print(novel_object_probs)
print(novel_verb_probs)
print(p_t4)

known_top3 = detector.known_top3(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features)
print(known_top3)
