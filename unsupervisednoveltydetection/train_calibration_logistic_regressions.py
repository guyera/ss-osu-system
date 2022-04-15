import torch
from torchvision.models import resnet50
import unsupervisednoveltydetection
import noveltydetectionfeatures
import sklearn.metrics
import noveltydetection
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Train calibration logistic regressions')

parser.add_argument(
    '--device',
    type = str,
    default = 'cpu'
)
parser.add_argument(
    '--epochs',
    type = int,
    default = 25000
)
parser.add_argument(
    '--lr',
    type = float,
    default = 0.05
)
parser.add_argument(
    '--backbone-load-file',
    type = str,
    required = True
)
parser.add_argument(
    '--detector-load-file',
    type = str,
    required = True
)
parser.add_argument(
    '--calibration-logistic-regressions-save-file',
    type = str,
    required = True
)

args = parser.parse_args()

backbone = resnet50(pretrained = False)
backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
backbone_state_dict = torch.load(args.backbone_load_file)
backbone.load_state_dict(backbone_state_dict)
backbone.eval()
backbone = backbone.to(args.device)

classifier = unsupervisednoveltydetection.common.ClassifierV2(256, 5, 12, 8, 72)
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(classifier, 5, 12, 8)
detector = detector.to(args.device)

state_dict = torch.load(args.detector_load_file)
detector.load_state_dict(state_dict)

testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = 'Custom',
    csv_path = 'Custom/annotations/dataset_v4_val.csv',
    training = False,
    image_batch_size = 16,
    backbone = backbone,
    feature_extraction_device = args.device,
    cache_to_disk = False
)

spatial_features = []
subject_labels = []
object_labels = []
verb_labels = []
subject_features = []
object_features = []
verb_features = []
        
with torch.no_grad():
    for example_spatial_features, _, _, _, subject_label, object_label, verb_label, subject_image, object_image, verb_image in testing_set:
        spatial_features.append(example_spatial_features)
        subject_labels.append(subject_label)
        object_labels.append(object_label)
        verb_labels.append(verb_label)
        subject_features.append(backbone(subject_image.unsqueeze(0)).squeeze(0) if subject_image is not None else None)
        object_features.append(backbone(object_image.unsqueeze(0)).squeeze(0) if object_image is not None else None)
        verb_features.append(backbone(verb_image.unsqueeze(0)).squeeze(0) if verb_image is not None else None)

results = detector(spatial_features, subject_features, verb_features, object_features, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = args.device))

subject_scores = results['subject_novelty_score']
object_scores = results['object_novelty_score']
verb_scores = results['verb_novelty_score']

# Reformat the S/V/O scores and labels into case 1/2/3 tensors. e.g. the
# case 1 score tensor will be Nx3, since case 1 means there's a present subject,
# object, and verb, and as such there will be 3 total novelty scores. In
# contrast, the case 3 score tensor will only be Nx1. All of the label tensors
# are just long tensors of size N, but their value ranges depends on the
# case (for instance, case 3 cannot by novelty type 1, since there's no
# subject).
case_1_scores, case_2_scores, case_3_scores, case_1_labels, case_2_labels, case_3_labels = noveltydetection.utils.separate_scores_and_labels(subject_scores, object_scores, verb_scores, subject_labels, object_labels, verb_labels)

print(case_1_scores.shape)
print(case_2_scores.shape)
print(case_3_scores.shape)

# Fit logistic regressions
case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(args.device)
case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(args.device)
case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(args.device)

noveltydetection.utils.fit_logistic_regression(case_1_logistic_regression, case_1_scores, case_1_labels, args.epochs, quiet = False)
noveltydetection.utils.fit_logistic_regression(case_2_logistic_regression, case_2_scores, case_2_labels, args.epochs, quiet = False)
noveltydetection.utils.fit_logistic_regression(case_3_logistic_regression, case_3_scores, case_3_labels, args.epochs, quiet = False)

def _state_dict(module):
    return {k: v.cpu() for k, v in module.state_dict().items()}

case_1_state_dict = _state_dict(case_1_logistic_regression)
case_2_state_dict = _state_dict(case_2_logistic_regression)
case_3_state_dict = _state_dict(case_3_logistic_regression)

state_dict = {}
state_dict['case_1_logistic_regression'] = case_1_state_dict
state_dict['case_2_logistic_regression'] = case_2_state_dict
state_dict['case_3_logistic_regression'] = case_3_state_dict
state_dict['case_1_scores'] = case_1_scores.cpu()
state_dict['case_2_scores'] = case_2_scores.cpu()
state_dict['case_3_scores'] = case_3_scores.cpu()
state_dict['case_1_labels'] = case_1_labels.cpu()
state_dict['case_2_labels'] = case_2_labels.cpu()
state_dict['case_3_labels'] = case_3_labels.cpu()

# torch.save(state_dict, args.calibration_logistic_regressions_save_file)
