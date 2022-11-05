import pickle
import time

import torch
from torchvision.models import resnet50, swin_t, swin_b

import unsupervisednoveltydetection
import noveltydetection

device = 'cuda:0'
model_ = 'swin_t' # 'swin_t' 'swin_b' 'resnet'

if model_ == 'resnet': 
    backbone = resnet50(weights="IMAGENET1K_V1") # pretrained = True, 
    backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
if model_ == 'swin_t': 
    backbone = swin_t(weights="IMAGENET1K_V1") # pretrained = True, 
    backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
if model_ == 'swin_b': 
    backbone = swin_b(weights="IMAGENET1K_V1") # pretrained = True, 
    backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)

backbone = backbone.to(device)

classifier = unsupervisednoveltydetection.common.ClassifierV2(256, 5, 12, 8, 72)
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(classifier, 5, 12, 8)
detector = detector.to(device)

case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(device)
case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(device)
case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(device)

activation_statistical_model = unsupervisednoveltydetection.common.ActivationStatisticalModel(model_).to(device)

trainer = unsupervisednoveltydetection.training.NoveltyDetectorTrainer('./', 'dataset_v4/dataset_v4_2_train.csv', 'dataset_v4/dataset_v4_2_val.csv', 'dataset_v4/dataset_v4_2_val_incident.csv', 'dataset_v4/dataset_v4_2_cal_corruption.csv', 64, model_ = model_)

start_time = time.time()
trainer.prepare_for_retraining(backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model)
trainer.train_novelty_detection_module(backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model)
end_time = time.time()
print(f'Time: {end_time - start_time}')

def state_dict(module):
    raw_state_dict = module.state_dict()
    return {k: v.cpu() for k, v in raw_state_dict.items()}

detector_state_dicts = {}
detector_state_dicts['classifier'] = detector.classifier.state_dict()
detector_state_dicts['confidence_calibrator'] = detector.confidence_calibrator.state_dict()
with open('known_combinations.pth', 'rb') as f:
    known_combinations = pickle.load(f)
detector_state_dicts['known_combinations'] = known_combinations

module_state_dicts = {}
module_state_dicts['module'] = detector_state_dicts
module_state_dicts['case_1_logistic_regression'] = state_dict(case_1_logistic_regression)
module_state_dicts['case_2_logistic_regression'] = state_dict(case_2_logistic_regression)
module_state_dicts['case_3_logistic_regression'] = state_dict(case_3_logistic_regression)
module_state_dicts['activation_statistical_model'] = activation_statistical_model.state_dict()

torch.save(state_dict(backbone), 'unsupervisednoveltydetection/' +model_ +'_backbone_2.pth')
torch.save(detector_state_dicts['classifier'], 'unsupervisednoveltydetection/' +model_ +'_classifier_2.pth')
torch.save(module_state_dicts, 'unsupervisednoveltydetection/' +model_ +'_unsupervised_novelty_detection_module_2.pth')
