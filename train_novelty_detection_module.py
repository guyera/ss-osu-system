import pickle
import time
import os

import torch

import unsupervisednoveltydetection
import noveltydetection
from backbone import Backbone

device = 'cuda:0'
architecture = Backbone.Architecture.swin_t
backbone = Backbone(architecture)
backbone = backbone.to(device)

classifier = unsupervisednoveltydetection.ClassifierV2(256, 5, 12, 8, 72)
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(classifier, 5, 12, 8)
detector = detector.to(device)

case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(device)
case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(device)
case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(device)

activation_statistical_model = noveltydetection.utils.ActivationStatisticalModel(architecture).to(device)

trainer = unsupervisednoveltydetection.training.NoveltyDetectorTrainer('./', 'dataset_v4/dataset_v4_2_train.csv', 'dataset_v4/dataset_v4_2_val.csv', 'dataset_v4/dataset_v4_2_cal_incident.csv', 'dataset_v4/dataset_v4_2_cal_corruption.csv', 64)

start_time = time.time()
trainer.prepare_for_retraining(backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model)
trainer.train_novelty_detection_module(backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model)
end_time = time.time()
print(f'Time: {end_time - start_time}')

detector_state_dicts = {}
detector_state_dicts['classifier'] = detector.classifier.state_dict()
detector_state_dicts['confidence_calibrator'] = detector.confidence_calibrator.state_dict()
with open('known_combinations.pth', 'rb') as f:
    known_combinations = pickle.load(f)
detector_state_dicts['known_combinations'] = known_combinations

module_state_dicts = {}
module_state_dicts['module'] = detector_state_dicts
module_state_dicts['case_1_logistic_regression'] = case_1_logistic_regression.state_dict()
module_state_dicts['case_2_logistic_regression'] = case_2_logistic_regression.state_dict()
module_state_dicts['case_3_logistic_regression'] = case_3_logistic_regression.state_dict()
module_state_dicts['activation_statistical_model'] = activation_statistical_model.state_dict()

save_dir = os.path.join(
    'pretrained-models',
    architecture.value['name']
)
os.makedirs(save_dir)
torch.save(
    backbone.state_dict(),
    os.path.join(save_dir, 'backbone.pth')
)
torch.save(
    module_state_dicts,
    os.path.join(save_dir, 'unsupervised_novelty_detection_module.pth')
)
