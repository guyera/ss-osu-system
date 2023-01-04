import pickle
import time
import os

import torch

import boxclassifier
import tupleprediction
from backbone import Backbone
from scoring import\
    ActivationStatisticalModel,\
    make_logit_scorer,\
    CompositeScorer

device = 'cuda:0'
architecture = Backbone.Architecture.swin_t
backbone = Backbone(architecture)
backbone = backbone.to(device)

n_known_species_cls = 10
n_species_cls = 30 # TODO Determine
n_known_activity_cls = 2
n_activity_cls = 4 # TODO Determine

classifier = boxclassifier.ClassifierV2(256, n_species_cls, n_activity_cls)
classifier = classifier.to(device)
confidence_calibrator = boxclassifier.ConfidenceCalibrator()
confidence_calibrator = confidence_calibrator.to(device)
tuple_predictor = tupleprediction.TuplePredictor(
    n_known_species_cls,
    n_known_activity_cls
)

activation_statistical_model = ActivationStatisticalModel(
    architecture
).to(device)
logit_scorer = make_logit_scorer(n_known_species_cls, n_known_activity_cls)
scorer = CompositeScorer((activation_statistical_model, logit_scorer))

novelty_type_classifier = tupleprediction.NoveltyTypeClassifier(
    scorer.n_scores()
).to(device)

trainer = tupleprediction.training.TuplePredictorTrainer('./', 'dataset_v4/train.csv', 'dataset_v4/valid.csv', 64, n_species_cls, n_activity_cls, n_known_species_cls, n_known_activity_cls)

start_time = time.time()
trainer.prepare_for_retraining(backbone, classifier, confidence_calibrator, novelty_type_classifier, activation_statistical_model)
trainer.train_novelty_detection_module(backbone, classifier, confidence_calibrator, novelty_type_classifier, activation_statistical_model, scorer)
end_time = time.time()
print(f'Time: {end_time - start_time}')

with open('known_combinations.pth', 'rb') as f:
    known_combinations = pickle.load(f)

tuple_prediction_state_dicts = {}
tuple_prediction_state_dicts['known_combinations'] = known_combinations
tuple_prediction_state_dicts['novelty_type_classifier'] = novelty_type_classifier.state_dict()
tuple_prediction_state_dicts['activation_statistical_model'] = activation_statistical_model.state_dict()

save_dir = os.path.join(
    'pretrained-models',
    architecture.value['name']
)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

torch.save(
    backbone.state_dict(),
    os.path.join(save_dir, 'backbone.pth')
)
torch.save(
    classifier.state_dict(),
    os.path.join(save_dir, 'classifier.pth')
)
torch.save(
    confidence_calibrator.state_dict(),
    os.path.join(save_dir, 'confidence-calibrator.pth')
)
torch.save(
    tuple_prediction_state_dicts,
    os.path.join(save_dir, 'tuple-prediction.pth')
)
