import pickle
import time
import os

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import boxclassifier
import tupleprediction
from backbone import Backbone
from scoring import\
    ActivationStatisticalModel,\
    make_logit_scorer,\
    CompositeScorer
from data.custom import build_species_label_mapping
from labelmapping import LabelMapper

dist.init_process_group('nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()
device_id = rank
device = f'cuda:{device_id}'

def train_sampler_fn(train_dataset):
    return DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )

architecture = Backbone.Architecture.swin_t
backbone = Backbone(architecture)
backbone = backbone.to(device)

training_image_batch_size = 64
training_batch_size = 64
training_buffer_size = 64
n_known_species_cls = 10
n_species_cls = 30 # TODO Determine
n_known_activity_cls = 2
n_activity_cls = 4 # TODO Determine
train_csv_path = 'dataset_v4/train.csv'
val_csv_path = 'dataset_v4/valid.csv'

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

label_mapping = build_species_label_mapping(train_csv_path)
trainer = tupleprediction.training.TuplePredictorTrainer('dataset_v4/', train_csv_path, val_csv_path, training_image_batch_size, training_batch_size, training_buffer_size, n_species_cls, n_activity_cls, n_known_species_cls, n_known_activity_cls, label_mapping, write_cache=(rank == 0))

trainer.prepare_for_retraining(backbone, classifier, confidence_calibrator, novelty_type_classifier, activation_statistical_model)

backbone = DDP(backbone, device_ids=[device_id])
classifier.ddp(device_ids=[device_id])

start_time = time.time()

species_classifier = classifier.species_classifier
activity_classifier = classifier.activity_classifier
species_calibrator = confidence_calibrator.species_calibrator
activity_calibrator = confidence_calibrator.activity_calibrator

# Retrain the backbone and classifiers
trainer.train_backbone_and_classifiers(
    backbone,
    species_classifier,
    activity_classifier,
    train_sampler_fn=train_sampler_fn
)

trainer.fit_activation_statistics(
    backbone.module,
    activation_statistical_model
)

# Retrain the classifier's temperature scaling calibrators
trainer.calibrate_temperature_scalers(
    backbone.module,
    species_classifier,
    activity_classifier,
    species_calibrator,
    activity_calibrator
)

# Retrain the logistic regressions
trainer.train_novelty_type_logistic_regressions(
    backbone.module,
    species_classifier,
    activity_classifier,
    novelty_type_classifier,
    activation_statistical_model,
    scorer
)

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

if rank == 0:
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
