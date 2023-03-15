from enum import Enum
import pickle
import time
import os
import argparse
from copy import deepcopy

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import boxclassifier
import tupleprediction
from tupleprediction.training import\
    Augmentation,\
    SchedulerType,\
    get_transforms,\
    get_datasets,\
    TuplePredictorTrainer,\
    EndToEndClassifierTrainer,\
    LogitLayerClassifierTrainer
from backbone import Backbone
from scoring import\
    ActivationStatisticalModel,\
    make_logit_scorer,\
    CompositeScorer
from labelmapping import LabelMapper
from utils import gen_custom_collate
from data.custom import build_species_label_mapping

class ClassifierTrainer(Enum):
    end_to_end = 'end-to-end'
    logit_layer = 'logit-layer'

    def __str__(self):
        return self.value

parser = argparse.ArgumentParser()

parser.add_argument(
    '--data-root',
    type=str,
    help='Data root directory',
    required=True
)

parser.add_argument(
    '--root-cache-dir',
    type=str,
    help='Root cache directory',
    required=True
)

parser.add_argument(
    '--train-csv-path',
    type=str,
    help='Path to training CSV',
    required=True
)

parser.add_argument(
    '--cal-csv-path',
    type=str,
    help='Path to calibration CSV',
    required=True
)

parser.add_argument(
    '--lr',
    type=float,
    default=0.005,
    help='Learning rate for backbone and classifiers'
)

parser.add_argument(
    '--label-smoothing',
    type=float,
    default=0.05,
    help='Label smoothing for training backbone and classifiers'
)

parser.add_argument(
    '--augmentation',
    type=lambda augmentation: Augmentation[augmentation],
    choices=list(Augmentation),
    default=Augmentation.rand_augment,
    help='Augmentation strategy'
)

parser.add_argument(
    '--batch-size',
    type=int,
    default=64,
    help='Batch size for training backbone and box classifiers'
)

parser.add_argument(
    '--n-known-val',
    type=int,
    default=4068,
    help='Number of known validation instances to pull from the training CSV'
)

parser.add_argument(
    '--scheduler-type',
    type=lambda scheduler_type: SchedulerType[scheduler_type],
    choices=list(SchedulerType),
    default=SchedulerType.cosine,
    help='Type of learning rate scheduler to use'
)

parser.add_argument(
    '--max-epochs',
    type=int,
    default=600,
    help='Max number of epochs for training backbone and classifiers'
)

parser.add_argument(
    '--root-checkpoint-dir',
    type=str,
    default=None,
    help='Checkpoint directory'
)

parser.add_argument(
    '--root-log-dir',
    type=str,
    default=None,
    help='Log directory'
)

parser.add_argument(
    '--classifier-trainer',
    type=ClassifierTrainer,
    choices=list(ClassifierTrainer),
    default=ClassifierTrainer.end_to_end,
    help='Classifier trainer to use'
)

parser.add_argument(
    '--pretrained-backbone-path',
    type=str,
    default=None
)

parser.add_argument(
    '--precomputed-feature-dir',
    type=str,
    default='./.features/resizepad=224/none/normalized'
)

args = parser.parse_args()

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
backbone = Backbone(architecture, pretrained=False).to(device)
backbone = DDP(backbone, device_ids=[device_id])

if args.pretrained_backbone_path is not None:
    backbone_state_dict = torch.load(
        args.pretrained_backbone_path,
        map_location='cuda:0'
    )
    backbone.load_state_dict(backbone_state_dict)

n_known_species_cls = 10
n_species_cls = 31
n_known_activity_cls = 2
n_activity_cls = 7

classifier = boxclassifier.ClassifierV2(
    256,
    n_species_cls,
    n_activity_cls
).to(device)
classifier.ddp(device_ids=[device_id])

confidence_calibrator = boxclassifier.ConfidenceCalibrator()
confidence_calibrator = confidence_calibrator.to(device)
tuple_predictor = tupleprediction.TuplePredictor(
    n_species_cls,
    n_activity_cls,
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

label_mapping = build_species_label_mapping(args.train_csv_path)
static_label_mapper = LabelMapper(deepcopy(label_mapping), update=False)
dynamic_label_mapper = LabelMapper(label_mapping, update=True)
box_transform, post_cache_train_transform, post_cache_val_transform =\
    get_transforms(args.augmentation)
train_dataset, val_known_dataset, val_dataset =\
    get_datasets(
        args.data_root,
        args.train_csv_path,
        args.cal_csv_path,
        n_species_cls,
        n_activity_cls,
        static_label_mapper,
        dynamic_label_mapper,
        box_transform,
        post_cache_train_transform,
        post_cache_val_transform,
        root_cache_dir=args.root_cache_dir,
        allow_write=(rank==0),
        n_known_val=args.n_known_val
    )

if args.classifier_trainer == ClassifierTrainer.end_to_end:
    classifier_trainer = EndToEndClassifierTrainer(
        backbone,
        args.lr,
        train_dataset,
        val_known_dataset,
        box_transform,
        post_cache_train_transform,
        retraining_batch_size=args.batch_size,
        train_sampler_fn=train_sampler_fn,
        root_checkpoint_dir=args.root_checkpoint_dir,
        patience=None,
        min_epochs=0,
        max_epochs=args.max_epochs,
        label_smoothing=args.label_smoothing,
        scheduler_type=args.scheduler_type,
        allow_write=(rank==0)
    )
else:
    train_feature_file = os.path.join(
        args.precomputed_feature_dir,
        'training.pth'
    )
    val_feature_file = os.path.join(
        args.precomputed_feature_dir,
        'validation.pth'
    )
    classifier_trainer = LogitLayerClassifierTrainer(
        backbone,
        args.lr,
        train_feature_file,
        val_feature_file,
        box_transform,
        post_cache_train_transform,
        device=backbone.device,
        patience=None,
        min_epochs=0,
        max_epochs=args.max_epochs,
        label_smoothing=args.label_smoothing
    )

trainer = TuplePredictorTrainer(
    train_dataset,
    val_known_dataset,
    val_dataset,
    box_transform,
    post_cache_train_transform,
    n_species_cls,
    n_activity_cls,
    dynamic_label_mapper,
    classifier_trainer
)

start_time = time.time()

species_classifier = classifier.species_classifier
activity_classifier = classifier.activity_classifier
species_calibrator = confidence_calibrator.species_calibrator
activity_calibrator = confidence_calibrator.activity_calibrator

# Retrain the backbone and classifiers
classifier_trainer.train(
    species_classifier,
    activity_classifier,
    args.root_log_dir,
    None
)

if rank == 0:
    # Refit activation statistics. Do it manually since the classifier trainer
    # might opt to skip this step. For instance, the logit layer trainer skips
    # this step since the backbone is not reset during accommodation.
    activation_stats_training_loader = DataLoader(
        val_known_dataset,
        batch_size = 32,
        shuffle = False,
        collate_fn=gen_custom_collate(),
        num_workers=2
    )
    backbone.eval()
    all_features = []
    with torch.no_grad():
        for _, _, _, _, batch in activation_stats_training_loader:
            batch = batch.to(device)
            features = activation_statistical_model.compute_features(
                backbone.module,
                batch
            )
            all_features.append(features)
    all_features = torch.cat(all_features, dim=0)
    activation_statistical_model.fit(all_features)

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

    tuple_prediction_state_dicts = {}
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
