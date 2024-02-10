import re
import pickle
import time
import os
import argparse
from copy import deepcopy

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import boxclassifier
import tupleprediction
from tupleprediction.training import\
    Augmentation,\
    SchedulerType,\
    get_transforms,\
    get_datasets,\
    FlattenedBoxImageDataset
from backbone import Backbone
from scoring import\
    ActivationStatisticalModel,\
    make_logit_scorer,\
    CompositeScorer
from labelmapping import LabelMapper
from data.custom import build_species_label_mapping

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
    '--batch-size',
    type=int,
    default=64,
    help='Batch size for computing features'
)

parser.add_argument(
    '--n-known-val',
    type=int,
    default=4068,
    help='Number of known validation instances to pull from the training CSV'
)

parser.add_argument(
    '--pretrained-backbone-path',
    type=str,
    default='./pretrained-models/swin_t/backbone.pth',
    help='Location of file containing pretrained backbone weights'
)

parser.add_argument(
    '--root-save-dir',
    type=str,
    default='./.features',
    help='Directory in which to save feature files'
)

args = parser.parse_args()

device = 'cuda:0'

architecture = Backbone.Architecture.swin_t
backbone = Backbone(architecture, pretrained=False).to(device)
backbone_state_dict = torch.load(
    args.pretrained_backbone_path,
    map_location=device
)
backbone_state_dict = {
    re.sub('^module\.', '', k): v for\
        k, v in backbone_state_dict.items()
}
backbone.load_state_dict(backbone_state_dict)
backbone.eval()

n_known_species_cls = 10
n_species_cls = 30 # TODO Determine
n_known_activity_cls = 2
n_activity_cls = 4 # TODO Determine

label_mapping = build_species_label_mapping(args.train_csv_path)
static_label_mapper = LabelMapper(deepcopy(label_mapping), update=False)
dynamic_label_mapper = LabelMapper(label_mapping, update=True)
box_transform, post_cache_train_transform, post_cache_val_transform =\
    get_transforms(Augmentation.none)
train_dataset, val_known_dataset, val_dataset = get_datasets(
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
    n_known_val=args.n_known_val
)


start_time = time.time()

flattened_train_dataset = FlattenedBoxImageDataset(train_dataset)
train_loader = DataLoader(
    flattened_train_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2
)

# Construct validation loaders for early stopping / model selection.
# I'm assuming our model selection strategy will be based solely on the
# validation classification accuracy and not based on novelty detection
# capabilities in any way. Otherwise, we can use the novel validation
# data to measure novelty detection performance. These currently aren't
# being stored (except in a special form for the logistic regressions),
# so we'd have to modify __init__().
flattened_val_dataset = FlattenedBoxImageDataset(val_known_dataset)
val_loader = DataLoader(
    flattened_val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=2
)

save_dir = os.path.join(
    args.root_save_dir,
    box_transform.path(),
    post_cache_train_transform.path(),
)
os.makedirs(save_dir, exist_ok=True)

training_features_path = os.path.join(save_dir, 'training.pth')
validation_features_path = os.path.join(save_dir, 'validation.pth')

# Determine the device to use based on the backbone's fc weights
device = backbone.device

train_box_features = []
train_species_labels = []
train_activity_labels = []

val_box_features = []
val_species_labels = []
val_activity_labels = []

with torch.no_grad():
    progress = tqdm(train_loader)
    for species_labels, activity_labels, box_images in progress:
        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = backbone(box_images)

        # Store
        train_box_features.append(box_features)
        train_species_labels.append(species_labels)
        train_activity_labels.append(activity_labels)


    train_box_features = torch.cat(train_box_features, dim=0)
    train_species_labels = torch.cat(train_species_labels, dim=0)
    train_activity_labels = torch.cat(train_activity_labels, dim=0)

    progress = tqdm(val_loader)
    for species_labels, activity_labels, box_images in progress:
        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = backbone(box_images)

        # Store
        val_box_features.append(box_features)
        val_species_labels.append(species_labels)
        val_activity_labels.append(activity_labels)


    val_box_features = torch.cat(val_box_features, dim=0)
    val_species_labels = torch.cat(val_species_labels, dim=0)
    val_activity_labels = torch.cat(val_activity_labels, dim=0)

torch.save(
    (train_box_features, train_species_labels, train_activity_labels),
    training_features_path
)
torch.save(
    (val_box_features, val_species_labels, val_activity_labels),
    validation_features_path
)




