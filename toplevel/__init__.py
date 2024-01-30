from enum import Enum
import json
import re
import itertools
from pathlib import Path
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from data.data_factory import DataFactory
from toplevel.util import *
from boximagedataset import BoxImageDataset
import pandas as pd
from tupleprediction import compute_probability_novelty
from adaptation.query_formulation import select_queries
import pickle
from scipy.stats import ks_2samp
from copy import deepcopy

import os

from backbone import Backbone
from sidetuningbackbone import SideTuningBackbone
from labelmapping import LabelMapper
from tupleprediction.training import\
    Augmentation,\
    SchedulerType,\
    get_transforms,\
    get_datasets,\
    TuplePredictorTrainer,\
    LogitLayerClassifierTrainer,\
    SideTuningClassifierTrainer,\
    EndToEndClassifierTrainer, \
    TransformingBoxImageDataset, \
    EWCClassifierTrainer
from data.custom import build_species_label_mapping

from taming_transformers.cycleGAN import CycleGAN

def gen_retrain_fn(device_id, train_sampler_fn, feedback_batch_sampler_fn, allow_write, allow_print, distributed=False, root_log_dir= './logs'):
    device = f'cuda:{device_id}' if device_id is not None else 'cpu'
    device_ids = [device_id] if device_id is not None else None
    def retrain(
            tuple_predictor_trainer,
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model,
            scorer,
            root_log_dir,
            model_unwrap_fn):
        # Move everything to target device
        backbone = backbone.to(device)
        classifier = classifier.to(device)
        confidence_calibrator = confidence_calibrator.to(device)
        novelty_type_classifier = novelty_type_classifier.to(device)
        scorer = scorer.to(device)

        if distributed:
            # Wrap backbone and classifiers in DDPs
            backbone = DDP(backbone, device_ids=device_ids, broadcast_buffers=False)
            classifier.ddp(device_ids=device_ids, broadcast_buffers=False)

        tuple_predictor_trainer.train_novelty_detection_module(
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model,
            scorer,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            root_log_dir=root_log_dir,
            model_unwrap_fn=model_unwrap_fn
        )

    return retrain

class TopLevelApp:
    class ClassifierTrainer(Enum):
        logit_layer = 'logit-layer'
        side_tuning = 'side-tuning'
        end_to_end = 'end-to-end'
        ewc_train = 'ewc-train'

        def __str__(self):
            return self.value

    def __init__(self, data_root, pretrained_models_dir, backbone_architecture,
            feedback_enabled, given_detection, log, log_dir, ignore_verb_novelty, train_csv_path, val_csv_path,
            trial_size, trial_batch_size, disable_retraining,
            root_cache_dir, n_known_val, classifier_trainer, precomputed_feature_dir, retraining_augmentation, retraining_lr, retraining_batch_size, retraining_val_interval, retraining_patience, retraining_min_epochs, retraining_max_epochs,
            retraining_label_smoothing, retraining_scheduler_type, feedback_loss_weight, retraining_loss_fn, class_frequency_file, gan_augment, device, retrain_fn, val_reduce_fn, model_unwrap_fn):

        pretrained_backbone_path = os.path.join(
            pretrained_models_dir,
            backbone_architecture.value['name'],
            'backbone.pth'
        )
        if not Path(pretrained_backbone_path).exists():
            raise Exception(f'pretrained backbone was not found in path {pretrained_backbone_path}')
        # if not Path(train_csv_path).exists():
        #     raise Exception(f'training CSV was not found in path {train_csv_path}')
        # if not Path(val_csv_path).exists():
        #     raise Exception(f'validation CSV was not found in path {val_csv_path}')
        # import ipdb; ipdb.set_trace()
        self.gan_augment = gan_augment
        if self.gan_augment:
            self.cycleGAN = CycleGAN('./taming_transformers/logs/vqgan_imagenet_f16_1024/configs/model.yaml','./taming_transformers/logs/vqgan_imagenet_f16_1024/checkpoints/last.ckpt')
        self.retraining_buffer = pd.DataFrame(columns=['image_path','filename','width','height','agent1_name','agent1_id'
                                                        ,'agent1_count','agent2_name','agent2_id','agent2_count','agent3_name',
                                                        'agent3_id','agent3_count','activities','activities_id','environment',
                                                        'novelty_type','master_id','novel'
                                                    ])
        self.retrain_num = 1
        self.device = device
        self.data_root = data_root
        self.pretrained_models_dir = pretrained_models_dir
        self.backbone_architecture = backbone_architecture
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.pretrained_backbone_path = pretrained_backbone_path
        self.n_species_cls = 31
        self.n_activity_cls = 7
        self.n_known_species_cls = 10
        self.n_known_activity_cls = 2
        self.post_red = False
        self.all_p_ni = torch.tensor([])
        self.all_p_ni_raw = np.array([])
        self.all_queries = torch.tensor([])
        self.all_nc = torch.tensor([])
        self.all_predictions = []
        self.all_p_type = torch.tensor([])
        self.all_red_light_scores = np.array([])
        self.p_type_override = None
        self.p_type_override_mask = None

        self.p_type_th = 0.75
        self.post_red_base = None
        self.batch_context = BatchContext()
        self.feedback_enabled = feedback_enabled
        self.p_val_cuttoff =  0.00221498
        self.windows_size = 40
                
        a = np.array([[self.p_val_cuttoff, 1], [1, 1]])
        b = np.array([0.5, 0])
        x = np.linalg.solve(a, b)
        self.pre_red_transform = np.array([x[0], x[1]])        
        
        a = np.array([[self.p_val_cuttoff, 1], [0, 1]])
        b = np.array([0.5, 1])
        x = np.linalg.solve(a, b)
        self.post_red_transform = np.array([x[0], x[1]])        
        
        self.log = log
        self.log_dir = log_dir
        self.characterization_preds = []
        self.per_image_p_type = torch.tensor([])
        self.ignore_verb_novelty = ignore_verb_novelty
        self.given_detection = given_detection
        self.red_light_img = None
        self.red_light_this_batch = False
        self.num_retrains_so_far = 0
        self.temp_path = Path(self.data_root+'/temp/')
        self.t_tn = None
        self.batch_num = 0 
        self.trial_size = trial_size
        self.trial_batch_size = trial_batch_size
        self.second_retrain_batch_num = (self.trial_size - 1000) // self.trial_batch_size
        self.disable_retraining = disable_retraining
        # Auxiliary debugging data
        self._classifier_debugging_data = {}

        self.root_cache_dir = root_cache_dir
        self.n_known_val = n_known_val
        self.classifier_trainer_enum = classifier_trainer
        self.precomputed_feature_dir = precomputed_feature_dir
        self.retraining_augmentation = retraining_augmentation
        self.retraining_lr = retraining_lr
        self.retraining_batch_size = retraining_batch_size
        self.retraining_val_interval = retraining_val_interval
        self.retraining_patience = retraining_patience
        self.retraining_min_epochs = retraining_min_epochs
        self.retraining_max_epochs = retraining_max_epochs
        self.retraining_label_smoothing = retraining_label_smoothing
        self.retraining_scheduler_type = retraining_scheduler_type
        self.feedback_loss_weight = feedback_loss_weight
        self.retraining_loss_fn = retraining_loss_fn
        self.class_frequencies = None
        if class_frequency_file is not None:
            self.class_frequencies = torch.load(class_frequency_file)

        self.backbone = Backbone(
            backbone_architecture,
            pretrained=False
        ).to(self.device)
        backbone_state_dict = torch.load(
            self.pretrained_backbone_path,
            map_location=self.device
        )
        backbone_state_dict = {
            re.sub('^module\.', '', k): v for\
                k, v in backbone_state_dict.items()
        }
        self.backbone.load_state_dict(backbone_state_dict)
        self.backbone.eval()

        self.und_manager = UnsupervisedNoveltyDetectionManager(
            self.pretrained_models_dir,
            self.backbone_architecture,
            self.n_species_cls,
            self.n_activity_cls,
            self.n_known_species_cls,
            self.n_known_activity_cls,
            self.device
        )

        self.box_transform,\
            post_cache_train_transform,\
            self.post_cache_val_transform =\
                get_transforms(self.retraining_augmentation)

        label_mapping = build_species_label_mapping(self.train_csv_path)
        self.static_label_mapper = LabelMapper(deepcopy(label_mapping), update=False)
        self.dynamic_label_mapper = LabelMapper(label_mapping, update=True)
        train_dataset,\
            val_known_dataset,\
            val_dataset =\
                get_datasets(
                    self.data_root,
                    self.train_csv_path,
                    self.val_csv_path,
                    self.n_species_cls,
                    self.n_activity_cls,
                    self.static_label_mapper,
                    self.dynamic_label_mapper,
                    self.box_transform,
                    post_cache_train_transform,
                    self.post_cache_val_transform,
                    root_cache_dir=self.root_cache_dir,
                    n_known_val=self.n_known_val
                )

        train_feature_file = os.path.join(
            self.precomputed_feature_dir,
            'training.pth'
        )
        val_feature_file = os.path.join(
            self.precomputed_feature_dir,
            'validation.pth'
        )
        self._val_reduce_fn = val_reduce_fn
        if self.classifier_trainer_enum == self.ClassifierTrainer.logit_layer:
            classifier_trainer = LogitLayerClassifierTrainer(
                self.retraining_lr,
                train_feature_file,
                val_feature_file,
                self.box_transform,
                post_cache_train_transform,
                feedback_batch_size=self.retraining_batch_size,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies
            )
        elif self.classifier_trainer_enum == self.ClassifierTrainer.side_tuning:
            self.backbone = SideTuningBackbone(self.backbone)
            classifier_trainer = SideTuningClassifierTrainer(
                self.retraining_lr,
                train_dataset,
                val_known_dataset,
                train_feature_file,
                val_feature_file,
                self.box_transform,
                post_cache_train_transform,
                feedback_batch_size=self.retraining_batch_size,
                retraining_batch_size=self.retraining_batch_size,
                val_interval=self.retraining_val_interval,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies
            )
        elif self.classifier_trainer_enum == self.ClassifierTrainer.end_to_end:
            classifier_trainer = EndToEndClassifierTrainer(
                self.retraining_lr,
                train_dataset,
                val_known_dataset,
                self.box_transform,
                post_cache_train_transform,
                retraining_batch_size=self.retraining_batch_size,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies,
                memory_cache=False,
                val_reduce_fn=self._val_reduce_fn
            )
        elif self.classifier_trainer_enum == self.ClassifierTrainer.ewc_train:
            classifier_trainer = EWCClassifierTrainer(
                self.retraining_lr,
                train_dataset,
                val_known_dataset,
                self.box_transform,
                post_cache_train_transform,
                retraining_batch_size=self.retraining_batch_size,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies,
                memory_cache=False,
                val_reduce_fn=self._val_reduce_fn
            )

        self.novelty_trainer = TuplePredictorTrainer(
            train_dataset,
            val_known_dataset,
            val_dataset,
            self.box_transform,
            post_cache_train_transform,
            self.n_species_cls,
            self.n_activity_cls,
            self.dynamic_label_mapper,
            classifier_trainer
        )

        self.retrain_fn = retrain_fn

        self.model_unwrap_fn = model_unwrap_fn
        
    def reset(self):
        self.post_red = False
        self.unsupervised_aucs = [None, None, None]
        self.supervised_aucs = [None, None, None]
        self.all_p_ni = torch.tensor([])
        self.all_queries = torch.tensor([])
        self.all_p_ni_raw = np.array([])
        self.all_nc = torch.tensor([], dtype=torch.long)
        self.all_predictions = []
        self.all_p_type = torch.tensor([])
        self.p_type_override = None
        self.p_type_override_mask = None
        self.post_red_base = None
        self.characterization_preds = []
        self.per_image_p_type = torch.tensor([])
        self.all_red_light_scores = np.array([])
        self.red_light_img = None
        self.red_light_this_batch = False
        self.t_tn = None
        self.batch_num = 0 
        self.num_retrains_so_far = 0
        self.retraining_buffer = self.retraining_buffer.iloc[0:0] 
        self.oracle_training = False 

        # Auxiliary debugging data
        self._classifier_debugging_data = {}
              
        self._reset_backbone_and_detectors()

    def process_batch(self, csv_path, test_id, round_id, img_paths, hint_typeA_data, hint_typeB_data):
        if csv_path is None:
            raise Exception('path to csv was None')


        path_to_all_data = f'/nfs/hpc/share/sail_on3/final/test_trials/api_tests/OND/image_classification/{test_id}_single_df.csv'
        csv_pd = pd.read_csv(path_to_all_data).iloc[round_id*10:round_id*10+10]

        # Check if 'novel' column exists and is numeric
        if 'novel' in csv_pd.columns and pd.api.types.is_numeric_dtype(csv_pd['novel']):
            sorted_csv = csv_pd.sort_values(by='novel', ascending=False)
            self.round_paths_sorted = sorted_csv['image_path'].head(10).tolist()
            # print(sorted_csv)
        else:
            print("Error: 'novel' column is missing or not numeric")

        self.batch_num += 1
        self.batch_context.reset()
        # initialize data loaders
        novelty_dataset, N, image_paths, bboxes, df = self._load_data(csv_path)

        # unsupervised novelty scores
        unsupervised_results = self.und_manager.score(self.backbone, novelty_dataset)

        scores = unsupervised_results['scores']
        species_probs = unsupervised_results['species_probs']
        activity_probs = unsupervised_results['activity_probs']
        assert len(scores) == len(species_probs) == len(activity_probs)

        # Get box counts from probs
        box_counts = [(x.shape[0] if x is not None else 0) for x in species_probs]
        
        # Construct dictionary containing debugging data for classifier,
        # including the image IDs and their responective species_probs and
        # activity_probs tensors
        assert len(image_paths) == len(species_probs)
        assert len(image_paths) == len(activity_probs)
        mapped_permutation =\
            self.dynamic_label_mapper.map_range(self.n_species_cls)
        for img_path, img_species_probs, img_activity_probs in\
                zip(image_paths, species_probs, activity_probs):
            mapped_img_species_probs = img_species_probs[:, mapped_permutation] if img_species_probs is not None else None
            self._classifier_debugging_data[img_path] = {
                'species_probs': mapped_img_species_probs.detach().cpu().numpy() if mapped_img_species_probs is not None else None,
                'activity_probs': img_activity_probs.detach().cpu().numpy() if img_activity_probs is not None else None
            }

        # compute P_n
        with torch.no_grad():
            # If we haven't received hint type A, but we have received a novel
            # feedback instance so that we know the trial novelty type anyway,
            # substitute it for the hint.
            if self.p_type_override is not None and hint_typeA_data is None:
                hint_a = self.p_type_override
            else:
                hint_a = hint_typeA_data

            batch_p_type, p_ni = compute_probability_novelty(
                scores,
                box_counts,
                self.und_manager.novelty_type_classifier,
                self.device,
                hint_a=hint_a,
                hint_b=hint_typeB_data
            )
        
        p_ni = p_ni.cpu().double()
        self.per_image_p_type = torch.cat([self.per_image_p_type, batch_p_type.cpu()])

        # Update batch p type based on given detection hints
        if self.given_detection and not self.post_red:
            if self.red_light_this_batch:
                # Given detection, red light hint occurs during this batch.
                # Use type 0 predictions for everything prior to the red
                # button
                idx = -1
                try:
                   idx = img_paths.index(self.red_light_img)
                except ValueError:
                    print(('specified red-light image was not found in the '
                           'given image paths.'))
                    raise Exception()
                updated_batch_p_type = batch_p_type[:]
                updated_batch_p_type[:idx, 0] = 1.0
                updated_batch_p_type[:idx, 1:] = 0.0
            else:
                # Given detection, no red button yet. Use type = predictions
                # everything this batch
                updated_batch_p_type = torch.zeros_like(batch_p_type)
                updated_batch_p_type[:, 0] = 1.0
        else:
            updated_batch_p_type = batch_p_type

        # Make predictions with updated batch p type
        with torch.no_grad():
            predictions = self._predict(
                species_probs,
                activity_probs,
                updated_batch_p_type
            )
            
            # Update / reorder species predictions using inverse dynamic label
            # mapping
            mapped_permutation =\
                self.dynamic_label_mapper.map_range(self.n_species_cls)
            mapped_predictions = []
            for s_c, s_p, a_c, a_p in predictions:
                mapped_predictions.append((
                    s_c[mapped_permutation],
                    s_p[mapped_permutation],
                    a_c,
                    a_p
                ))
            predictions = mapped_predictions

        self.batch_context.p_ni = p_ni
        self.batch_context.image_paths = image_paths
        self.batch_context.bboxes = bboxes
        self.batch_context.predictions = predictions
        self.batch_context.p_type = batch_p_type.cpu()
        self.batch_context.round_id = round_id

        red_light_scores = self._compute_red_light_scores(p_ni, N, img_paths)

        if not self.post_red or not self.feedback_enabled:
            self._accumulate(predictions, p_ni, p_ni.numpy(), batch_p_type.cpu())

        self.all_red_light_scores = np.concatenate([self.all_red_light_scores, red_light_scores])
        
        ret = {}
        ret['p_ni'] = p_ni.tolist()
        ret['red_light_score'] = red_light_scores
        ret['predictions'] = predictions
                                                                  
        return ret
        
    def red_light_hint_callback(self, path):
        self.red_light_img = path
        self.red_light_this_batch = True

    def test_completed_callback(self, test_id):
        if not self.log:
            return
        
        test_dir = Path(self.log_dir).joinpath(test_id)
        test_dir.mkdir(exist_ok=True) 
        with open(test_dir.joinpath(f'{test_id}.pkl'), 'wb') as handle:
            logs = {}
            logs['p_ni'] = self.all_p_ni.numpy()
            logs['p_ni_raw'] = self.all_p_ni_raw        
            logs['per_img_p_type'] = self.per_image_p_type.numpy()
            logs['post_red_base'] = self.post_red_base
            logs['characterization'] = self.characterization_preds
            logs['red_light_scores'] = self.all_red_light_scores
            logs['per_box_predictions'] = self._classifier_debugging_data
            logs['queries'] = self.all_queries.numpy()

            pickle.dump(logs, handle)
        return self.all_p_ni.numpy()

    def select_queries(self, feedback_max_ids):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"

        bbox_counts = [
            len(self.batch_context.bboxes[os.path.basename(img_path)])\
                for img_path in self.batch_context.image_paths
        ]
        bbox_counts = torch.tensor(
            bbox_counts,
            dtype=torch.long
        )
        if self.oracle_training:
            # selected_img_paths = self.round_paths_sorted[:feedback_max_ids]
            print("Collecting Oracle feedback")
            selected_img_paths = self.round_paths_sorted[:3] + self.round_paths_sorted[-2:]

            query_indices = []
            for path in selected_img_paths:
                try:
                    index = self.batch_context.image_paths.index(path)
                    query_indices.append(index)
                except ValueError:
                    print(f"Path {path} not found in batch context image paths.")
            selected_img_paths2 =\
                [self.batch_context.image_paths[i] for i in query_indices]
        else:
            query_indices = select_queries(
                feedback_max_ids,
                self.batch_context.p_ni,
                bbox_counts
            )
            selected_img_paths =\
                [self.batch_context.image_paths[i] for i in query_indices]
            
        bboxes = {}
        for img_path in selected_img_paths:
            img_name = os.path.basename(img_path)
            bboxes[img_name] = self.batch_context.bboxes[img_name]

        N = len(self.batch_context.image_paths)
        self.batch_context.query_mask = torch.zeros(N, dtype=torch.long)
        self.batch_context.query_mask[query_indices] = 1
        self.batch_context.query_indices = query_indices

        self.all_queries = torch.cat((self.all_queries, self.batch_context.query_mask), dim=0)

        return selected_img_paths, bboxes

    def feedback_callback(self, feedback_csv_path):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"

        p_ni_raw = np.copy(self.batch_context.p_ni)
        # Adjust batch_context.p_type, batch_context.p_ni, and characterization
        # strategy based on feedback

        df = pd.read_csv(feedback_csv_path)
        nov_types = df['novelty_type'].to_numpy()
        # import ipdb; ipdb.set_trace()
        for idx, nov_type in enumerate(nov_types):
            batch_idx = self.batch_context.query_indices[idx]

            # Shift novelty type to get index:
            # Types 0, 2, 3, 4, 5, 6 get mapped to 0, 1, 2, 3, 4, 5
            if nov_type >= 2:
                nov_type -= 1

            # Update p_type
            self.batch_context.p_type[batch_idx, :] = 0.0
            self.batch_context.p_type[batch_idx, nov_type] = 1.0

            # Update p_ni
            if nov_type == 0:
                self.batch_context.p_ni[batch_idx] = 0.0
            else:
                self.batch_context.p_ni[batch_idx] = 1.0

                # If this is the first novel feedback instance, initialize the
                # override and update the rest of the batch's p_type and p_ni
                # values to match
                if self.p_type_override is None:
                    # Novel feedback---update characterization strategy by
                    # initializing the p_type_override and mask
                    self.p_type_override = nov_type
                    p_type_override_mask = torch.ones(6, dtype=torch.bool)
                    p_type_override_mask[0] = False
                    p_type_override_mask[nov_type] = False
                    self.p_type_override_mask = p_type_override_mask


                    # Update batch_context.p_type to match the characterization
                    # override
                    self.batch_context.p_type[:, p_type_override_mask] = 0.0
                    normalizer =\
                        self.batch_context.p_type.sum(dim=1)
                    self.batch_context.p_type = self.batch_context.p_type /\
                        normalizer[:, None]
                    # Remove NaNs by setting them to the normalized override
                    # mask compliment
                    self.batch_context.p_type[normalizer == 0] =\
                        (~p_type_override_mask[:]).to(torch.double)
                    self.batch_context.p_type = self.batch_context.p_type /\
                        self.batch_context.p_type.sum(dim=1, keepdim=True)

                    # Update batch_context.p_ni as well
                    self.batch_context.p_ni =\
                        1 - self.batch_context.p_type[:, 0]

        self._accumulate(self.batch_context.predictions, self.batch_context.p_ni, p_ni_raw,
            self.batch_context.p_type)

        self.retraining_buffer = pd.concat([self.retraining_buffer, df[df['novel'] == 1]])
        
        self.novelty_trainer.add_feedback_data(self.data_root, feedback_csv_path)

        if not self.disable_retraining:
            self._retrain_supervised_detectors(feedback_csv_path)                

    def _predict(self, species_probs, activity_probs, batch_p_type):
        return self.und_manager.predict(species_probs, activity_probs, batch_p_type)

    def _compute_red_light_scores(self, p_ni, N, img_paths):
        red_light_scores = np.ones(N)
        all_p_ni = np.concatenate([self.all_p_ni.numpy(), p_ni.numpy()])
        
        if not self.post_red:
            if all_p_ni.shape[0] >= 300:
                start = all_p_ni.shape[0] - N
                for i in range(max(start, 300 + self.windows_size), start + N):
                    p_val = ks_2samp(all_p_ni[:300], all_p_ni[i - self.windows_size: i], alternative='greater', method='exact')[1]
                    red_light_scores[i - start] = p_val

                if not self.given_detection:
                    EPS = 0

                    p_gt_th = np.nonzero([p < self.p_val_cuttoff - EPS for p in red_light_scores])[0]
                    self.post_red = p_gt_th.shape[0] > 0                                
                    first_novelty_instance_idx = p_gt_th[0] if self.post_red else red_light_scores.shape[0]

                    self.post_red_base = all_p_ni.shape[0] - p_ni.shape[0] + first_novelty_instance_idx if self.post_red else None
        else:
            start = all_p_ni.shape[0] - N
            
            for i in range(max(start, 300 + self.windows_size), start + N):
                p_val = ks_2samp(all_p_ni[:300], all_p_ni[i - self.windows_size: i], alternative='greater')[1]
                red_light_scores[i - start] = p_val
            
        for i in range(len(red_light_scores)):
            if red_light_scores[i] >= self.p_val_cuttoff:
                red_light_scores[i] = self.pre_red_transform[0] * red_light_scores[i] + self.pre_red_transform[1]
            else:
                red_light_scores[i] = self.post_red_transform[0] * red_light_scores[i] + self.post_red_transform[1]
         
        if self.given_detection and not self.post_red and not self.red_light_this_batch:
            red_light_scores = np.zeros_like(red_light_scores)
        
        if self.given_detection and self.red_light_this_batch:
            self.red_light_this_batch = False
            self.post_red = True
            idx = -1
            
            try:
               idx = img_paths.index(self.red_light_img)
            except ValueError:
                print(f'specified red-light image was not found in the given image paths.')
                raise Exception()

            red_light_scores[:idx] = 0
            red_light_scores[idx] = 1

            self.post_red_base = all_p_ni.shape[0] - p_ni.shape[0] + idx
            self.red_light_img = None           
            
        return red_light_scores
            

    def characterize_round_zeros(self):
        round_characterization_preds = np.zeros(6)
        round_characterization_preds[0] = 1.0
        self.characterization_preds.append(round_characterization_preds)

    def characterize_round(self, red_light_dec):
        if red_light_dec:
            self.characterize_round_zeros()
            return

        assert torch.numel(self.all_p_type) > 0, "no per-image p_type"
        assert self.all_p_ni.shape[0] == self.all_p_type.shape[0], "p_type/p_ni shape mismatch"
    
        filter_v = self.all_p_ni[self.post_red_base:] >= self.p_type_th
        if not torch.any(filter_v):
            self.characterize_round_zeros()
            return
        
        filtered = self.all_p_type[self.post_red_base:][filter_v]
    
        prior = 0.20 if not self.ignore_verb_novelty else 1/3
    
        log_p_type_1 = self._infer_log_p_type(prior, filtered[:, 1])
        log_p_type_2 = self._infer_log_p_type(prior, filtered[:, 2])
        log_p_type_3 = self._infer_log_p_type(prior, filtered[:, 3])
        log_p_type_4 = self._infer_log_p_type(prior, filtered[:, 4])
        log_p_type_5 = self._infer_log_p_type(prior, filtered[:, 5])

        round_characterization_logits = torch.tensor([log_p_type_1, log_p_type_2, log_p_type_3, log_p_type_4, log_p_type_5])
        round_characterization_preds = torch.nn.functional.softmax(round_characterization_logits, dim=0).double()

        self.characterization_preds.append(round_characterization_preds.numpy())

        assert not torch.any(torch.isnan(round_characterization_preds)), "NaNs in p_type."
        assert not torch.any(torch.isinf(round_characterization_preds)), "Infs in p_type."
            

    def _infer_log_p_type(self, prior, evidence):
        LARGE_NEG_CONSTANT = -50.0

        zero_indices = torch.nonzero(torch.isclose(evidence, torch.zeros_like(evidence))).view(-1)
        log_ev = torch.log(evidence)
        log_ev[zero_indices] = LARGE_NEG_CONSTANT
        evidence = torch.sum(log_ev)
        log_p_type = prior + evidence

        return log_p_type

    def _load_data(self, csv_path):
        novelty_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = self.data_root,
            csv_path = csv_path,
            training = False,
            n_species_cls=self.n_species_cls,
            n_activity_cls=self.n_activity_cls,
            label_mapper=self.static_label_mapper,
            box_transform=self.box_transform,
            cache_dir=None,
            image_filter=None,
            write_cache=False
        )

        # TODO Verify this is right...
        novelty_dataset = TransformingBoxImageDataset(
            novelty_dataset,
            self.post_cache_val_transform
        )

        N = len(novelty_dataset)
        
        df = pd.read_csv(csv_path, index_col=0)
        image_paths = df['image_path'].to_list()
        json_path = f'{os.path.splitext(csv_path)[0]}.json'
        with open(json_path, 'r') as f:
            bboxes = json.load(f)
        for img_path in image_paths:
            bn = os.path.basename(img_path)
            if bn not in bboxes:
                bboxes[bn] = []

        return novelty_dataset, N, image_paths, bboxes, df

    def _accumulate(self, predictions, p_ni, p_ni_raw, p_type):
        self.all_predictions += predictions
        self.all_p_ni = torch.cat([self.all_p_ni, p_ni])
        self.all_p_ni_raw = np.concatenate([self.all_p_ni_raw, p_ni_raw])
        self.all_p_type = torch.cat([self.all_p_type, p_type])
        
        assert self.all_p_ni.shape[0] == self.all_p_type.shape[0] == len(self.all_predictions)

    def _reset_backbone_and_detectors(self):
        self.backbone = Backbone(
            self.backbone_architecture,
            pretrained=False
        ).to(self.device)
        backbone_state_dict = torch.load(
            self.pretrained_backbone_path,
            map_location=self.device
        )
        backbone_state_dict = {
            re.sub('^module\.', '', k): v for\
                k, v in backbone_state_dict.items()
        }
        self.backbone.load_state_dict(backbone_state_dict)
        self.backbone.eval()

        self.und_manager = UnsupervisedNoveltyDetectionManager(
            self.pretrained_models_dir,
            self.backbone_architecture,
            self.n_species_cls,
            self.n_activity_cls,
            self.n_known_species_cls,
            self.n_known_activity_cls,
            self.device
        )

        self.box_transform,\
            post_cache_train_transform,\
            self.post_cache_val_transform =\
                get_transforms(self.retraining_augmentation)

        label_mapping = build_species_label_mapping(self.train_csv_path)
        self.static_label_mapper = LabelMapper(deepcopy(label_mapping), update=False)
        self.dynamic_label_mapper = LabelMapper(label_mapping, update=True)
        train_dataset,\
            val_known_dataset,\
            val_dataset =\
                get_datasets(
                    self.data_root,
                    self.train_csv_path,
                    self.val_csv_path,
                    self.n_species_cls,
                    self.n_activity_cls,
                    self.static_label_mapper,
                    self.dynamic_label_mapper,
                    self.box_transform,
                    post_cache_train_transform,
                    self.post_cache_val_transform,
                    root_cache_dir=self.root_cache_dir,
                    n_known_val=self.n_known_val
                )

        train_feature_file = os.path.join(
            self.precomputed_feature_dir,
            'training.pth'
        )
        val_feature_file = os.path.join(
            self.precomputed_feature_dir,
            'validation.pth'
        )
        if self.classifier_trainer_enum == self.ClassifierTrainer.logit_layer:
            classifier_trainer = LogitLayerClassifierTrainer(
                self.retraining_lr,
                train_feature_file,
                val_feature_file,
                self.box_transform,
                post_cache_train_transform,
                feedback_batch_size=self.retraining_batch_size,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies
            )
        elif self.classifier_trainer_enum == self.ClassifierTrainer.side_tuning:
            self.backbone = SideTuningBackbone(self.backbone)
            classifier_trainer = SideTuningClassifierTrainer(
                self.retraining_lr,
                train_dataset,
                val_known_dataset,
                train_feature_file,
                val_feature_file,
                self.box_transform,
                post_cache_train_transform,
                feedback_batch_size=self.retraining_batch_size,
                retraining_batch_size=self.retraining_batch_size,
                val_interval=self.retraining_val_interval,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies
            )
        elif self.classifier_trainer_enum == self.ClassifierTrainer.end_to_end:
            classifier_trainer = EndToEndClassifierTrainer(
                self.retraining_lr,
                train_dataset,
                val_known_dataset,
                self.box_transform,
                post_cache_train_transform,
                retraining_batch_size=self.retraining_batch_size,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies,
                memory_cache=False,
                val_reduce_fn=self._val_reduce_fn
            )
        elif self.classifier_trainer_enum == self.ClassifierTrainer.ewc_train:
            classifier_trainer = EWCClassifierTrainer(
                self.retraining_lr,
                train_dataset,
                val_known_dataset,
                self.box_transform,
                post_cache_train_transform,
                retraining_batch_size=self.retraining_batch_size,
                patience=self.retraining_patience,
                min_epochs=self.retraining_min_epochs,
                max_epochs=self.retraining_max_epochs,
                label_smoothing=self.retraining_label_smoothing,
                feedback_loss_weight=self.feedback_loss_weight,
                loss_fn=self.retraining_loss_fn,
                class_frequencies=self.class_frequencies,
                memory_cache=False,
                val_reduce_fn=self._val_reduce_fn
            )
        self.novelty_trainer = TuplePredictorTrainer(
            train_dataset,
            val_known_dataset,
            val_dataset,
            self.box_transform,
            post_cache_train_transform,
            self.n_species_cls,
            self.n_activity_cls,
            self.dynamic_label_mapper,
            classifier_trainer
        )

    def _retrain_supervised_detectors(self, feedback_csv_path):
        # retrain_cond_1 = self.num_retrains_so_far == 0 and self.novelty_trainer.n_feedback_examples() >= 15
        # retrain_cond_2 = (self.batch_num == self.second_retrain_batch_num) and (self.novelty_trainer.n_feedback_examples() > 0)
        retrain_cond_1 = self.num_retrains_so_far == 0 and len(self.retraining_buffer) >= 15
        retrain_cond_2 = (self.batch_num == self.second_retrain_batch_num) 
        # if retrain_cond_1 or retrain_cond_2:
        if retrain_cond_2:
            csv_path = self.temp_path.joinpath(f'{os.getpid()}_batch_{self.batch_context.round_id}_retrain.csv')
            self.retraining_buffer.to_csv(csv_path, index=True)
            csv_path_temp = os.path.join(self.temp_path, f'{os.getpid()}_batch_{self.batch_context.round_id}_retrain.csv')
            
            self.gan_augment = False
            if self.gan_augment == True:
                box_dict = {}
                self.cycleGAN.load_datasets(self.data_root, self.train_csv_path, csv_path_temp, 4) 
                self.cycleGAN.train(400)
                self.novelty_trainer.add_feedback_data(self.data_root, csv_path_temp)   

                # box_dict = {}
                # csv_pd = pd.read_csv(csv_path_temp, na_values=[''])
                # # with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/valid.json', 'r') as f:
                # with open('/nfs/hpc/share/sail_on3/final/test.json', 'r') as f:
                #     box_dict_valid = json.load(f)
                # for index, row in csv_pd.iterrows():
                #     if not box_dict_valid.get(row['filename']):
                #         print(row['filename']," Not in Json file")
                #         csv_pd = csv_pd.drop(index)
                #     else:
                #         box_dict[row['filename']] = box_dict_valid[row['filename']]
                # csv_pd.to_csv(csv_path_temp, index=True)

                # with open(csv_path_temp[:-4]+'.json', 'w') as file:
                #     # Write the dictionary to the file as json
                #     json.dump(box_dict, file)


                # self.cycleGAN.delete_models()
            # else:
            #     # self.novelty_trainer.add_feedback_data(self.data_root, feedback_csv_path)   

            #     # box_dict = {}
            #     # csv_pd = pd.read_csv(csv_path_temp, na_values=[''])
            #     # # with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/valid.json', 'r') as f:
            #     # with open('/nfs/hpc/share/sail_on3/final/test.json', 'r') as f:
            #     #     box_dict_valid = json.load(f)
            #     # for index, row in csv_pd.iterrows():
            #     #     if not box_dict_valid.get(row['filename']):
            #     #         print(row['filename']," Not in Json file")
            #     #         csv_pd = csv_pd.drop(index)
            #     #     else:
            #     #         box_dict[row['filename']] = box_dict_valid[row['filename']]
                
                # box_dict = {}
            #     if self.retrain_num == 1:
            #         csv_path = '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_fog.csv'
            #         csv_pd = pd.read_csv(csv_path, na_values=['']).head(1000)
            #         with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_fog.json', 'r') as f:
            #             box_dict_valid = json.load(f)
            #     if self.retrain_num == 2:
            #         csv_path = '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_fog.csv'
            #         csv_pd = pd.read_csv(csv_path, na_values=['']).tail(1000)
            #         with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_fog.json', 'r') as f:
            #             box_dict_valid = json.load(f)
                
                # if self.retrain_num == 1:
                #     csv_path = '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_snow.csv'
                #     csv_pd = pd.read_csv(csv_path, na_values=['']).head(1000)
                #     with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_snow.json', 'r') as f:
                #         box_dict_valid = json.load(f)
            #     if self.retrain_num == 4:
            #         csv_path = '/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_snow.csv'
            #         csv_pd = pd.read_csv(csv_path, na_values=['']).tail(1000)
            #         with open('/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train_corrupt_snow.json', 'r') as f:
            #             box_dict_valid = json.load(f)
                
                # for index, row in csv_pd.iterrows():
                #     if not box_dict_valid.get(row['filename']):
                #         print(row['filename']," Not in Json file")
                #         csv_pd = csv_pd.drop(index)
                #     else:
                #         box_dict[row['filename']] = box_dict_valid[row['filename']]
                # csv_pd.to_csv(csv_path_temp, index=True)

                # with open(csv_path_temp[:-4]+'.json', 'w') as file:
                #     # Write the dictionary to the file as json
                #     json.dump(box_dict, file)
                # self.retrain_num+=1
                # self.novelty_trainer.add_feedback_data(self.data_root, csv_path_temp)   

            ## self.num_retrains_so_far += 1
            self.novelty_trainer.prepare_for_retraining(
                self.backbone,
                self.und_manager.classifier, 
                self.und_manager.confidence_calibrator,
                self.und_manager.novelty_type_classifier,
                self.und_manager.activation_statistical_model
            )
            # TODO Log training by passing in desired log directory
            self.retrain_fn(
                self.novelty_trainer,
                self.backbone,
                self.und_manager.classifier, 
                self.und_manager.confidence_calibrator,
                self.und_manager.novelty_type_classifier,
                self.und_manager.activation_statistical_model,
                self.und_manager.scorer,
                self.log_dir,
                self.model_unwrap_fn,
            )

            # self.backbone.eval()
            # self.retraining_buffer = self.retraining_buffer.iloc[0:0]
            ## torch.cuda.empty_cache()
            # import gc   
            # gc.collect()
