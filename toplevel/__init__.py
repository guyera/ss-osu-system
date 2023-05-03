from enum import Enum
import json
import re
import itertools
from pathlib import Path
import numpy as np
import torch
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
    SideTuningClassifierTrainer
from data.custom import build_species_label_mapping

class TopLevelApp:
    class ClassifierTrainer(Enum):
        logit_layer = 'logit-layer'
        side_tuning = 'side-tuning'

        def __str__(self):
            return self.value

    def __init__(self, data_root, pretrained_models_dir, backbone_architecture,
            feedback_enabled, given_detection, log, log_dir, ignore_verb_novelty, train_csv_path, val_csv_path,
            trial_size, trial_batch_size, disable_retraining,
            root_cache_dir, n_known_val, classifier_trainer, precomputed_feature_dir, retraining_augmentation, retraining_lr, retraining_batch_size, retraining_val_interval, retraining_patience, retraining_min_epochs, retraining_max_epochs,
            retraining_label_smoothing, retraining_scheduler_type, feedback_loss_weight, retraining_loss_fn, class_frequency_file):

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
        self.p_val_cuttoff =  0.035808 #0.002215 #0.08803683 #   0.0085542  #0.01042724
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
        self.temp_path = Path('./session/temp/')
        self.t_tn = None
        self.batch_num = 0 
        self.trial_size = trial_size
        self.trial_batch_size = trial_batch_size
        self.second_retrain_batch_num = (self.trial_size - 30) // self.trial_batch_size
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
        ).to('cuda:0')
        backbone_state_dict = torch.load(
            self.pretrained_backbone_path,
            map_location='cuda:0'
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
            self.n_known_activity_cls
        )

        self.box_transform,\
            post_cache_train_transform,\
            post_cache_val_transform =\
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
                    post_cache_val_transform,
                    root_cache_dir=self.root_cache_dir,
                    allow_write=True,
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
                self.backbone,
                self.retraining_lr,
                train_feature_file,
                val_feature_file,
                self.box_transform,
                post_cache_train_transform,
                device=self.backbone.device,
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
                self.backbone,
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
        
    def reset(self):
        self.post_red = False
        self.unsupervised_aucs = [None, None, None]
        self.supervised_aucs = [None, None, None]
        self.all_p_ni = torch.tensor([])
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

        # Auxiliary debugging data
        self._classifier_debugging_data = {}
              
        self._reset_backbone_and_detectors()

    def process_batch(self, csv_path, test_id, round_id, img_paths, hint_typeA_data, hint_typeB_data):
        if csv_path is None:
            raise Exception('path to csv was None')

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
                self.backbone.device,
                hint_a=hint_a,
                hint_b=hint_typeB_data
            )
        
        p_ni = p_ni.cpu().float()
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
            dtype=torch.long,
            device=self.batch_context.p_ni.device
        )
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
                        (~p_type_override_mask[:]).to(torch.float)
                    self.batch_context.p_type = self.batch_context.p_type /\
                        self.batch_context.p_type.sum(dim=1, keepdim=True)

                    # Update batch_context.p_ni as well
                    self.batch_context.p_ni =\
                        1 - self.batch_context.p_type[:, 0]

        self._accumulate(self.batch_context.predictions, self.batch_context.p_ni, p_ni_raw,
            self.batch_context.p_type)

        self.novelty_trainer.add_feedback_data(self.data_root, feedback_csv_path)

        if not self.disable_retraining:
            self._retrain_supervised_detectors()                

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
        round_characterization_preds = torch.nn.functional.softmax(round_characterization_logits, dim=0).float()

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
            write_cache=False,
            image_filter=None
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
        ).to('cuda:0')
        backbone_state_dict = torch.load(
            self.pretrained_backbone_path,
            map_location='cuda:0'
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
            self.n_known_activity_cls
        )

        self.box_transform,\
            post_cache_train_transform,\
            post_cache_val_transform =\
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
                    post_cache_val_transform,
                    root_cache_dir=self.root_cache_dir,
                    allow_write=True,
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
                self.backbone,
                self.retraining_lr,
                train_feature_file,
                val_feature_file,
                self.box_transform,
                post_cache_train_transform,
                device=self.backbone.device,
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
                self.backbone,
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

    def _retrain_supervised_detectors(self):
        retrain_cond_1 = self.num_retrains_so_far == 0 and self.novelty_trainer.n_feedback_examples() >= 15
        retrain_cond_2 = (self.batch_num == self.second_retrain_batch_num) and (self.novelty_trainer.n_feedback_examples() > 0)

        if retrain_cond_1 or retrain_cond_2:
            self.num_retrains_so_far += 1
            self.novelty_trainer.prepare_for_retraining(
                self.und_manager.classifier, 
                self.und_manager.confidence_calibrator,
                self.und_manager.novelty_type_classifier,
                self.und_manager.activation_statistical_model
            )
            self.novelty_trainer.train_novelty_detection_module(
                self.backbone,
                self.und_manager.classifier, 
                self.und_manager.confidence_calibrator,
                self.und_manager.novelty_type_classifier,
                self.und_manager.activation_statistical_model,
                self.und_manager.scorer
            )
            self.backbone.eval()
