import re
import itertools
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.data_factory import DataFactory
from utils import custom_collate
from toplevel.util import *
from boximagedataset import BoxImageDataset
import pandas as pd
from tupleprediction import compute_probability_novelty
from adaptation.query_formulation import select_queries
import pickle
from scipy.stats import ks_2samp

import os

from backbone import Backbone
from data.custom import build_species_label_mapping
from labelmapping import LabelMapper
from tupleprediction.training import\
    Augmentation,\
    SchedulerType,\
    get_transforms,\
    get_datasets,\
    TuplePredictorTrainer,\
    LogitLayerClassifierTrainer

class TopLevelApp:
    def __init__(self, data_root, pretrained_models_dir, backbone_architecture,
            feedback_enabled, given_detection, log, log_dir, ignore_verb_novelty, train_csv_path, val_csv_path,
            trial_size, trial_batch_size, disable_retraining,
            root_cache_dir, n_known_val, precomputed_feature_dir, retraining_augmentation, retraining_lr, retraining_batch_size, retraining_patience, retraining_min_epochs, retraining_max_epochs,
            retraining_label_smoothing, retraining_scheduler_type, feedback_loss_weight):

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
        self.n_species_cls = 30
        self.n_activity_cls = 4
        self.n_known_species_cls = 10
        self.n_known_activity_cls = 2
        self.post_red = False
        self.all_p_ni = torch.tensor([])
        self.all_p_ni_raw = np.array([])
        self.all_nc = torch.tensor([])
        self.all_top_1_svo = []
        self.all_top_3_svo = []
        self.all_p_type = torch.tensor([])
        self.all_red_light_scores = np.array([])

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
        
        self.subj_novelty_scores_un = []
        self.verb_novelty_scores_un = []
        self.obj_novelty_scores_un = []
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

        self.root_cache_dir = root_cache_dir
        self.n_known_val = n_known_val
        self.precomputed_feature_dir = precomputed_feature_dir
        self.retraining_augmentation = retraining_augmentation
        self.retraining_lr = retraining_lr
        self.retraining_batch_size = retraining_batch_size
        self.retraining_patience = retraining_patience
        self.retraining_min_epochs = retraining_min_epochs
        self.retraining_max_epochs = retraining_max_epochs
        self.retraining_label_smoothing = retraining_label_smoothing
        self.retraining_scheduler_type = retraining_scheduler_type
        self.feedback_loss_weight = feedback_loss_weight

        self.mergedSVO = []
        self.mergedprobs = []

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

        label_mapping = build_species_label_mapping(self.train_csv_path)
        self.box_transform,\
            post_cache_train_transform,\
            post_cache_val_transform =\
                get_transforms(self.retraining_augmentation)

        train_dataset,\
            val_known_dataset,\
            val_dataset,\
            dynamic_label_mapper,\
            self.static_label_mapper =\
                get_datasets(
                    self.data_root,
                    self.train_csv_path,
                    self.val_csv_path,
                    self.n_species_cls,
                    self.n_activity_cls,
                    label_mapping,
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
            feedback_loss_weight=self.feedback_loss_weight
        )

        self.novelty_trainer = TuplePredictorTrainer(
            train_dataset,
            val_known_dataset,
            val_dataset,
            self.box_transform,
            post_cache_train_transform,
            self.n_species_cls,
            self.n_activity_cls,
            dynamic_label_mapper,
            classifier_trainer
        )
        
    def reset(self):
        self.post_red = False
        self.unsupervised_aucs = [None, None, None]
        self.supervised_aucs = [None, None, None]
        self.all_p_ni = torch.tensor([])
        self.all_p_ni_raw = np.array([])
        self.all_nc = torch.tensor([], dtype=torch.long)
        self.all_top_1_svo = []
        self.all_top_3_svo = []
        self.all_p_type = torch.tensor([])
        self.post_red_base = None
        self.subj_novelty_scores_un = []
        self.verb_novelty_scores_un = []
        self.obj_novelty_scores_un = []
        self.characterization_preds = []
        self.per_image_p_type = torch.tensor([])
        self.all_red_light_scores = np.array([])
        self.red_light_img = None
        self.red_light_this_batch = False
        self.t_tn = None
        self.batch_num = 0 
        self.num_retrains_so_far = 0
        self.mergedSVO = []
        self.mergedprobs = []
              
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

        scores_u = unsupervised_results['scores']
        species_probs = unsupervised_results['species_probs']
        activity_probs = unsupervised_results['activity_probs']
        assert len(scores_u) == len(species_probs_u) == len(activity_probs_u)

        self.scores_un += scores_u

        # compute P_n
        with torch.no_grad():
            batch_p_type, p_ni = compute_probability_novelty(
                scores_u,
                self.und_manager.novelty_type_classifier,
                hint_a=hint_typeA_data,
                hint_b=hint_typeB_data
            )
        
        p_ni = p_ni.cpu().float()
        batch_p_type = batch_p_type.cpu()
        self.per_image_p_type = torch.cat([self.per_image_p_type, batch_p_type])
        top3 = None
        top3_probs = None

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
        if not self.given_detection:
            predictions = self._predict(
                species_probs,
                activity_probs,
                updated_batch_p_type
            )

        self.predictions.append(predictions)

        self.batch_context.p_ni = p_ni
        self.batch_context.scores_u = scores_u
        self.batch_context.image_paths = image_paths
        self.batch_context.bboxes = bboxes
        self.batch_context.novelty_dataset = novelty_dataset
        self.batch_context.predictions = predictions
        self.batch_context.p_type = batch_p_type
        self.batch_context.round_id = round_id

        red_light_scores = self._compute_red_light_scores(p_ni, N, img_paths)

        if not self.post_red or not self.feedback_enabled:
            self._accumulate(top3, p_ni, p_ni.numpy(), batch_p_type)

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
            logs['subj_novelty_scores_un'] = [s.cpu() if s is not None else None for s in self.subj_novelty_scores_un]
            logs['verb_novelty_scores_un'] = [s.cpu() if s is not None else None for s in self.verb_novelty_scores_un]
            logs['obj_novelty_scores_un'] = [s.cpu() if s is not None else None for s in self.obj_novelty_scores_un]
            logs['p_ni'] = self.all_p_ni.numpy()
            logs['p_ni_raw'] = self.all_p_ni_raw        
            logs['per_img_p_type'] = self.per_image_p_type.numpy()
            logs['mergedSVO'] = self.mergedSVO
            logs['mergedprobs'] = self.mergedprobs
            logs['post_red_base'] = self.post_red_base
            logs['characterization'] = self.characterization_preds
            logs['red_light_scores'] = self.all_red_light_scores

            pickle.dump(logs, handle)
        return self.all_p_ni.numpy()

    def select_queries(self, feedback_max_ids):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"

        query_indices = select_queries(feedback_max_ids, torch.tensor([1/3, 1/3, 1/3, 0]), self.batch_context.p_ni, 
            self.batch_context.subject_novelty_scores_u, self.batch_context.verb_novelty_scores_u, 
            self.batch_context.object_novelty_scores_u)

        img_paths = self.batch_context.image_paths[query_indices]
        bboxes = {}
        for img_path in img_paths:
            img_name = os.path.basename(img_path)
            bboxes[img_name] = self.batch_context.bboxes[img_name]

        N = len(self.batch_context.image_paths)
        self.batch_context.query_mask = torch.zeros(N, dtype=torch.long)
        self.batch_context.query_mask[query_indices] = 1
        self.batch_context.query_indices = query_indices

        return [self.batch_context.image_paths[i] for i in query_indices], bboxes

    def feedback_callback(self, feedback_csv_path):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"

        N = len(self.batch_context.image_paths)
        feedback_t = torch.tensor(feedback, dtype=torch.long)

        p_ni_raw = np.copy(self.batch_context.p_ni)
        # TODO adjust batch_context.p_ni and batch_context.all_p_type
        # based on feedback (the characterization becomes trivial once we've
        # observed a positive feedback instance, so we want to make some
        # override characterization prediction tensor that's None until we've
        # gotten such an instance, at which point it's the proper
        # characterization tensor. Then, we should adjust how
        # characterize_round works to check for that override and use it
        # when available)

        self._accumulate(self.batch_context.top_3, self.batch_context.p_ni, p_ni_raw,
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
            if all_p_ni.shape[0] >= 60:    
                start = all_p_ni.shape[0] - N                
                for i in range(max(start, 60 + self.windows_size), start + N):
                    p_val = ks_2samp(all_p_ni[:60], all_p_ni[i - self.windows_size: i], alternative='greater', method='exact')[1]
                    red_light_scores[i - start] = p_val

                if not self.given_detection:
                    EPS = 0

                    p_gt_th = np.nonzero([p < self.p_val_cuttoff - EPS for p in red_light_scores])[0]
                    self.post_red = p_gt_th.shape[0] > 0                                
                    first_novelty_instance_idx = p_gt_th[0] if self.post_red else red_light_scores.shape[0]

                    self.post_red_base = all_p_ni.shape[0] - p_ni.shape[0] + first_novelty_instance_idx if self.post_red else None
        else:
            start = all_p_ni.shape[0] - N
            
            for i in range(max(start, 60 + self.windows_size), start + N):
                p_val = ks_2samp(all_p_ni[:60], all_p_ni[i - self.windows_size: i], alternative='greater')[1]
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
            
    def _merge_top3_SVOs(self, top3_non_novel, top3_novel, batch_p_ni):
        batch_p_ni = batch_p_ni.reshape(-1)
        N = len(batch_p_ni)
        top3_non_novel = [[(x[i][0], x[i][1], x[i][1] * (1 - y)) for i in range(3)] for x, y in zip(top3_non_novel, batch_p_ni)]
        top3_novel = [[(x[i][0], x[i][1], x[i][1] * y) for i in range(3)] for x, y in zip(top3_novel, batch_p_ni)]
        all_tuples = [x + y for x, y in zip(top3_non_novel, top3_novel)]
        comb_iter = [itertools.combinations(all_tuples[i], 3) for i in range(N)]
        scores = [list(map(lambda x: (x, x[0][2] + x[1][2] + x[2][2]), comb_iter[i])) for i in range(N)]
        top_3_svo = [sorted(i, key=lambda x: x[1], reverse=True)[0][0] for i in scores]
        top_3_svo = [sorted(i, key=lambda x: x[2], reverse=True) for i in top_3_svo]
        top_3_svo = [((a[0][0], a[0][2].item()), (a[1][0], a[1][2].item()), (a[2][0], a[2][2].item())) for a in top_3_svo]

        return top_3_svo


    def characterize_round_zeros(self):
        round_characterization_preds = np.zeros(6)
        round_characterization_preds[0] = 1.0
        self.characterization_preds.append(round_characterization_preds)

    def characterize_round(self, red_light_dec):
        if self.red_light_dec:
            self.characterize_round_zeros()
            return

        assert self.post_red, "type-inference shouldn't be called pre-red button"
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
            write_cache=False
        )

        N = len(novelty_dataset)
        
        df = pd.read_csv(csv_path, index_col=0)
        image_paths = df['image_path'].to_list()
        json_path = f'{os.path.splitext(csv_path)[0]}.json'
        with open(json_path, 'r') as f:
            bboxes = json.load(f)

        cases = df.apply(lambda row : self._compute_case(row['subject_ymin'], row['subject_xmin'], 
            row['subject_ymax'], row['subject_xmax'], 
            row['object_ymin'], row['object_xmin'], row['object_ymax'], row['object_xmax']), axis = 1)
        df['case'] = cases.to_numpy()

        return novelty_dataset, N, image_paths, bboxes, df

    def _compute_case(self, sbox_ymin, sbox_xmin, sbox_ymax, sbox_xmax, obox_ymin, obox_xmin, obox_ymax, obox_xmax):
        has_subject = sbox_ymin >= 0 and sbox_ymax >= 0 and sbox_xmin >= 0 and sbox_xmax >= 0
        has_object = obox_ymin >= 0 and obox_ymax >= 0 and obox_xmin >= 0 and obox_xmax >= 0

        if has_subject and has_object:
            return 1
        
        if has_subject and not has_object:
            return 2

        if not has_subject and has_object:
            return 3

        raise Exception(f'Invalid subject/object box coords, {has_subject}, {has_object}')

    def _accumulate(self, top_3, p_ni, p_ni_raw, p_type):
        self.all_top_3_svo += top_3
        self.all_p_ni = torch.cat([self.all_p_ni, p_ni])
        self.all_p_ni_raw = np.concatenate([self.all_p_ni_raw, p_ni_raw])
        self.all_p_type = torch.cat([self.all_p_type, p_type])
        
        assert self.all_p_ni.shape[0] == self.all_p_type.shape[0] == len(self.all_top_3_svo)

    def _reset_backbone_and_detectors(self):
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

        label_mapping = build_species_label_mapping(self.train_csv_path)
        self.box_transform,\
            post_cache_train_transform,\
            post_cache_val_transform =\
                get_transforms(self.retraining_augmentation)

        train_dataset,\
            val_known_dataset,\
            val_dataset,\
            dynamic_label_mapper,\
            self.static_label_mapper =\
                get_datasets(
                    self.data_root,
                    self.train_csv_path,
                    self.val_csv_path,
                    self.n_species_cls,
                    self.n_activity_cls,
                    label_mapping,
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
            feedback_loss_weight=self.feedback_loss_weight
        )

        self.novelty_trainer = TuplePredictorTrainer(
            train_dataset,
            val_known_dataset,
            val_dataset,
            self.box_transform,
            post_cache_train_transform,
            self.n_species_cls,
            self.n_activity_cls,
            dynamic_label_mapper,
            classifier_trainer
        )

    def _retrain_supervised_detectors(self):
        retrain_cond_1 = self.num_retrains_so_far == 0 and self.novelty_trainer.n_feedback_examples() >= 15
        retrain_cond_2 = (self.batch_num == self.second_retrain_batch_num) and (self.novelty_trainer.n_feedback_examples() > 0)
        
        if retrain_cond_1 or retrain_cond_2:
            self.num_retrains_so_far += 1
            self.novelty_trainer.prepare_for_retraining(self.backbone, self.und_manager.classifier, 
                self.und_manager.case_1_logistic_regression,
                self.und_manager.case_2_logistic_regression,
                self.und_manager.case_3_logistic_regression,
                self.und_manager.activation_statistical_model)
            self.novelty_trainer.train_novelty_detection_module(self.backbone, self.und_manager.classifier, 
                self.und_manager.case_1_logistic_regression,
                self.und_manager.case_2_logistic_regression,
                self.und_manager.case_3_logistic_regression,
                self.und_manager.activation_statistical_model)
