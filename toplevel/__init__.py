import itertools
import pathlib
import numpy as np
import torch
from torch.utils.data import DataLoader
from data.data_factory import DataFactory
from utils import custom_collate
from ensemble import Ensemble
from toplevel.util import *
import noveltydetectionfeatures
import pandas as pd
from noveltydetection.utils import compute_probability_novelty
from adaptation.query_formulation import select_queries
import pickle


class TopLevelApp:
    def __init__(self, ensemble_path, data_root, pretrained_unsupervised_module_path, th, feedback_enabled, given_detection):

        if not pathlib.Path(ensemble_path).exists():
            raise Exception(f'pretrained SCG models were not found in path {ensemble_path}')

        self.data_root = data_root
        self.NUM_SUBJECT_CLASSES = 5
        self.NUM_OBJECT_CLASSES = 12
        self.NUM_VERB_CLASSES = 8

        self.NUM_APP_FEATURES = 256 * 7 * 7
        self.NUM_VERB_FEATURES = self.NUM_APP_FEATURES + 2 * 36
        self.post_red = False
        self.p_type_dist = torch.tensor([0.25, 0.25, 0.25, 0.25])

        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
        self.all_cases = torch.tensor([])
        self.all_p_ni = torch.tensor([])
        self.all_nc = torch.tensor([])
        self.all_top_1_svo = []
        self.all_top_3_svo = []
        self.all_p_type = torch.tensor([])

        self.scg_ensemble = Ensemble(ensemble_path, self.NUM_OBJECT_CLASSES, self.NUM_SUBJECT_CLASSES, 
            self.NUM_VERB_CLASSES, data_root=None, cal_csv_path=None, val_csv_path=None)

        self.und_manager = UnsupervisedNoveltyDetectionManager(pretrained_unsupervised_module_path, 
            self.NUM_SUBJECT_CLASSES, 
            self.NUM_OBJECT_CLASSES, 
            self.NUM_VERB_CLASSES, 
            self.NUM_APP_FEATURES, 
            self.NUM_VERB_FEATURES)

        self.snd_manager = SupervisedNoveltyDetectionManager(self.NUM_APP_FEATURES, self.NUM_VERB_FEATURES)

        self.red_button_th = 0.6870563
        self.p_type_th = 0.75
        self.post_red_base = None
        self.batch_context = BatchContext()

        self.unsupervised_aucs = [None, None, None]
        self.supervised_aucs = [None, None, None]
        
        self.curr_test_id = None
        self.curr_round_id = None
        self.hint_round = None
        self.hint_img_path = None

        self.feedback_enabled = feedback_enabled
        self.given_detection = given_detection
        
        self.pre_red_transform = np.array([0.5 / self.red_button_th, 0])
        
        a = np.array([[self.red_button_th, 1], [1, 1]])
        b = np.array([0.5, 1])
        x = np.linalg.solve(a, b)
        self.post_red_transform = np.array([x[0], x[1]])
    
    def reset(self):
        self.post_red = False
        self.p_type_dist = torch.tensor([0.25, 0.25, 0.25, 0.25])
        self.unsupervised_aucs = [None, None, None]
        self.supervised_aucs = [None, None, None]
        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
        self.all_cases = torch.tensor([])
        self.all_p_ni = torch.tensor([])
        self.all_nc = torch.tensor([], dtype=torch.long)
        self.all_top_1_svo = []
        self.all_top_3_svo = []
        self.all_p_type = torch.tensor([])
        self.post_red_base = None
        
        self.snd_manager.reset()
        
        self.curr_test_id = None
        self.curr_round_id = None
        self.hint_round = None
        self.hint_img_path = None

    def run(self, csv_path, test_id, round_id, img_paths):
        if csv_path is None:
            raise Exception('path to csv was None')

        self.curr_test_id = test_id
        self.curr_round_id = round_id

        self.batch_context.reset()

        # Initialize data loaders
        scg_data_loader, novelty_dataset, N, image_paths, batch_cases = self._load_data(csv_path)

        # Compute top-3 SVOs from SCG ensemble
        scg_preds = self.scg_ensemble.get_top3_SVOs(scg_data_loader, False)

        # Unsupervised novelty scores
        unsupervised_results = self.und_manager.score(novelty_dataset, self.p_type_dist)

        subject_novelty_scores_u = unsupervised_results['subject_novelty_score']
        verb_novelty_scores_u = unsupervised_results['verb_novelty_score']
        object_novelty_scores_u = unsupervised_results['object_novelty_score']
        subject_novelty_scores_u = unsupervised_results['subject_novelty_score']
        verb_novelty_scores_u = unsupervised_results['verb_novelty_score']
        object_novelty_scores = unsupervised_results['object_novelty_score']

        subject_novelty_scores = subject_novelty_scores_u
        verb_novelty_scores = verb_novelty_scores_u
        object_novelty_scores = object_novelty_scores_u

        if self.post_red and self.feedback_enabled:
            subject_novelty_scores_u_copy = [s.clone() if s is not None else None for s in subject_novelty_scores_u]
            verb_novelty_scores_u_copy = [s.clone() if s is not None else None for s in verb_novelty_scores_u]
            object_novelty_scores_u_copy = [s.clone() if s is not None else None for s in object_novelty_scores_u]

            # Supervised novelty scores
            subject_novelty_scores_s, verb_novelty_scores_s, object_novelty_scores_s = self.snd_manager.score(
                novelty_dataset, subject_novelty_scores_u_copy, verb_novelty_scores_u_copy, object_novelty_scores_u_copy)

            self.supervised_aucs = self.snd_manager.get_svo_detectors_auc()
            self.unsupervised_aucs = self.und_manager.get_svo_detectors_auc()

            # pick the scores from the detector which had a higher AUC
            if self.unsupervised_aucs[0] < self.supervised_aucs[0]:
                subject_novelty_scores = subject_novelty_scores_s

            if self.unsupervised_aucs[1] < self.supervised_aucs[1]:
                verb_novelty_scores = verb_novelty_scores_s
            
            if self.unsupervised_aucs[2] < self.supervised_aucs[2]:
                object_novelty_scores = object_novelty_scores_s

        assert len(subject_novelty_scores) == len(verb_novelty_scores) == len(object_novelty_scores)

        # Compute P_ni
        case_1_lr, case_2_lr, case_3_lr = self.und_manager.get_calibrators()
        
        with torch.no_grad():
            batch_p_type, p_ni = compute_probability_novelty(subject_novelty_scores, verb_novelty_scores, object_novelty_scores, 
                case_1_lr, case_2_lr, case_3_lr)

        p_ni = p_ni.cpu().float()
        batch_p_type = batch_p_type.cpu()

        if self.hint_img_path in img_paths:
            first_novelty = img_paths.index(self.hint_img_path)
            p_ni[first_novelty] = 1.0

        # Merge top-3 SVOs
        merged = self._merge_top3_SVOs(scg_preds, unsupervised_results['top3'], p_ni)
        top_1 = [m[0][0] for m in merged]
        top_3 = [[e[0] for e in m] for m in merged]
        top_3_probs = [[e[1] for e in m] for m in merged]

        batch_preds_is_nc = torch.tensor([self._is_svo_type_4(t) for t in top_1], dtype=torch.long)
        
        if not self.post_red or not self.feedback_enabled:
            batch_query_mask = torch.zeros(N, dtype=torch.long)
            batch_feedback_mask = torch.zeros(N, dtype=torch.long)

            self._accumulate(top_1, top_3, p_ni, batch_cases, batch_preds_is_nc, batch_feedback_mask, batch_query_mask, batch_p_type)
        else:
            # decide which novelty detector to use
            self.supervised_aucs = self.snd_manager.get_svo_detectors_auc()
            self.unsupervised_aucs = self.und_manager.get_svo_detectors_auc()
                        
        self.batch_context.p_ni = p_ni
        self.batch_context.subject_novelty_scores_u = subject_novelty_scores_u
        self.batch_context.verb_novelty_scores_u = verb_novelty_scores_u
        self.batch_context.object_novelty_scores_u = object_novelty_scores_u
        self.batch_context.subject_novelty_scores_best = subject_novelty_scores
        self.batch_context.verb_novelty_scores_best = verb_novelty_scores
        self.batch_context.object_novelty_scores_best = object_novelty_scores
        self.batch_context.image_paths = image_paths
        self.batch_context.novelty_dataset = novelty_dataset
        self.batch_context.top_1 = top_1
        self.batch_context.top_3 = top_3
        self.batch_context.cases = batch_cases
        self.batch_context.preds_is_nc = batch_preds_is_nc
        self.batch_context.p_type = batch_p_type
        
        p_ni = p_ni.numpy()
        
        if not self.post_red:
            red_light_scores = self._compute_moving_avg(self.all_p_ni, N)
            first_novelty_instance_idx = None
        
            if not self.given_detection:
                EPS = 1e-3
                p_gt_th = np.nonzero([p > self.red_button_th + EPS for p in red_light_scores])[0]
                self.post_red = p_gt_th.shape[0] > 0
                first_novelty_instance_idx = p_gt_th[0] if self.post_red else red_light_scores.shape[0]
                self.post_red_base = self.all_p_ni.shape[0] - p_ni.shape[0] + first_novelty_instance_idx if self.post_red else None
            elif self.curr_round_id == self.hint_round:
                self.post_red = True
                first_novelty_instance_idx = self.hint_round
                self.post_red_base = self.all_p_ni.shape[0] - p_ni.shape[0] + first_novelty_instance_idx
            else:
                first_novelty_instance_idx = red_light_scores.shape[0]
                
            red_light_scores_pre = [self.pre_red_transform[0] * p + self.pre_red_transform[1] for p in red_light_scores[:first_novelty_instance_idx]]
            red_light_scores_post = [self.post_red_transform[0] * p + self.post_red_transform[1] for p in red_light_scores[first_novelty_instance_idx:]]
            red_light_scores = red_light_scores_pre + red_light_scores_post            
        else:
            all_p_ni = np.concatenate([self.all_p_ni.numpy(), p_ni])
            red_light_scores = self._compute_moving_avg(all_p_ni, N)
            red_light_scores = [self.post_red_transform[0] * p + self.post_red_transform[1] for p in red_light_scores]
         
        red_light_scores = np.clip(red_light_scores, 0.0, 1.0)
         
        # dictionary containing return values
        ret = {}
        ret['p_ni'] = p_ni.tolist()
        ret['red_light_score'] = red_light_scores
        ret['svo'] = top_3
        ret['svo_probs'] = top_3_probs
               
        return ret

    def select_queries(self, feedback_max_ids):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"

        query_indices = select_queries(feedback_max_ids, self.p_type_dist, self.batch_context.p_ni, 
            self.batch_context.subject_novelty_scores_best, self.batch_context.verb_novelty_scores_best, 
            self.batch_context.object_novelty_scores_best)

        if len(query_indices) > feedback_max_ids:
            raise Exception('number of queries exceeded feedback budget.')

        N = len(self.batch_context.image_paths)
        self.batch_context.query_mask = torch.zeros(N, dtype=torch.long)
        self.batch_context.query_mask[query_indices] = 1
        self.batch_context.query_indices = query_indices

        return [self.batch_context.image_paths[i] for i in query_indices]

    def feedback_callback(self, feedback_results):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"
        assert all([f in self.batch_context.image_paths for f in feedback_results.keys()]), "query/feedback mismatch."

        if len(feedback_results) != len(self.batch_context.query_indices):
            print(f"\nWARNING: query/feedback mismatch, requested feedback for {len(self.batch_context.query_indices)} images but got feedback for {len(feedback_results)} images. Recalculating query mask...\n")

            N = len(self.batch_context.image_paths)
            self.batch_context.query_mask = torch.zeros(N, dtype=torch.long)

            query_indices = [self.batch_context.image_paths.index(k) for k in feedback_results.keys()]

            self.batch_context.query_mask[query_indices] = 1
            self.batch_context.query_indices = query_indices
            
        for k, v in feedback_results.items():
            print(f'Feedback recevied for {k}: {v}')
            
        N = len(self.batch_context.image_paths)
        self.batch_context.feedback_mask = torch.zeros(N, dtype=torch.long)

        feedback = []
        for q in self.batch_context.query_indices:
            img_path = self.batch_context.image_paths[q]
            feedback.append(feedback_results[img_path])

        feedback_t = torch.tensor(feedback, dtype=torch.long)
        self.batch_context.feedback_mask[self.batch_context.query_indices] = feedback_t

        # adjust p_ni based on feedback 
        self.batch_context.p_ni[self.batch_context.query_indices] = feedback_t.float()
        
        self._accumulate(self.batch_context.top_1, self.batch_context.top_3, self.batch_context.p_ni, self.batch_context.cases, 
            self.batch_context.preds_is_nc, self.batch_context.feedback_mask, self.batch_context.query_mask, 
            self.batch_context.p_type)
        
        # if not torch.isclose(torch.max(self.p_type_dist), torch.ones(1, dtype=torch.float32))[0]:
        self._type_inference()

        # feedback interpretation and unsupervised score contexts update
        all_subject_indices, all_verb_indices, all_object_indices, subject_labels, verb_labels, object_labels = self._build_supervised_samples()

        # retrains supervised detector
        self._train_supervised_detector(all_subject_indices, all_verb_indices, all_object_indices, 
            subject_labels, verb_labels, object_labels)

    def red_light_hint_callback(self, first_novelty_img_path):
        assert self.hint_round is None, "red-light hint was previously set"
        assert self.curr_test_id is not None and self.curr_round_id is not None
        self.hint_round = self.curr_round_id + 1
        self.hint_img_path = first_novelty_img_path

    def _compute_moving_avg(self, p_n, num_elems):
        base_offset = p_n.shape[0] - num_elems
        assert base_offset >= 0, "There has to be at least 8 images in each round."
        res = np.zeros(num_elems)
        
        for i in range(base_offset, base_offset + num_elems):
            if i > 7:
                res[i - base_offset] = p_n[:i + 1][-8:].mean()
                
        return res

    def _merge_top3_SVOs(self, top3_non_novel, top3_novel, batch_p_ni):
        batch_p_ni = batch_p_ni.view(-1).numpy()
        N = len(batch_p_ni)
        top3_non_novel = [[(x[i][0], x[i][1], x[i][1] * (1 - y)) for i in range(3)] for x, y in zip(top3_non_novel, batch_p_ni)]
        top3_novel = [[(x[i][0], x[i][1], x[i][1] * y) for i in range(3)] for x, y in zip(top3_novel, batch_p_ni)]
        all_tuples = [x + y for x, y in zip(top3_non_novel, top3_novel)]
        comb_iter = [itertools.combinations(all_tuples[i], 3) for i in range(N)]
        scores = [list(map(lambda x: (x, x[0][2] + x[1][2] + x[2][2]), comb_iter[i])) for i in range(N)]
        top_3_svo = [sorted(i, key=lambda x: x[1], reverse=True)[0][0] for i in scores]
        top_3_svo = [sorted(i, key=lambda x: x[2], reverse=True) for i in top_3_svo]
        top_3_svo = [((a[0][0], a[0][2]), (a[1][0], a[1][2]), (a[2][0], a[2][2])) for a in top_3_svo]

        return top_3_svo

    def _type_inference(self):
        assert self.post_red, "type-inference shouldn't be called pre-red button"
        assert torch.numel(self.all_p_type) > 0, "no per-image p_type"
        assert self.all_p_ni.shape[0] == self.all_p_type.shape[0], "p_type/p_ni shape mismatch"
    
        filter_v = self.all_p_ni[self.post_red_base:] >= self.p_type_th
        
        if not torch.any(filter_v):
            return
        
        filtered = self.all_p_type[self.post_red_base:][filter_v]
    
        # types
        log_p_type_1 = self._infer_log_p_type(filtered[:, 0])
        log_p_type_2 = self._infer_log_p_type(filtered[:, 1])
        log_p_type_3 = self._infer_log_p_type(filtered[:, 2])
        log_p_type_4 = self._infer_log_p_type(filtered[:, 3])
    
        self.p_type_dist = torch.tensor([log_p_type_1, log_p_type_2, log_p_type_3, log_p_type_4])
        self.p_type_dist = torch.nn.functional.softmax(self.p_type_dist, dim=0).float()
        
        assert not torch.any(torch.isnan(self.p_type_dist)), "NaNs in p_type."
        assert not torch.any(torch.isinf(self.p_type_dist)), "Infs in p_type."

    def _infer_log_p_type(self, evidence):
        LARGE_NEG_CONSTANT = -50.0

        zero_indices = torch.nonzero(torch.isclose(evidence, torch.zeros_like(evidence))).view(-1)
        log_ev = torch.log(evidence)
        log_ev[zero_indices] = LARGE_NEG_CONSTANT
        evidence = torch.sum(log_ev)
        log_p_type = np.log(0.25) + evidence

        return log_p_type

    def _load_data(self, csv_path):
        valset = DataFactory(
            name="Custom", 
            data_root=self.data_root,
            csv_path=csv_path,
            training=False)

        scg_data_loader = DataLoader(
            dataset=valset,
            collate_fn=custom_collate, 
            batch_size=1,
            num_workers=1, 
            pin_memory=False,
            sampler=None)

        novelty_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = self.data_root,
            csv_path = csv_path,
            training = False,
            image_batch_size = 1,
            feature_extraction_device = 'cuda:0',
            cache_to_disk = False)

        N = len(valset)
        
        df = pd.read_csv(csv_path, index_col=0)
        image_paths = df['new_image_path'].to_list()

        cases = df.apply(lambda row : self._compute_case(row['subject_ymin'], row['subject_xmin'], 
            row['subject_ymax'], row['subject_xmax'], 
            row['object_ymin'], row['object_xmin'], row['object_ymax'], row['object_xmax']), axis = 1)

        return scg_data_loader, novelty_dataset, N, image_paths, torch.tensor(cases.to_numpy(), dtype=torch.long)

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

    def _is_svo_type_4(self, svo):
        # whether SV is novel or unknown
        if svo[0] == -1 or svo[1] == -1 or svo[0] == 0 or svo[1] == 0:
            return 0
        
        # whether SVO is a known-combination
        if svo in self.scg_ensemble.train_tuples:
            return 0

        # whether SVO is type-2 or type-5
        if (svo[0], svo[1]) in self.scg_ensemble.train_SVs:
            return 0
            
        return 1

    def _accumulate(self, top_1, top_3, p_ni, batch_cases, preds_is_nc, feedback_mask, query_mask, p_type):
        self.all_top_1_svo += top_1
        self.all_top_3_svo += top_3
        self.all_cases = torch.cat([self.all_cases, batch_cases])
        self.all_nc = torch.cat([self.all_nc, preds_is_nc])
        self.all_p_ni = torch.cat([self.all_p_ni, p_ni])
        self.all_feedback = torch.cat([self.all_feedback, feedback_mask])
        self.all_query_masks = torch.cat([self.all_query_masks, query_mask])
        self.all_p_type = torch.cat([self.all_p_type, p_type])
        
        assert len(self.all_top_1_svo) == self.all_p_ni.shape[0] == self.all_p_type.shape[0]
        assert self.all_nc.shape == self.all_p_ni.shape
        assert self.all_feedback.shape == self.all_query_masks.shape == self.all_cases.shape == self.all_p_ni.shape

    def _build_supervised_samples(self):
        query_indices = torch.tensor(self.batch_context.query_indices, dtype=torch.long)

        t_star = torch.argmax(self.p_type_dist)
        use_hard_labels = torch.isclose(self.p_type_dist[t_star], torch.ones(1, dtype=torch.float32))[0]
        t_star += 1

        batch_feedback = self.batch_context.feedback_mask[self.batch_context.query_indices]

        novel_indices_mask = (batch_feedback == 1).long()
        non_novel_indices_mask = (batch_feedback == 0).long()
        
        case_1_mask = (self.batch_context.cases[query_indices] == 1).long()
        case_2_mask = (self.batch_context.cases[query_indices] == 2).long()
        case_1_or_2_mask = case_1_mask + case_2_mask
        case_1_or_3_mask = case_1_mask + (self.batch_context.cases[query_indices] == 3).long()

        # negative samples
        negative_subject_indices = query_indices[(non_novel_indices_mask == 1) & (case_1_or_2_mask == 1)]
        subject_labels = torch.zeros(negative_subject_indices.shape[0])

        negative_verb_indices = query_indices[(non_novel_indices_mask == 1) & (case_1_or_2_mask == 1)]
        verb_labels = torch.zeros(negative_verb_indices.shape[0])

        negative_object_indices = query_indices[(non_novel_indices_mask == 1) & (case_1_or_3_mask == 1)]
        object_labels = torch.zeros(negative_object_indices.shape[0])

        # positive samples
        if use_hard_labels:
            positive_subject_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_2_mask == 1)] if t_star == 1 else \
                torch.tensor([], dtype=torch.long)
            subject_labels = torch.cat([subject_labels, torch.ones(positive_subject_indices.shape[0])])

            positive_verb_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_2_mask == 1)] if t_star == 2 else \
                torch.tensor([], dtype=torch.long)
            verb_labels = torch.cat([verb_labels, torch.ones(positive_verb_indices.shape[0])])

            positive_object_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_3_mask == 1)] if t_star == 3 else \
                torch.tensor([], dtype=torch.long)
            object_labels = torch.cat([object_labels, torch.ones(positive_object_indices.shape[0])])
        else:
            positive_subject_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_2_mask == 1)]
            subject_labels = torch.cat([subject_labels, torch.ones(positive_subject_indices.shape[0]) * self.p_type_dist[0]])

            positive_verb_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_2_mask == 1)]
            verb_labels = torch.cat([verb_labels, torch.ones(positive_verb_indices.shape[0]) * self.p_type_dist[1]])

            positive_object_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_3_mask == 1)]
            object_labels = torch.cat([object_labels, torch.ones(positive_object_indices.shape[0]) * self.p_type_dist[2]])
            
        all_subject_indices = torch.cat([negative_subject_indices, positive_subject_indices])
        all_verb_indices = torch.cat([negative_verb_indices, positive_verb_indices])
        all_object_indices = torch.cat([negative_object_indices, positive_object_indices])

        subject_labels = 1.0 - subject_labels
        verb_labels = 1.0 - verb_labels
        object_labels = 1.0 - object_labels

        return all_subject_indices, all_verb_indices, all_object_indices, subject_labels, verb_labels, object_labels

    def _train_supervised_detector(self, all_subject_indices, all_verb_indices, all_object_indices,
        subject_labels, verb_labels, object_labels):

        assert self.batch_context.is_set(), "no batch context"

        appearance_features = [self.batch_context.novelty_dataset.subject_appearance_features[i].view(-1) for i in all_subject_indices]
        subject_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])

        spatial_features = [self.batch_context.novelty_dataset.spatial_features[i].view(-1) for i in all_verb_indices]
        spatial_features = torch.vstack(spatial_features) if len(spatial_features) > 1 else torch.tensor([])
        appearance_features = [self.batch_context.novelty_dataset.verb_appearance_features[i].view(-1) for i in all_verb_indices]
        appearance_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        verb_features = torch.hstack([spatial_features, appearance_features])

        appearance_features = [self.batch_context.novelty_dataset.object_appearance_features[i].view(-1) for i in all_object_indices]
        object_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])

        subject_novelty_scores_u = torch.tensor([self.batch_context.subject_novelty_scores_u[i] for i in all_subject_indices.tolist()])
        verb_novelty_scores_u = torch.tensor([self.batch_context.verb_novelty_scores_u[i] for i in all_verb_indices.tolist()])
        object_novelty_scores_u = torch.tensor([self.batch_context.object_novelty_scores_u[i] for i in all_object_indices.tolist()])

        assert subject_features.shape[0] == subject_labels.shape[0]
        assert verb_features.shape[0] == verb_labels.shape[0]
        assert object_features.shape[0] == object_labels.shape[0]

        # update supervised novelty dataset
        self.snd_manager.feedback_callback(subject_features, verb_features, object_features, 
            subject_novelty_scores_u, verb_novelty_scores_u, object_novelty_scores_u, 
            subject_labels, verb_labels, object_labels)

        # train supervised novelty detector for the whole data
        self.snd_manager.train()        