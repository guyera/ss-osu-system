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
    def __init__(self, ensemble_path, data_root, pretrained_unsupervised_module_path, th, feedback_enabled):

        if not pathlib.Path(ensemble_path).exists():
            raise Exception(f'pretrained SCG models were not found in path {ensemble_path}')

        self.data_root = data_root
        self.NUM_SUBJECT_CLASSES = 5
        self.NUM_OBJECT_CLASSES = 12
        self.NUM_VERB_CLASSES = 8

        self.NUM_APP_FEATURES = 256 * 7 * 7
        self.NUM_VERB_FEATURES = self.NUM_APP_FEATURES + 2 * 36
        self.post_red = False
        self.p_type_dist = torch.tensor([0.0, 0.25, 0.25, 0.25, 0.25])

        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
        self.all_cases = torch.tensor([])
        self.all_p_ni = torch.tensor([])
        self.all_nc = torch.tensor([])
        self.all_s_unknown = torch.tensor([])
        self.all_v_unknown = torch.tensor([])
        self.all_o_unknown = torch.tensor([])
        self.all_top_1_svo = []
        self.all_top_3_svo = []

        self.scg_ensemble = Ensemble(ensemble_path, self.NUM_OBJECT_CLASSES, self.NUM_SUBJECT_CLASSES, 
            self.NUM_VERB_CLASSES, data_root=None, cal_csv_path=None, val_csv_path=None)

        self.und_manager = UnsupervisedNoveltyDetectionManager(pretrained_unsupervised_module_path, 
            self.NUM_SUBJECT_CLASSES, 
            self.NUM_OBJECT_CLASSES, 
            self.NUM_VERB_CLASSES, 
            self.NUM_APP_FEATURES, 
            self.NUM_VERB_FEATURES)

        self.snd_manager = SupervisedNoveltyDetectionManager(self.NUM_APP_FEATURES, self.NUM_VERB_FEATURES)

        self.threshold = th
        self.batch_context = BatchContext()

        self.unsupervised_aucs = None
        self.supervised_aucs = None
        
        self.feedback_enabled = feedback_enabled
        
    def reset(self):
        self.post_red = False
        self.p_type_dist = torch.tensor([0.0, 0.25, 0.25, 0.25, 0.25])
        self.unsupervised_aucs = None
        self.supervised_aucs = None
        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
        self.all_cases = torch.tensor([])
        self.all_p_ni = torch.tensor([])
        self.all_nc = torch.tensor([], dtype=torch.long)
        self.all_s_unknown = torch.tensor([], dtype=torch.long)
        self.all_v_unknown = torch.tensor([], dtype=torch.long)
        self.all_o_unknown = torch.tensor([], dtype=torch.long)
        self.all_top_1_svo = []
        self.all_top_3_svo = []

        self.und_manager.reset()
        self.snd_manager.reset()

    def run(self, csv_path, test_id, round_id, img_paths):
        if csv_path is None:
            raise Exception('path to csv was None')

        self.batch_context.reset()

        # Initialize data loaders
        scg_data_loader, novelty_dataset, N, image_paths, batch_cases = self._load_data(csv_path)

        # Compute top-3 SVOs from SCG ensemble
        scg_preds = self.scg_ensemble.get_top3_SVOs(scg_data_loader, False)

        # Unsupervised novelty scores
        unsupervised_results = self.und_manager.score(novelty_dataset, self.p_type_dist)

        # Compute P_ni
        subject_novelty_scores_u = unsupervised_results['subject_novelty_score']
        verb_novelty_scores_u = unsupervised_results['verb_novelty_score']
        object_novelty_scores_u = unsupervised_results['object_novelty_score']
        p_n_t4 = unsupervised_results['p_n_t4']

        subject_novelty_scores = subject_novelty_scores_u
        verb_novelty_scores = verb_novelty_scores_u
        object_novelty_scores = object_novelty_scores_u

        subject_scores_ctx, verb_scores_ctx, object_scores_ctx = self.und_manager.get_score_contexts() 

        # if self.post_red and self.feedback_enabled:
        #     subject_novelty_scores_u_copy = [s.clone() if s is not None else None for s in subject_novelty_scores_u]
        #     verb_novelty_scores_u_copy = [s.clone() if s is not None else None for s in verb_novelty_scores_u]
        #     object_novelty_scores_u_copy = [s.clone() if s is not None else None for s in object_novelty_scores_u]

        #     # Supervised novelty scores
        #     subject_novelty_scores_s, verb_novelty_scores_s, object_novelty_scores_s = self.snd_manager.score(
        #         novelty_dataset, subject_novelty_scores_u_copy, verb_novelty_scores_u_copy, object_novelty_scores_u_copy)

        #     # pick the scores from the detector which had a higher AUC
        #     if self.unsupervised_aucs[0] < self.supervised_aucs[0]:
        #         subject_novelty_scores = subject_novelty_scores_s

        #     if self.unsupervised_aucs[1] < self.supervised_aucs[1]:
        #         verb_novelty_scores = verb_novelty_scores_s
            
        #     if self.unsupervised_aucs[2] < self.supervised_aucs[2]:
        #         object_novelty_scores = object_novelty_scores_s

        assert len(subject_novelty_scores) == len(verb_novelty_scores) == len(object_novelty_scores)

        p_ni = compute_probability_novelty(subject_novelty_scores, verb_novelty_scores, object_novelty_scores, 
            p_n_t4, subject_scores_ctx, verb_scores_ctx, object_scores_ctx, self.p_type_dist.to("cuda:0"))

        p_ni = p_ni.cpu().float()

        # Merge top-3 SVOs
        merged = self._merge_top3_SVOs(scg_preds, unsupervised_results['top3'], p_ni)
        top_1 = [m[0][0] for m in merged]
        top_3 = [[e[0] for e in m] for m in merged]
        top_3_probs = [[e[1] for e in m] for m in merged]

        is_subj_novel = lambda svo: 0 if svo[0] is None else int(svo[0]) == 0
        is_verb_novel = lambda svo: 0 if svo[1] is None or svo[0] is None else int(svo[1]) == 0 and int(svo[0]) != -1
        is_obj_novel = lambda svo: 0 if svo[2] is None else int(svo[2]) == 0
        batch_preds_is_nc = torch.tensor([self._is_svo_type_4(t) for t in top_1], dtype=torch.long)
        s_unk = torch.tensor([is_subj_novel(t) for t in top_1], dtype=torch.long)
        v_unk = torch.tensor([is_verb_novel(t) for t in top_1], dtype=torch.long)
        o_unk = torch.tensor([is_obj_novel(t) for t in top_1], dtype=torch.long)
        
        # cache previous all_p_ni
        all_p_ni_np = self.all_p_ni.numpy()
        
        if not self.post_red or not self.feedback_enabled:
            batch_query_mask = torch.zeros(N, dtype=torch.long)
            batch_feedback_mask = torch.zeros(N, dtype=torch.long)

            self._accumulate(top_1, top_3, p_ni, batch_cases, batch_feedback_mask, batch_query_mask, 
                batch_preds_is_nc, s_unk, v_unk, o_unk)

            # novelty type inference
            if not torch.isclose(torch.max(self.p_type_dist), torch.ones(1))[0]:
                self._type_inference()
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
        self.batch_context.s_unknown = s_unk
        self.batch_context.v_unknown = v_unk
        self.batch_context.o_unknown = o_unk
        
        # cusum
        p_ni = p_ni.numpy()
        red_light_scores = [self._cusum(np.concatenate([all_p_ni_np, p_ni[:i]])) for i in range(1, p_ni.shape[0] + 1)]
        assert len(red_light_scores) == p_ni.shape[0]

        if not self.post_red:
            self.post_red = any([score > 0.5 for score in red_light_scores])
            
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

        query_indices = select_queries(feedback_max_ids, torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2]), self.batch_context.p_ni, 
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
            self.batch_context.feedback_mask, self.batch_context.query_mask, self.batch_context.preds_is_nc, 
            self.batch_context.s_unknown, self.batch_context.v_unknown, self.batch_context.o_unknown)
        
        if not torch.isclose(torch.max(self.p_type_dist), torch.ones(1))[0]:
            self._type_inference()

        # feedback interpretation and unsupervised score contexts update
        all_subject_indices, all_verb_indices, all_object_indices, subject_labels, verb_labels, object_labels = self._process_feedback()

        # retrains supervised detector
        # self._train_supervised_detector(all_subject_indices, all_verb_indices, all_object_indices, 
        #     subject_labels, verb_labels, object_labels)

        # if self.post_red and self.unsupervised_aucs is None:
        #     self.supervised_aucs = self.snd_manager.get_svo_detectors_auc()
        #     self.unsupervised_aucs = self.und_manager.get_svo_detectors_auc()

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
        EPSILON = 0.05
        p_not_novel = 1.0 - self.all_p_ni
        s_known = self.all_s_unknown ^ 1
        v_known = self.all_v_unknown ^ 1
        o_known = self.all_o_unknown ^ 1
        not_nc = self.all_nc ^ 1
        
        # type-0
        c1 = s_known * v_known * o_known * not_nc
        c2 = c1 ^ 1
        ev_0 = c1 + c2 * EPSILON
        p_type_0 = self._infer_log_p_type_from_evidence(ev_0)

        # type-1
        c2 = (c1 ^ 1) * self.all_s_unknown
        c3 = (c1 + c2) ^ 1
        ev_1 = c1 * p_not_novel + c2 * self.all_p_ni + c3 * EPSILON
        p_type_1 = self._infer_log_p_type_from_evidence(ev_1)
        
        # type-2
        c2 = (c1 ^ 1) * self.all_v_unknown
        c3 = (c1 + c2) ^ 1
        ev_2 = c1 * p_not_novel + c2 * self.all_p_ni + c3 * EPSILON
        p_type_2 = self._infer_log_p_type_from_evidence(ev_2)

        # type-3
        c2 = (c1 ^ 1) * self.all_o_unknown
        c3 = (c1 + c2) ^ 1
        ev_3 = c1 * p_not_novel + c2 * self.all_p_ni + c3 * EPSILON
        p_type_3 = self._infer_log_p_type_from_evidence(ev_3)

        # type-4
        c2 = (c1 ^ 1) * self.all_nc
        c3 = (c1 + c2) ^ 1
        ev_4 = c1 * p_not_novel + c2 * self.all_p_ni + c3 * EPSILON
        p_type_4 = self._infer_log_p_type_from_evidence(ev_4)

        self.p_type_dist = torch.tensor([p_type_0, p_type_1, p_type_2, p_type_3, p_type_4])
        self.p_type_dist = torch.nn.functional.softmax(self.p_type_dist, dim=0)

        assert not torch.any(torch.isnan(self.p_type_dist)), "NaNs in p_type."
        assert not torch.any(torch.isinf(self.p_type_dist)), "Infs in p_type."

    def _process_feedback(self):
        # assert self.batch_context.batch_query_indices.shape[0] == self.batch_context.batch_feedback.shape[0]
        # assert all([i < N for i in batch_query_indices]), "Batch index was out of bound."
        query_indices = torch.tensor(self.batch_context.query_indices, dtype=torch.long)

        t_star = torch.argmax(self.p_type_dist)
        use_hard_labels = torch.isclose(self.p_type_dist[t_star], torch.ones(1))[0]

        batch_feedback = self.batch_context.feedback_mask[self.batch_context.query_indices]

        novel_indices_mask = (batch_feedback == 1).long()
        non_novel_indices_mask = (batch_feedback == 0).long()
        
        nominal_indices = query_indices[non_novel_indices_mask == 1]
        novel_indices = query_indices[novel_indices_mask == 1]
        
        subject_novelty_scores_u_nominal = torch.tensor([self.batch_context.subject_novelty_scores_u[i] for i in nominal_indices])
        subject_novelty_scores_u_novel = torch.tensor([self.batch_context.subject_novelty_scores_u[i] for i in novel_indices])
        
        object_novelty_scores_u_nominal = torch.tensor([self.batch_context.object_novelty_scores_u[i] for i in nominal_indices])
        object_novelty_scores_u_novel = torch.tensor([self.batch_context.object_novelty_scores_u[i] for i in novel_indices])
        
        verb_novelty_scores_u_nominal = torch.tensor([self.batch_context.verb_novelty_scores_u[i] for i in nominal_indices])
        verb_novelty_scores_u_novel = torch.tensor([self.batch_context.verb_novelty_scores_u[i] for i in novel_indices])

        # update score contexts for unsueprvised novelty detector
        self.und_manager.feedback_callback(subject_novelty_scores_u_nominal, subject_novelty_scores_u_novel,
            verb_novelty_scores_u_nominal, verb_novelty_scores_u_novel,
            object_novelty_scores_u_nominal, object_novelty_scores_u_novel)

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

            positive_object_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_3_mask == 1)] if t_star == 3 else \
                torch.tensor([], dtype=torch.long)
            object_labels = torch.cat([object_labels, torch.ones(positive_object_indices.shape[0])])

            if t_star == 2:
                positive_verb_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_2_mask == 1)]
            elif t_star == 5:
                positive_verb_indices = query_indices[(novel_indices_mask == 1) & (case_2_mask == 1)]
            else:
                positive_verb_indices = torch.tensor([], dtype=torch.long)

            verb_labels = torch.cat([verb_labels, torch.ones(positive_verb_indices.shape[0])])
        else:
            positive_subject_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_2_mask == 1)]
            subject_labels = torch.cat([subject_labels, torch.ones(positive_subject_indices.shape[0]) * self.p_type_dist[0]])

            positive_object_indices = query_indices[(novel_indices_mask == 1) & (case_1_or_3_mask == 1)]
            object_labels = torch.cat([object_labels, torch.ones(positive_object_indices.shape[0]) * self.p_type_dist[2]])

            positive_verb_indices = query_indices[(novel_indices_mask == 1) & (case_1_mask == 1)]
            verb_labels = torch.cat([verb_labels, torch.ones(positive_verb_indices.shape[0]) * self.p_type_dist[1]])
            
            t = query_indices[(novel_indices_mask == 1) & (case_2_mask == 1)]
            positive_verb_indices = torch.cat([positive_verb_indices, t])
            verb_labels = torch.cat([verb_labels, torch.ones(t.shape[0]) * (self.p_type_dist[1] + self.p_type_dist[4])])

        all_subject_indices = torch.cat([negative_subject_indices, positive_subject_indices])
        all_verb_indices = torch.cat([negative_verb_indices, positive_verb_indices])
        all_object_indices = torch.cat([negative_object_indices, positive_object_indices])

        return all_subject_indices, all_verb_indices, all_object_indices, subject_labels, verb_labels, object_labels

    def _cusum(self, p_nis):
        t = p_nis.shape[0]

        if t < 2:
            return 0.0

        p_nis = p_nis[::-1]

        alphas = np.cumsum(p_nis, 0)[:-1]
        ms = np.arange(1, t)
        alphas /= ms

        order_statistics = np.sort(p_nis)
        ks = ((1 - alphas) * ms).astype(np.int)
        scores = [np.mean(order_statistics[k:m + 1]) for k, m in zip(ks, ms)]
        max_score = np.max(scores)

        red_light_score = 0
        if max_score < self.threshold:
            red_light_score = max_score * 0.5 / self.threshold
        else:
            a = np.array([[self.threshold, 1], [1, 1]])
            b = np.array([0.5, 1])
            x = np.linalg.solve(a, b)
            red_light_score = max_score * x[0] + x[1]
            
        return red_light_score

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

    def _accumulate(self, top_1, top_3, p_ni, batch_cases, feedback_mask, query_mask, preds_is_nc, s_unk, v_unk, o_unk):
        self.all_top_1_svo += top_1
        self.all_top_3_svo += top_3
        self.all_p_ni = torch.cat([self.all_p_ni, p_ni])
        self.all_cases = torch.cat([self.all_cases, batch_cases])
        self.all_feedback = torch.cat([self.all_feedback, feedback_mask])
        self.all_query_masks = torch.cat([self.all_query_masks, query_mask])
        self.all_nc = torch.cat([self.all_nc, preds_is_nc])
        self.all_s_unknown = torch.cat([self.all_s_unknown, s_unk])        
        self.all_v_unknown = torch.cat([self.all_v_unknown, v_unk])
        self.all_o_unknown = torch.cat([self.all_o_unknown, o_unk])

        assert self.all_p_ni.shape == self.all_cases.shape == self.all_feedback.shape == self.all_query_masks.shape == self.all_nc.shape
        assert self.all_nc.shape == self.all_s_unknown.shape == self.all_v_unknown.shape == self.all_o_unknown.shape

    def _infer_log_p_type_from_evidence(self, evidence):
        LARGE_NEG_CONSTANT = -50.0

        zero_indices = torch.nonzero(torch.isclose(evidence, torch.zeros_like(evidence))).view(-1)

        # assert all([not torch.isclose(e, torch.zeros(1)) for e in evidence]), "Zeros encountered, log(0) is undefined."
        log_ev = torch.log(evidence)
        log_ev[zero_indices] = LARGE_NEG_CONSTANT
        evidence = torch.sum(log_ev)
        p_type = np.log(0.2) + evidence

        return p_type

    def _train_supervised_detector(self, all_subject_indices, all_verb_indices, all_object_indices,
        subject_labels, verb_labels, object_labels):

        assert self.batch_context.is_set(), "no batch context"

        appearance_features = [self.batch_context.novelty_dataset.subject_appearance_features[i].view(-1) for i in all_subject_indices]
        subject_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        # assert len(appearance_features) == 0 or subject_features.shape[1] == self.NUM_APP_FEATURES

        spatial_features = [self.batch_context.novelty_dataset.spatial_features[i].view(-1) for i in all_verb_indices]
        spatial_features = torch.vstack(spatial_features) if len(spatial_features) > 1 else torch.tensor([])
        appearance_features = [self.batch_context.novelty_dataset.verb_appearance_features[i].view(-1) for i in all_verb_indices]
        appearance_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        verb_features = torch.hstack([spatial_features, appearance_features])
        # assert torch.numel(appearance_features) == 0 or verb_features.shape[1] == self.NUM_VERB_FEATURES

        appearance_features = [self.batch_context.novelty_dataset.object_appearance_features[i].view(-1) for i in all_object_indices]
        object_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        # assert len(appearance_features) == 0 or object_features.shape[1] == self.NUM_APP_FEATURES

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