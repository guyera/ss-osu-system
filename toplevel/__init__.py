import itertools
import pathlib
import torch
from models.scg import SpatiallyConditionedGraph as SCG
from torch.utils.data import DataLoader
from data.data_factory import DataFactory
from utils import custom_collate
from ensemble.ensemble import Ensemble
from toplevel.util import *
import noveltydetectionfeatures
import pandas as pd
from noveltydetection.utils import compute_probability_novelty
from toplevel.cusum import SprtPredictor


class TopLevelApp:
    def __init__(self, ensemble_path, num_subject_classes, num_object_classes, num_verb_classes, 
        data_root, csv_path, pretrained_unsupervised_module_path, test_batch_size):

        if not pathlib.Path(ensemble_path).exists():
            raise Exception(f'pretrained SCG models were not found in path {ensemble_path}')

        self.data_root = data_root
        self.csv_path = csv_path
        self.num_subject_classes = num_subject_classes
        self.num_object_classes = num_object_classes
        self.num_verb_classes = num_verb_classes

        self.NUM_APP_FEATURES = 256 * 7 * 7
        self.NUM_VERB_FEATURES = self.NUM_APP_FEATURES + 2 * 36
        self.post_red = False
        self.p_type_dist = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        self.test_batch_size = test_batch_size
        # TODO adjusut
        self.query_budget = 10

        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
        self.all_cases = torch.tensor([])
        self.all_p_ni = torch.tensor([])
        self.all_nc = torch.tensor([])
        self.all_top_svo = []

        # Initializing SCG ensemble
        self.nets = []
        for p in pathlib.Path(ensemble_path).glob('*'):
            if p.suffix != '.pt':
                continue

            self.nets.append(SCG(
                num_classes=num_verb_classes,
                num_obj_classes=num_object_classes, 
                num_subject_classes=num_subject_classes,
                num_iterations=2, 
                postprocess=False,
                max_subject=15, 
                max_object=15,
                box_score_thresh=0.0,
                distributed=False))

            # Loading pretrained models
            checkpoint = torch.load(p, map_location="cpu")
            self.nets[-1].load_state_dict(checkpoint['model_state_dict'])
            self.nets[-1].cuda()
            self.nets[-1].eval()

        self.scg_ensemble = Ensemble(self.nets, 
            num_object_classes, 
            num_subject_classes, 
            num_verb_classes)

        self.und_manager = UnsupervisedNoveltyDetectionManager(pretrained_unsupervised_module_path, 
            num_subject_classes, 
            num_object_classes, 
            num_verb_classes, 
            self.NUM_APP_FEATURES, 
            self.NUM_VERB_FEATURES)

        self.snd_manger = SupervisedNoveltyDetectionManager(self.NUM_APP_FEATURES, self.NUM_VERB_FEATURES, 
            budget=self.query_budget)

        scale_factor = None
        threshold = None
        self.change_point_detector = SprtPredictor(None, None, scale_factor, threshold)

    def run(self, csv_path = None):
        if csv_path:
            self.csv_path = csv_path

        # Initializing data loaders
        # TODO change novelty_dataset to hande the case where there are no labels
        scg_data_loader, novelty_dataset = self._get_data_loaders()
        batch_cases = self._compute_batch_cases()

        # Computing top-3 SVOs from SCG ensemble
        scg_preds = self.scg_ensemble.get_top3_SVOs(scg_data_loader, False)
        # scg_preds = [[((0, 0, 0), torch.rand((1,))) for _ in range(3)] for _ in range(len(novelty_dataset))]

        # Unsupervised novelty scores
        unsupervised_results = self.und_manager.score(novelty_dataset, self.p_type_dist)

        # Computing P_ni
        subject_novelty_scores_u = unsupervised_results['subject_novelty_score']
        verb_novelty_scores_u = unsupervised_results['verb_novelty_score']
        object_novelty_scores_u = unsupervised_results['object_novelty_score']
        p_n_t4 = unsupervised_results['p_n_t4']

        subject_novelty_scores = subject_novelty_scores_u
        verb_novelty_scores = verb_novelty_scores_u
        object_novelty_scores = object_novelty_scores_u

        subject_scores_ctx, verb_scores_ctx, object_scores_ctx = self.und_manager.get_score_contexts() 

        if self.post_red:
            # Supervised novelty scores
            subject_novelty_scores_s, verb_novelty_scores_s, object_novelty_scores_s = self.snd_manger.score(
                novelty_dataset, subject_novelty_scores_u, verb_novelty_scores_u, object_novelty_scores_u)

            # pick the scores from the detector with higher AUC
            if self.unsupervised_aucs[0] < self.supervised_aucs[0]:
                subject_novelty_scores = subject_novelty_scores_s

            if self.unsupervised_aucs[1] < self.supervised_aucs[1]:
                verb_novelty_scores = verb_novelty_scores_s
            
            if self.unsupervised_aucs[2] < self.supervised_aucs[2]:
                object_novelty_scores = object_novelty_scores_s

        p_ni = compute_probability_novelty(subject_novelty_scores, verb_novelty_scores, object_novelty_scores, 
            p_n_t4, subject_scores_ctx, verb_scores_ctx, object_scores_ctx, self.p_type_dist)

        # p_ni = torch.ones(len(scg_preds)) * 0.5 

        # Merging top-3 SVOs
        merged = self.build_merged_top3_SVOs(scg_preds, unsupervised_results['top3'], p_ni)
        top_1 = [m[0][0] for m in merged]
        batch_preds_is_nc = self._is_novel_combination(top_1)

        # set the query-related masks
        batch_query_mask = torch.zeros(self.test_batch_size, dtype=torch.long)
        batch_feedback_mask = torch.zeros(self.test_batch_size, dtype=torch.long)

        if not self.post_red:
            # novelty type inference
            self.type_inference(top_1, p_ni, batch_query_mask, batch_feedback_mask, 
                batch_cases, batch_preds_is_nc)

            # cusum
            self.update_world_state()
        else:
            # post queries
            query_indices = self.snd_manger.get_query_indices(self.p_type_dist, p_ni, subject_novelty_scores, 
                verb_novelty_scores, object_novelty_scores)

            batch_query_mask[query_indices] = 1

            # TODO send queries and set feedback
            feedback = None

            feedback = feedback.long()
            batch_feedback_mask[query_indices] = feedback

            if not torch.isclose(torch.max(self.p_type_dist), torch.ones(1))[0]:
                self.type_inference(top_1, p_ni, batch_query_mask, batch_feedback_mask, batch_cases, 
                    batch_preds_is_nc)

            # processeses feedback by:
            # 1. interpreting feedback and updating supervised dataset accordingly
            # 2. updating unsupervised score contexts using the supervised scores for supervised dataset
            self.process_feedback(query_indices, feedback, novelty_dataset, subject_novelty_scores_u, 
                verb_novelty_scores_u, object_novelty_scores_u)

            # to decide which novelty detector to use
            self.supervised_aucs = self.snd_manger.get_svo_detectors_auc()
            self.unsupervised_aucs = self.und_manager.get_svo_detectors_auc()

    def build_merged_top3_SVOs(self, top3_non_novel, top3_novel, p_ni):
        """
        Merges top-3 non-novel SVOs and top-3 novel SVOs
    
        Parameters:
        -----------
        top3_non_novel:
            List of length N where each element is a list of length 3 containing tuples of following form:
                    ((S, V, O), Probability)
        top3_novel: 
            List of length N where each element is a list of length 3 containing tuples of following form:
                    ((S, V, O), Probability)
        p_ni:
            The Probability that each image contains novelty. A tensor with shape [N, 1]
    
        Returns:
        --------
            List of length N where each element is a list of length 3 containing tuples of following form:
                    ((S, V, O), Probability)
        """  
        p_ni = p_ni.view(-1).numpy()
        N = len(p_ni)
        top3_non_novel = [[(x[i][0], x[i][1], x[i][1] * (1 - y)) for i in range(3)] for x, y in zip(top3_non_novel, p_ni)]
        top3_novel = [[(x[i][0], x[i][1], x[i][1] * y) for i in range(3)] for x, y in zip(top3_novel, p_ni)]
        all_tuples = [x + y for x, y in zip(top3_non_novel, top3_novel)]
        comb_iter = [itertools.combinations(all_tuples[i], 3) for i in range(N)]
        scores = [list(map(lambda x: (x, x[0][2] + x[1][2] + x[2][2]), comb_iter[i])) for i in range(N)]
        top_1_svo = [sorted(i, key=lambda x: x[1], reverse=True)[0] for i in scores]
        top_1_svo = [((a[0][0][0], a[0][0][1]), (a[0][1][0], a[0][1][1]), (a[0][2][0], a[0][2][1])) 
            for a in top_1_svo]

        return top_1_svo

    def type_inference(self, batch_top_svo, batch_p_ni, batch_q_i, batch_f_i, batch_case_i, batch_nc_i):
        batch_p_ni[batch_q_i == 1] = batch_f_i

        self.all_top_svo += batch_top_svo
        self.all_p_ni = torch.cat([self.all_p_ni, batch_p_ni])
        self.all_cases = torch.cat([self.all_cases, batch_case_i])
        self.all_feedback = torch.cat([self.all_feedback, batch_f_i])
        self.all_query_mask = torch.cat([self.all_query_mask, batch_q_i])
        self.all_nc = torch.cat([self.all_nc, batch_nc_i])

        assert self.all_p_ni.shape == self.all_cases == self.all_feedback == self.all_query_mask == self.all_nc

        p_not_novel = 1.0 - self.all_p_ni
        s_unk = torch.tensor([int(a[0][0]) == 0 for a in self.all_top_svo], dtype=torch.int)
        v_unk = torch.tensor([int(a[0][1]) == 0 for a in self.all_top_svo], dtype=torch.int)
        o_unk = torch.tensor([int(a[0][2]) == 0 for a in self.all_top_svo], dtype=torch.int)

        # type-1
        case_3 = (self.all_cases == 3) * p_not_novel
        case_not_3 = (self.all_cases != 3) * s_unk * self.all_p_ni 
        evidence_1 = case_3 + case_not_3
        p_type_1 = self._infer_p_type_from_evidence(evidence_1)

        # type-2
        case_not_3 = (self.all_cases != 3) * v_unk * self.all_p_ni
        evidence_2 = case_3 + case_not_3
        p_type_2 = self._infer_p_type_from_evidence(evidence_2)

        # type-3
        case_not_3 = (self.all_cases != 3) * p_not_novel
        case_3 = (self.all_cases == 3) * o_unk * self.all_p_ni
        evidence_3 = case_3 + case_not_3
        p_type_3 = self._infer_p_type_from_evidence(evidence_3)

        # type-4
        case_not_2 = (self.all_cases != 2) * p_not_novel
        case_1 = (self.all_cases == 1) * (self.all_nc == 1) * (s_unk ^ 1) * (v_unk ^ 1) * (o_unk ^ 1) * self.all_p_ni
        evidence_4 = case_not_2 + case_1
        p_type_4 = self._infer_p_type_from_evidence(evidence_4)

        # type-5
        case_not_2 = (self.all_cases != 2) * p_not_novel
        case_2 = (self.all_cases == 2) * v_unk * self.all_p_ni
        evidence_5 = case_not_2 + case_2
        p_type_5 = self._infer_p_type_from_evidence(evidence_5)

        return torch.tensor([p_type_1, p_type_2, p_type_3, p_type_4, p_type_5])

    def process_feedback(self, batch_query_indices, batch_feedback, batch_cases, novelty_dataset, 
        batch_subj_novelty_scores_u, batch_verb_novelty_scores_u, batch_obj_novelty_scores_u):

        assert batch_query_indices.shape[0] == batch_feedback.shape[0]
        assert all([i < self.test_batch_size for i in batch_query_indices]), "Batch index was out of bound."
        query_indices = batch_query_indices.long()

        t_star = torch.argmax(self.p_type_dist)
        use_hard_labels = torch.isclose(self.p_type_dist[t_star], torch.ones(1))[0]
        t_star += 1

        novel_indices = (batch_feedback == 1).long()
        non_novel_indices = (batch_feedback == 0).long()

        case_1 = (batch_cases[query_indices] == 1).long()
        case_2 = (batch_cases[query_indices] == 2).long()
        case_1_or_2 = case_1 + case_2
        case_1_or_3 = case_1 + (batch_cases[query_indices] == 3).long()

        # negative samples
        negative_subject_indices = query_indices[(non_novel_indices == 1) & (case_1_or_2 == 1)]
        subject_labels = torch.zeros(negative_subject_indices.shape[0])

        negative_verb_indices = query_indices[(non_novel_indices == 1) & (case_1_or_2 == 1)]
        verb_labels = torch.zeros(negative_verb_indices.shape[0])

        negative_object_indices = query_indices[(non_novel_indices == 1) & (case_1_or_3 == 1)]
        object_labels = torch.zeros(negative_object_indices.shape[0])

        # positive samples
        if use_hard_labels:
            positive_subject_indices = query_indices[(novel_indices == 1) & (case_1_or_2 == 1)] if t_star == 1 else torch.tensor([], dtype=torch.long)
            subject_labels = torch.cat([subject_labels, torch.ones(positive_subject_indices.shape[0])])

            positive_object_indices = query_indices[(novel_indices == 1) & (case_1_or_3 == 1)] if t_star == 3 else torch.tensor([], dtype=torch.long)
            object_labels = torch.cat([object_labels, torch.ones(positive_object_indices.shape[0])])

            if t_star == 2:
                positive_verb_indices = query_indices[(novel_indices == 1) & (case_1_or_2 == 1)]
            elif t_star == 5:
                positive_verb_indices = query_indices[(novel_indices == 1) & (case_2 == 1)]
            else:
                positive_verb_indices = torch.tensor([], dtype=torch.long)

            verb_labels = torch.cat([verb_labels, torch.ones(positive_verb_indices.shape[0])])
        else:
            positive_subject_indices = query_indices[(novel_indices == 1) & (case_1_or_2 == 1)]
            subject_labels = torch.cat([subject_labels, torch.ones(positive_subject_indices.shape[0]) * self.p_type_dist[0]])

            positive_object_indices = query_indices[(novel_indices == 1) & (case_1_or_3 == 1)]
            object_labels = torch.cat([object_labels, torch.ones(positive_object_indices.shape[0]) * self.p_type_dist[2]])

            positive_verb_indices = query_indices[(novel_indices == 1) & (case_1 == 1)]
            verb_labels = torch.cat([verb_labels, torch.ones(positive_verb_indices.shape[0]) * self.p_type_dist[1]])
            
            t = query_indices[(novel_indices == 1) & (case_2 == 1)]
            positive_verb_indices = torch.cat([positive_verb_indices, t])
            verb_labels = torch.cat([verb_labels, torch.ones(t.shape[0]) * (self.p_type_dist[1] + self.p_type_dist[4])])

        all_subject_indices = torch.cat([negative_subject_indices, positive_subject_indices])
        all_verb_indices = torch.cat([negative_verb_indices, positive_verb_indices])
        all_object_indices = torch.cat([negative_object_indices, positive_object_indices])

        subject_novelty_start_offset = len(negative_subject_indices)
        verb_novelty_start_offset = len(negative_verb_indices)
        object_novelty_start_offset = len(negative_object_indices)

        self._update_novelty_detectors(novelty_dataset, all_subject_indices, all_verb_indices, all_object_indices, 
            subject_labels, verb_labels, object_labels, 
            subject_novelty_start_offset, verb_novelty_start_offset, object_novelty_start_offset,
            batch_subj_novelty_scores_u, batch_verb_novelty_scores_u, batch_obj_novelty_scores_u)

    def update_world_state(self, p_ni):
        known_log_prob = 1 - p_ni
        novel_log_prob = torch.log(p_ni).numpy()
        changed, red_light_score, per_image_nov_prob = self.change_point_detector.predict_from_log_probs(known_log_prob, novel_log_prob)

        self.post_red = changed
        
    def _get_data_loaders(self):
        valset = DataFactory(
            name="Custom", 
            data_root=self.data_root,
            csv_path=self.csv_path,
            training=False,
            num_subj_cls=self.num_subject_classes,
            num_obj_cls=self.num_object_classes,
            num_action_cls=self.num_verb_classes)

        scg_data_loader = DataLoader(
            dataset=valset,
            collate_fn=custom_collate, 
            batch_size=1,
            num_workers=1, 
            pin_memory=True,
            sampler=None)

        novelty_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = self.data_root,
            csv_path = self.csv_path,
            num_subj_cls = self.num_subject_classes,
            num_obj_cls = self.num_object_classes,
            num_action_cls = self.num_verb_classes,
            training = False,
            image_batch_size = 4,
            feature_extraction_device = 'cuda:0')

        return scg_data_loader, novelty_dataset

    def _compute_case(self, sbox, obox):
        if sbox > 0 and obox > 0:
            return 1
        
        if sbox > 1 and obox == -1:
            return 2

        if sbox == -1 and obox > 0:
            return 3

        raise Exception('Invalid subject/object box id.')

    def _compute_batch_cases(self):
        df = pd.read_csv(self.csv_path, index_col=0)
        df['case'] = df.apply(lambda row : self._compute_case(row['subject_id'], row['object_id']), axis = 1)

        return torch.tensor(df['case'].to_numpy(), dtype=torch.long)

    def _is_svo_type_4(self, svo):
        known_tuples = self.scg_ensemble.train_tuples

        # whether svo contains nulls 
        if svo[0] == -1 or svo[1] == -1 or svo[2] == -1:
            return 0

        # this SVO is a known-combination or contains nulls
        if svo in known_tuples:
            return 0
        
        # whether svo contains novelty
        if svo[0] == 0 or svo[1] == 0 or svo[2] == 0:
            return 0

        return 1

    def _is_novel_combination(self, top_1):
        known_tuples = self.scg_ensemble.train_tuples
        is_nc = [self._is_svo_type_4(t) not in known_tuples for t in top_1]

        return torch.tensor(is_nc, dtype=torch.long)

    def _infer_p_type_from_evidence(self, evidence):
        LARGE_NEG_CONSTANT = -50.0

        zero_indices = torch.nonzero(torch.isclose(evidence, torch.zeros_like(evidence))).view(-1)
        evidence[zero_indices] = LARGE_NEG_CONSTANT
        assert all([e > 0 for e in evidence]), "Zeros encountered, log(0) is undefined."
        evidence = torch.sum(torch.log(evidence))
        p_type = torch.exp(torch.log(0.2) + evidence)

        return p_type

    def _does_svo_contain_novelty(self, svo):
        if svo[0] == 0 or svo[1] == 0 or svo[2] == 0:
            return 1

        return 0

    def _update_novelty_detectors(self, dataset, all_subject_indices, all_verb_indices, all_object_indices,
        subject_labels, verb_labels, object_labels,
        subject_novelty_start_offset, verb_novelty_start_offset, object_novelty_start_offset,
        batch_subject_novelty_scores_u, batch_verb_novelty_scores_u, batch_object_novelty_scores_u):

        appearance_features = [dataset.subject_appearance_features[i].view(-1) for i in all_subject_indices]
        subject_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        assert subject_features.shape[1] == self.NUM_APP_FEATURES

        spatial_features = [dataset.spatial_features[i].view(-1) for i in all_verb_indices]
        spatial_features = torch.vstack(spatial_features) if len(spatial_features) > 1 else torch.tensor([])
        appearance_features = [dataset.verb_appearance_features[i].view(-1) for i in all_verb_indices]
        appearance_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        verb_features = torch.hstack([spatial_features, appearance_features])
        assert verb_features.shape[1] == self.NUM_VERB_FEATURES

        appearance_features = [dataset.object_appearance_features[i].view(-1) for i in all_object_indices]
        object_features = torch.vstack(appearance_features) if len(appearance_features) > 0 else torch.tensor([])
        assert object_features.shape[1] == self.NUM_APP_FEATURES

        subject_novelty_scores = [batch_subject_novelty_scores_u[i] for i in all_subject_indices.tolist()]
        verb_novelty_scores = [batch_verb_novelty_scores_u[i] for i in all_verb_indices.tolist()]
        object_novelty_scores = [batch_object_novelty_scores_u[i] for i in all_object_indices.tolist()]

        assert subject_features.shape[0] == subject_labels.shape[0]
        assert verb_features.shape[0] == verb_labels.shape[0]
        assert object_features.shape[0] == object_labels.shape[0]

        # update supervised novelty dataset
        self.snd_manger.feedback_callback(subject_features, verb_features, object_features, 
            subject_novelty_scores, verb_novelty_scores, object_novelty_scores, 
            subject_labels, verb_labels, object_labels)

        # train supervised novelty detector for the whole data
        S_nov_scores, V_nov_scores, O_nov_scores = self.snd_manger.train()

        # update score contexts for unsueprvised novelty detector
        self.und_manager.feedback_callback(S_nov_scores[:subject_novelty_start_offset], S_nov_scores[:subject_novelty_start_offset],
            V_nov_scores[verb_novelty_start_offset:], V_nov_scores[:verb_novelty_start_offset],
            O_nov_scores[object_novelty_start_offset:], O_nov_scores[:object_novelty_start_offset])

