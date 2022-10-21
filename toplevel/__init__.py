import itertools
from pathlib import Path
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
from scipy.stats import ks_2samp
from torchvision.models import resnet50, swin_t
from unsupervisednoveltydetection.training import NoveltyDetectorTrainer

import os


class TopLevelApp:
    def __init__(self, ensemble_path, data_root, pretrained_unsupervised_module_path, pretrained_backbone_path, 
        feedback_enabled, given_detection, log, log_dir, ignore_verb_novelty, train_csv_path, val_csv_path, val_incident_csv_path,
        trial_size, trial_batch_size, retraining_batch_size, disable_retraining):

        if not Path(ensemble_path).exists():
            raise Exception(f'pretrained SCG model was not found in path {ensemble_path}')
        if not Path(pretrained_backbone_path).exists():
            raise Exception(f'pretrained backbone was not found in path {pretrained_backbone_path}')
        # if not Path(train_csv_path).exists():
        #     raise Exception(f'training CSV was not found in path {train_csv_path}')
        # if not Path(val_csv_path).exists():
        #     raise Exception(f'validation CSV was not found in path {val_csv_path}')
        # import ipdb; ipdb.set_trace()
        self.data_root = data_root
        self.train_csv_path = train_csv_path
        self.val_csv_path = val_csv_path
        self.val_incident_csv_path = val_incident_csv_path
        self.pretrained_backbone_path = pretrained_backbone_path
        self.pretrained_unsupervised_module_path = pretrained_unsupervised_module_path
        self.NUM_SUBJECT_CLASSES = 5
        self.NUM_OBJECT_CLASSES = 12
        self.NUM_VERB_CLASSES = 8
        self.NUM_SPATIAL_FEATURES = 2 * 36
        self.post_red = False
        self.p_type_dist =torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20]) if not ignore_verb_novelty else torch.tensor([1/3, 0, 1/3, 1/3])
        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
        self.all_p_ni = torch.tensor([])
        self.all_p_ni_raw = np.array([])
        self.all_nc = torch.tensor([])
        self.all_top_1_svo = []
        self.all_top_3_svo = []
        self.all_p_type = torch.tensor([])
        self.all_red_light_scores = np.array([])

        self.scg_ensemble = Ensemble(ensemble_path, self.NUM_OBJECT_CLASSES, self.NUM_SUBJECT_CLASSES, 
            self.NUM_VERB_CLASSES, data_root=None, cal_csv_path=None, val_csv_path=None)

        self.p_type_th = 0.75
        self.post_red_base = None
        self.batch_context = BatchContext()
        self.feedback_enabled = feedback_enabled
        self.p_val_cuttoff =  0.0085542
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
        self.p_type_hist = [self.p_type_dist.numpy()]
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
        self.retraining_buffer = pd.DataFrame(columns=['new_image_path', 'subject_name', 'subject_id', 'original_subject_id',
            'object_name', 'object_id', 'original_object_id', 'verb_name',
            'verb_id', 'original_verb_id', 'image_width', 'image_height',
            'subject_ymin', 'subject_xmin', 'subject_ymax', 'subject_xmax',
            'object_ymin', 'object_xmin', 'object_ymax', 'object_xmax'])
        self.retraining_batch_size = retraining_batch_size
        self.disable_retraining = disable_retraining
                
        self._reset_backbone_and_detectors()
        
    def reset(self):
        self.post_red = False
        self.p_type_dist = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20]) if not self.ignore_verb_novelty else torch.tensor([1/3, 0, 1/3, 1/3])
        self.unsupervised_aucs = [None, None, None]
        self.supervised_aucs = [None, None, None]
        self.all_query_masks = torch.tensor([])
        self.all_feedback = torch.tensor([])
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
        self.p_type_hist = [self.p_type_dist.numpy()]
        self.per_image_p_type = torch.tensor([])
        self.all_red_light_scores = np.array([])
        self.red_light_img = None
        self.red_light_this_batch = False
        self.t_tn = None
        self.batch_num = 0 
        self.num_retrains_so_far = 0
        self.retraining_buffer = self.retraining_buffer.iloc[0:0]        
        self._reset_backbone_and_detectors()

    def process_batch(self, csv_path, test_id, round_id, img_paths):
        if csv_path is None:
            raise Exception('path to csv was None')

        self.batch_num += 1
        self.batch_context.reset()
        # initialize data loaders
        scg_data_loader, novelty_dataset, N, image_paths, df = self._load_data(csv_path)
        self.batch_context.df = df

        # top-3 SVOs from SCG ensemble
        scg_preds = self.scg_ensemble.get_all_SVO_preds(scg_data_loader, False)

        # unsupervised novelty scores
        unsupervised_results, incident_activation_statistical_scores = self.und_manager.score(self.backbone, novelty_dataset)

        subject_novelty_scores_u = unsupervised_results['subject_novelty_score']
        verb_novelty_scores_u = unsupervised_results['verb_novelty_score']
        object_novelty_scores_u = unsupervised_results['object_novelty_score']
        assert len(subject_novelty_scores_u) == len(verb_novelty_scores_u) == len(object_novelty_scores_u)

        self.subj_novelty_scores_un += subject_novelty_scores_u
        self.verb_novelty_scores_un += verb_novelty_scores_u
        self.obj_novelty_scores_un += object_novelty_scores_u


        # compute P_n                        
        with torch.no_grad():
            case_1_lr, case_2_lr, case_3_lr = self.und_manager.get_calibrators()
            batch_p_type, p_ni = compute_probability_novelty(subject_novelty_scores_u, verb_novelty_scores_u, object_novelty_scores_u, 
                incident_activation_statistical_scores, case_1_lr, case_2_lr, case_3_lr, ignore_t2_in_pni=self.ignore_verb_novelty, p_t4=unsupervised_results['p_t4'])

        p_ni = p_ni.cpu().float()            
        batch_p_type = batch_p_type.cpu()
        self.per_image_p_type = torch.cat([self.per_image_p_type, batch_p_type])

        top3 = None
        top3_probs = None
                
        if not self.given_detection:
            top3, top3_probs = self._top3(scg_preds, p_ni, novelty_dataset, batch_p_type)
        else:
            top3, top3_probs = self._top3_given_detection(scg_preds, p_ni, novelty_dataset, batch_p_type, img_paths)
        
        
        self.batch_context.p_ni = p_ni
        self.batch_context.subject_novelty_scores_u = subject_novelty_scores_u
        self.batch_context.verb_novelty_scores_u = verb_novelty_scores_u
        self.batch_context.object_novelty_scores_u = object_novelty_scores_u
        self.batch_context.image_paths = image_paths
        self.batch_context.novelty_dataset = novelty_dataset
        self.batch_context.top_3 = top3
        self.batch_context.p_type = batch_p_type
        self.batch_context.round_id = round_id
        
        red_light_scores = self._compute_red_light_scores(p_ni, N, img_paths)

        if not self.post_red or not self.feedback_enabled:
            batch_query_mask = torch.zeros(N, dtype=torch.long)
            batch_feedback_mask = torch.zeros(N, dtype=torch.long)
            self._accumulate(top3, p_ni, p_ni.numpy(), batch_feedback_mask, batch_query_mask, batch_p_type)

        self.all_red_light_scores = np.concatenate([self.all_red_light_scores, red_light_scores])
        # import ipdb; ipdb.set_trace()
        ret = {}
        ret['p_ni'] = p_ni.tolist()
        ret['red_light_score'] = red_light_scores
        ret['svo'] = top3
        ret['svo_probs'] = top3_probs

        # print(ret)
                                                                  
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
            logs['post_red_base'] = self.post_red_base
            logs['p_type'] = self.p_type_hist
            logs['red_light_scores'] = self.all_red_light_scores

            pickle.dump(logs, handle)

    def select_queries(self, feedback_max_ids):
        assert self.post_red, "query selection shoudn't happen pre-red button"
        assert self.batch_context.is_set(), "no batch context."
        assert self.feedback_enabled, "feedback is disabled"

        query_indices = select_queries(feedback_max_ids, torch.tensor([1/3, 1/3, 1/3, 0]), self.batch_context.p_ni, 
            self.batch_context.subject_novelty_scores_u, self.batch_context.verb_novelty_scores_u, 
            self.batch_context.object_novelty_scores_u)

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
            
        if self.t_tn is None:
            for i, img in enumerate(self.batch_context.image_paths):
                if img in feedback_results and feedback_results[img] == 1:
                    self.t_tn = i + self.all_p_ni.shape[0]
                    break
            
        N = len(self.batch_context.image_paths)
        self.batch_context.feedback_mask = torch.zeros(N, dtype=torch.long)

        feedback = []
        for q in self.batch_context.query_indices:
            img_path = self.batch_context.image_paths[q]
            feedback.append(feedback_results[img_path])

        feedback_t = torch.tensor(feedback, dtype=torch.long)
        self.batch_context.feedback_mask[self.batch_context.query_indices] = feedback_t

        # adjust p_ni based on feedback
        p_ni_raw = np.copy(self.batch_context.p_ni)
        self.batch_context.p_ni[self.batch_context.query_indices] = feedback_t.float()
        
        self._accumulate(self.batch_context.top_3, self.batch_context.p_ni, p_ni_raw,
            self.batch_context.feedback_mask, self.batch_context.query_mask, self.batch_context.p_type)

        self._type_inference()
                
        f = self.batch_context.df.apply(lambda row : feedback_results[row['new_image_path']] if row['new_image_path'] in feedback_results else -1, 
            axis = 1)      
        self.batch_context.df['feedback'] = f

        if not self.disable_retraining:
            self._retrain_supervised_detectors()                

    def _top3(self, scg_preds, p_ni, novelty_dataset, batch_p_type):
        top3 = None
        top3_probs = None
        # merged function is called inside top3
        top3_merged = self.und_manager.top3(self.backbone, novelty_dataset, batch_p_type, self.p_type_dist, scg_preds, p_ni)
        top3 = [[e[0] for e in m] for m in top3_merged]
        top3_probs = [[e[1] for e in m] for m in top3_merged]     
        # if self.t_tn is None:
        #     top3 = [[e[0] for e in m] for m in scg_preds]
        #     top3_probs = [[e[1] for e in m] for m in scg_preds]    
        # else:
        #     assert self.post_red
        #     import ipdb; ipdb.set_trace()
        #     top3_und = self.und_manager.top3(self.backbone, novelty_dataset, batch_p_type, self.p_type_dist)
        #     merged = self._merge_top3_SVOs(scg_preds, top3_und, p_ni)
        #     top3 = [[e[0] for e in m] for m in merged]
        #     top3_probs = [[e[1] for e in m] for m in merged]         
                       
        return top3, top3_probs
                       
    def _top3_given_detection(self, scg_preds, p_ni, novelty_dataset, batch_p_type, img_paths):
        top3 = None
        top3_probs = None
        
        if not self.post_red:
            # if the red-light image happens to be in this batch
            if self.red_light_this_batch:
                idx = -1
            
                try:
                   idx = img_paths.index(self.red_light_img)
                except ValueError:
                    print(f'specified red-light image was not found in the given image paths.')
                    raise Exception()
            
                # use scg predictions up until the red-light image, merge with the unsupervised top3 from that point on
                p_ni_for_top3 = np.copy(p_ni)
                p_ni_for_top3[:idx] = 0
                top3_und = self.und_manager.top3(self.backbone, novelty_dataset, batch_p_type, self.p_type_dist)
                # merged = self._merge_top3_SVOs(scg_preds, top3_und, p_ni_for_top3)
                top3 = [[e[0] for e in m] for m in merged]
                top3_probs = [[e[1] for e in m] for m in merged] 
            else:
                top3 = [[e[0] for e in m] for m in scg_preds]
                top3_probs = [[e[1] for e in m] for m in scg_preds]    
        # just merge as usual 
        else:
            top3_und = self.und_manager.top3(self.backbone, novelty_dataset, batch_p_type, self.p_type_dist)
            merged = self._merge_top3_SVOs(scg_preds, top3_und, p_ni)
            top3 = [[e[0] for e in m] for m in merged]
            top3_probs = [[e[1] for e in m] for m in merged]          
            
        return top3, top3_probs
        
    def _compute_red_light_scores(self, p_ni, N, img_paths):
        red_light_scores = np.ones(N)
        all_p_ni = np.concatenate([self.all_p_ni.numpy(), p_ni.numpy()])
        
        if not self.post_red:
            if all_p_ni.shape[0] >= 60:    
                start = all_p_ni.shape[0] - N
                
                for i in range(max(start, 60 + self.windows_size), start + N):
                    p_val = ks_2samp(all_p_ni[:60], all_p_ni[i - self.windows_size: i], alternative='greater')[1]
                    red_light_scores[i - start] = p_val

                if not self.given_detection:
                    EPS = 1e-4
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

    def _type_inference(self):
        assert self.post_red, "type-inference shouldn't be called pre-red button"
        assert torch.numel(self.all_p_type) > 0, "no per-image p_type"
        assert self.all_p_ni.shape[0] == self.all_p_type.shape[0], "p_type/p_ni shape mismatch"
    
        filter_v = self.all_p_ni[self.post_red_base:] >= self.p_type_th
        
        if not torch.any(filter_v):
            return
        
        filtered = self.all_p_type[self.post_red_base:][filter_v]
    
        prior = 0.25 if not self.ignore_verb_novelty else 1/3
    
        log_p_type_1 = self._infer_log_p_type(prior, filtered[:, 0])
        log_p_type_2 = self._infer_log_p_type(prior if not self.ignore_verb_novelty else 0, 
                                              filtered[:, 1] if not self.ignore_verb_novelty else torch.zeros(filtered.shape[0]))
        log_p_type_3 = self._infer_log_p_type(prior, filtered[:, 2])
        log_p_type_4 = self._infer_log_p_type(prior, filtered[:, 3])
    
        self.p_type_dist = torch.tensor([log_p_type_1, log_p_type_2, log_p_type_3, log_p_type_4])
        self.p_type_dist = torch.nn.functional.softmax(self.p_type_dist, dim=0).float()
        
        self.p_type_hist.append(self.p_type_dist.numpy())
                
        assert not torch.any(torch.isnan(self.p_type_dist)), "NaNs in p_type."
        assert not torch.any(torch.isinf(self.p_type_dist)), "Infs in p_type."

    def _infer_log_p_type(self, prior, evidence):
        LARGE_NEG_CONSTANT = -50.0

        zero_indices = torch.nonzero(torch.isclose(evidence, torch.zeros_like(evidence))).view(-1)
        log_ev = torch.log(evidence)
        log_ev[zero_indices] = LARGE_NEG_CONSTANT
        evidence = torch.sum(log_ev)
        log_p_type = prior + evidence

        return log_p_type

    def _load_data(self, csv_path):
        valset = DataFactory(name="Custom", 
            data_root=self.data_root,
            csv_path=csv_path,
            training=False)

        scg_data_loader = DataLoader(dataset=valset,
            collate_fn=custom_collate, 
            batch_size=1,
            num_workers=1, 
            pin_memory=False,
            sampler=None)
            
        novelty_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(name = 'Custom',
            data_root = self.data_root,
            csv_path = csv_path,
            training = False,
            image_batch_size = 16,
            backbone = self.backbone,
            feature_extraction_device = 'cuda:0',
            cache_to_disk = False)

        N = len(valset)
        
        df = pd.read_csv(csv_path, index_col=0)
        image_paths = df['new_image_path'].to_list()

        cases = df.apply(lambda row : self._compute_case(row['subject_ymin'], row['subject_xmin'], 
            row['subject_ymax'], row['subject_xmax'], 
            row['object_ymin'], row['object_xmin'], row['object_ymax'], row['object_xmax']), axis = 1)
        df['case'] = cases.to_numpy()

        return scg_data_loader, novelty_dataset, N, image_paths, df

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

    def _accumulate(self, top_3, p_ni, p_ni_raw, feedback_mask, query_mask, p_type):
        self.all_top_3_svo += top_3
        self.all_p_ni = torch.cat([self.all_p_ni, p_ni])
        self.all_p_ni_raw = np.concatenate([self.all_p_ni_raw, p_ni_raw])
        self.all_feedback = torch.cat([self.all_feedback, feedback_mask])
        self.all_query_masks = torch.cat([self.all_query_masks, query_mask])
        self.all_p_type = torch.cat([self.all_p_type, p_type])
        
        assert self.all_p_ni.shape[0] == self.all_p_type.shape[0] == len(self.all_top_3_svo)
        assert self.all_feedback.shape == self.all_query_masks.shape == self.all_p_ni.shape

    def _reset_backbone_and_detectors(self):
        self.und_manager = UnsupervisedNoveltyDetectionManager(self.pretrained_unsupervised_module_path, 
            self.NUM_SUBJECT_CLASSES, 
            self.NUM_VERB_CLASSES, 
            self.NUM_OBJECT_CLASSES, 
            self.NUM_SPATIAL_FEATURES,
            0.98)        
        if 'resnet' in self.pretrained_backbone_path:
            backbone = resnet50(weights= None) #resnet50(pretrained = False)  weights="IMAGENET1K_V1"
            backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
            backbone_state_dict = torch.load(self.pretrained_backbone_path)
            backbone.load_state_dict(backbone_state_dict)
            model_ = 'resnet'
           

        if 'swin_t' in self.pretrained_backbone_path:
            backbone = swin_t(weights= None) # pretrained = False, 
            backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
            backbone_state_dict = torch.load(self.pretrained_backbone_path)
            backbone.load_state_dict(backbone_state_dict)
            model_ = 'swin_t'
        backbone = backbone.to('cuda:0')
        backbone.eval()
        self.backbone = backbone

        self.novelty_trainer = NoveltyDetectorTrainer(self.data_root, self.train_csv_path, self.val_csv_path, self.val_incident_csv_path, self.retraining_batch_size, model_)

    def _retrain_supervised_detectors(self):
        t_star = torch.argmax(self.p_type_dist) + 1
        df_temp = self.batch_context.df.copy()
        
        # subject
        df_subj_neg_1 = df_temp[(df_temp['case'] == 1) & (df_temp['feedback'] == 0)]
        df_subj_neg_2 = df_temp[(df_temp['case'] == 2) & (df_temp['feedback'] == 0)]
        df_subj = pd.concat([df_subj_neg_1, df_subj_neg_2])
        
        # verb
        df_verb_neg_1 = df_temp[(df_temp['case'] == 1) & (df_temp['feedback'] == 0)]
        df_verb_neg_2 = df_temp[(df_temp['case'] == 2) & (df_temp['feedback'] == 0)]
        df_verb = pd.concat([df_verb_neg_1, df_verb_neg_2])
        
        # object
        df_obj_neg_1 = df_temp[(df_temp['case'] == 1) & (df_temp['feedback'] == 0)]
        df_obj_neg_3 = df_temp[(df_temp['case'] == 3) & (df_temp['feedback'] == 0)]
        df_obj = pd.concat([df_obj_neg_1, df_obj_neg_3])
                
        if t_star == 1:
            df_subj_pos_1 = df_temp[(df_temp['case'] == 1) & (df_temp['feedback'] == 1)]
            df_subj_pos_2 = df_temp[(df_temp['case'] == 2) & (df_temp['feedback'] == 1)]
            df_subj_pos = pd.concat([df_subj_pos_1, df_subj_pos_2])
            df_subj = pd.concat([df_subj, df_subj_pos])
                                   
        if t_star == 2:
            df_verb_pos_1 = df_temp[(df_temp['case'] == 1) & (df_temp['feedback'] == 1)]
            df_verb_pos_2 = df_temp[(df_temp['case'] == 2) & (df_temp['feedback'] == 1)]
            df_verb_pos = pd.concat([df_verb_pos_1, df_verb_pos_2])
            df_verb = pd.concat([df_verb, df_verb_pos])
                        
        if t_star == 3:
            df_obj_pos_1 = df_temp[(df_temp['case'] == 1) & (df_temp['feedback'] == 1)]
            df_obj_pos_3 = df_temp[(df_temp['case'] == 3) & (df_temp['feedback'] == 1)]
            df_obj_pos = pd.concat([df_obj_pos_1, df_obj_pos_3])
            df_obj = pd.concat([df_obj, df_obj_pos])
                
        p_type = self.p_type_dist.numpy()
        use_hard_labels = np.isclose(p_type[t_star - 1], 1.0)
        
        if df_subj.shape[0] > 0:
            df_subj['is_novel'] = df_subj.apply(lambda row: row['feedback'] * use_hard_labels + 
                row['feedback'] * (1 - use_hard_labels) * p_type[0], axis=1)        
            df_subj = df_subj[df_subj['is_novel'] >= 0.9]
      
        if df_verb.shape[0] > 0:
            df_verb['is_novel'] = df_verb.apply(lambda row: row['feedback'] * use_hard_labels 
                + row['feedback'] * (1 - use_hard_labels) * p_type[1], axis=1)
            df_verb = df_verb[df_verb['is_novel'] >= 0.9]

        if df_obj.shape[0] > 0:
            df_obj['is_novel'] = df_obj.apply(lambda row: row['feedback'] * use_hard_labels * (row['case'] == 3) + 
                row['feedback'] * (1 - use_hard_labels) * p_type[2], axis=1)
            df_obj = df_obj[df_obj['is_novel'] >= 0.9]

        df_final = pd.concat([df_subj, df_verb, df_obj]) 
        
        if df_final.shape[0] > 0:
            df_final.drop(columns=['case', 'feedback', 'is_novel'], inplace=True)
            df_final.drop_duplicates(inplace=True)        
            self.retraining_buffer = pd.concat([self.retraining_buffer, df_final])
            
        retrain_cond_1 = self.num_retrains_so_far == 0 and self.retraining_buffer.shape[0] >= 15
        retrain_cond_2 = (self.batch_num == self.second_retrain_batch_num) and (self.retraining_buffer.shape[0] > 0)
        
        if retrain_cond_1 or retrain_cond_2:
            self.num_retrains_so_far += 1
            csv_path = self.temp_path.joinpath(f'{os.getpid()}_batch_{self.batch_context.round_id}_retrain.csv')
            self.retraining_buffer.to_csv(csv_path, index=True)

                
            self.novelty_trainer.add_feedback_data(self.data_root, csv_path)
            self.novelty_trainer.prepare_for_retraining(self.backbone, self.und_manager.detector, 
                self.und_manager.case_1_logistic_regression,
                self.und_manager.case_2_logistic_regression,
                self.und_manager.case_3_logistic_regression)
            self.novelty_trainer.train_novelty_detection_module(self.backbone, self.und_manager.detector, 
                self.und_manager.case_1_logistic_regression,
                self.und_manager.case_2_logistic_regression,
                self.und_manager.case_3_logistic_regression)
            
            self.retraining_buffer = self.retraining_buffer.iloc[0:0]
            assert self.retraining_buffer.shape[0] == 0
                
            if csv_path.exists():
                os.remove(csv_path)