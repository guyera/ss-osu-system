import os
from abc import ABC, abstractmethod

import torch
import matplotlib.pyplot as plt

import unsupervisednoveltydetection.common
import unsupervisednoveltydetection.training

class TrialLevelPTypeStrategy(ABC):
    @abstractmethod
    def revise_p_type(self, p_type):
        return NotImplemented

class ThresholdTrialLevelPType(TrialLevelPTypeStrategy):
    '''
    Parameters:
        trial_level_p_type: Tensor of size [4] indicating trial level p_type
            estimates for type 1, 2, 3, and 4, respectively.
        threshold: Threshold to compare trial_level_p_type against.
            A hyperparameter.
    '''
    def __init__(self, trial_level_p_type, threshold = 0.98):
        exceeds_threshold = trial_level_p_type >= threshold
        if torch.any(exceeds_threshold):
            self.revised_p_type = exceeds_threshold.to(torch.int)
        else:
            self.revised_p_type = None

    def revise_p_type(self, p_type):
        if self.revised_p_type is not None:
            revised_p_type = p_type * 0 + self.revised_p_type
        else:
            revised_p_type = p_type

        return revised_p_type

class WeightedAverageTrialLevelPType(TrialLevelPTypeStrategy):
    '''
    Parameters:
        trial_level_p_type: Tensor of size [4] indicating trial level p_type
            estimates for type 1, 2, 3, and 4, respectively.
        alpha: Number or scalar tensor indicating the weight for the weighted
            average, i.e. the proportion of the weighted average contributed
            by p(type_i) (per-instance p_type). alpha = 0 means to ignore
            per-instance p(type_i) and look only at trial-level p(type), and
            alpha = 1 means the opposite.
    '''
    def __init__(self, trial_level_p_type, alpha):
        self.trial_level_p_type = trial_level_p_type
        self.alpha = alpha

    def revise_p_type(self, p_type):
        return self.alpha * p_type + (1 - self.alpha) * self.trial_level_p_type

class UnsupervisedNoveltyDetectorLogger:
    def __init__(self):
        self.subject_novelty_scores = []
        self.verb_novelty_scores = []
        self.object_novelty_scores = []

    def log_novelty_scores(self, subject_novelty_scores, verb_novelty_scores, object_novelty_scores):
        self.subject_novelty_scores += subject_novelty_scores
        self.verb_novelty_scores += verb_novelty_scores
        self.object_novelty_scores += object_novelty_scores

    def save_novelty_scores(self, novelty_score_dir_path):
        torch.save(self.subject_novelty_scores, os.path.join(novelty_score_dir_path, 'subject.pth'))
        torch.save(self.verb_novelty_scores, os.path.join(novelty_score_dir_path, 'verb.pth'))
        torch.save(self.object_novelty_scores, os.path.join(novelty_score_dir_path, 'object.pth'))

class UnsupervisedNoveltyDetector:
    def __init__(self, classifier, num_subj_cls, num_obj_cls, num_action_cls):
        self.device = 'cpu'
        
        self.classifier = classifier
        self.confidence_calibrator = unsupervisednoveltydetection.common.ConfidenceCalibrator()
        
        self.known_svo_combinations = torch.zeros(num_subj_cls - 1, num_action_cls - 1, num_obj_cls - 1, dtype = torch.bool)
        self.known_sv_combinations = torch.zeros(num_subj_cls - 1, num_action_cls - 1, dtype = torch.bool)
        self.known_so_combinations = torch.zeros(num_subj_cls - 1, num_obj_cls - 1, dtype = torch.bool)
        self.known_vo_combinations = torch.zeros(num_action_cls - 1, num_obj_cls - 1, dtype = torch.bool)
        self.known_svo_set = set()
        self.known_sv_set = set()
        self.known_so_set = set()
        self.known_vo_set = set()
    
    def to(self, device):
        self.device = device
        self.classifier = self.classifier.to(device)
        self.confidence_calibrator = self.confidence_calibrator.to(device)
        self.known_svo_combinations = self.known_svo_combinations.to(device)
        self.known_sv_combinations = self.known_sv_combinations.to(device)
        self.known_so_combinations = self.known_so_combinations.to(device)
        self.known_vo_combinations = self.known_vo_combinations.to(device)
        return self

    def _update_known_combination_tensors(self):
        self.known_svo_combinations[:] = False
        subject_indices = []
        verb_indices = []
        object_indices = []
        for subject_index, verb_index, object_index in self.known_svo_set:
            subject_indices.append(subject_index)
            verb_indices.append(verb_index)
            object_indices.append(object_index)
        self.known_svo_combinations[(subject_indices, verb_indices, object_indices)] = True

        self.known_sv_combinations[:] = False
        subject_indices = []
        verb_indices = []
        for subject_index, verb_index in self.known_sv_set:
            subject_indices.append(subject_index)
            verb_indices.append(verb_index)
        self.known_sv_combinations[(subject_indices, verb_indices)] = True

        self.known_so_combinations[:] = False
        subject_indices = []
        object_indices = []
        for subject_index, object_index in self.known_so_set:
            subject_indices.append(subject_index)
            object_indices.append(object_index)
        self.known_so_combinations[(subject_indices, object_indices)] = True

        self.known_vo_combinations[:] = False
        verb_indices = []
        object_indices = []
        for verb_index, object_index in self.known_vo_set:
            verb_indices.append(verb_index)
            object_indices.append(object_index)
        self.known_vo_combinations[(verb_indices, object_indices)] = True

    def load_state_dict(self, state_dict):
        classifier_state_dict = state_dict['classifier']
        confidence_calibrator_state_dict = state_dict['confidence_calibrator']
        known_combinations = state_dict['known_combinations']
        self.known_svo_set = known_combinations['svo']
        self.known_sv_set = known_combinations['sv']
        self.known_so_set = known_combinations['so']
        self.known_vo_set = known_combinations['vo']
        
        self.classifier.load_state_dict(classifier_state_dict)
        self.confidence_calibrator.load_state_dict(confidence_calibrator_state_dict)
        
        self._update_known_combination_tensors()

    # Accepts four lists of tuples of integers, and removes those tuples from
    # the corresponding sets of known tuples. Used for simulating type 4
    # novelty.
    def remove_known_tuples(self, svo_tuples = None, sv_tuples = None, so_tuples = None, vo_tuples = None):
        if svo_tuples is not None:
            for svo in svo_tuples:
                if svo in self.known_svo_set:
                    self.known_svo_set.remove(svo)
        
        if sv_tuples is not None:
            for sv in sv_tuples:
                if sv in self.known_sv_set:
                    self.known_sv_set.remove(sv)
        
        if so_tuples is not None:
            for so in so_tuples:
                if so in self.known_so_set:
                    self.known_so_set.remove(so)
        
        if vo_tuples is not None:
            for vo in vo_tuples:
                if vo in self.known_vo_set:
                    self.known_vo_set.remove(vo)
        
        self._update_known_combination_tensors()
    
    def _case_1(self, subject_probs, object_probs, verb_probs, p_type, k):
        t1_raw_joint_probs = verb_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        t1_known_joint_probs = self.known_vo_combinations.to(torch.int) * t1_raw_joint_probs
        t1_known_joint_probs_sum = t1_known_joint_probs.sum()
        if t1_known_joint_probs_sum.item() == 0:
            t1_conditional_joint_probs = t1_known_joint_probs
        else:
            t1_conditional_joint_probs = t1_known_joint_probs / t1_known_joint_probs_sum
        t1_joint_probs = t1_conditional_joint_probs * p_type[0]
        
        t2_raw_joint_probs = subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        t2_known_joint_probs = self.known_so_combinations.to(torch.int) * t2_raw_joint_probs
        t2_known_joint_probs_sum = t2_known_joint_probs.sum()
        if t2_known_joint_probs_sum.item() == 0:
            t2_conditional_joint_probs = t2_known_joint_probs
        else:
            t2_conditional_joint_probs = t2_known_joint_probs / t2_known_joint_probs_sum
        t2_joint_probs = t2_conditional_joint_probs * p_type[1]
        
        t3_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        t3_known_joint_probs = self.known_sv_combinations.to(torch.int) * t3_raw_joint_probs
        t3_known_joint_probs_sum = t3_known_joint_probs.sum()
        if t3_known_joint_probs_sum.item() == 0:
            t3_conditional_joint_probs = t3_known_joint_probs
        else:
            t3_conditional_joint_probs = t3_known_joint_probs / t3_known_joint_probs_sum
        t3_joint_probs = t3_conditional_joint_probs * p_type[2]
        
        t4_raw_joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
        t4_unknown_joint_probs = (1 - self.known_svo_combinations.to(torch.int)) * t4_raw_joint_probs
        t4_unknown_joint_probs_sum = t4_unknown_joint_probs.sum()
        if t4_unknown_joint_probs_sum.item() == 0:
            t4_conditional_joint_probs = t4_unknown_joint_probs
        else:
            t4_conditional_joint_probs = t4_unknown_joint_probs / t4_unknown_joint_probs_sum
        t4_joint_probs = t4_conditional_joint_probs * p_type[3]

        t67_raw_joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
        t67_known_joint_probs = self.known_svo_combinations.to(torch.int) * t67_raw_joint_probs
        t67_known_joint_probs_sum = t67_known_joint_probs.sum()
        if t67_known_joint_probs_sum.item() == 0:
            t67_conditional_joint_probs = t67_known_joint_probs
        else:
            t67_conditional_joint_probs = t67_known_joint_probs / t67_known_joint_probs_sum
        t67_joint_probs = t67_conditional_joint_probs * p_type[4]
        
        # We've computed P(Correct | N_i), but we also have the case information.
        # We need to compute:
        #   P(Correct | N_i, case = 1)
        # = P(Correct | N_i, type =/= 5)
        # = P(Correct, type =/= 5 | N_i) / P(type =/= 5 | N_i)
        ####### Correctness of a novel tuple implies the trial type, so type
        ####### P(Correct, type =/= 5) = P(Correct). Type is also independent
        ####### of N_i. So P(type =/= 5 | N_i) = P(type =/= 5).
        # = P(Correct | N_i) / P(type =/= 5)
        # = P(Correct | N_i) / (\sum_{t in [1, 2, 3, 4]} P(type = t)).
        
        # So divide each joint probability by the type normalizer, or sum of
        # probabilities of possible types.
        
        type_normalizer = p_type[0] + p_type[1] + p_type[2] + p_type[3] + p_type[4]
        
        if type_normalizer > 0:
            t1_joint_probs /= type_normalizer
            t2_joint_probs /= type_normalizer
            t3_joint_probs /= type_normalizer
            t4_joint_probs /= type_normalizer
            t67_joint_probs /= type_normalizer
        
        flattened_t1_joint_probs = torch.flatten(t1_joint_probs)
        sorted_t1_joint_probs, sorted_t1_joint_prob_indices = torch.sort(flattened_t1_joint_probs, descending = True)
        
        flattened_t2_joint_probs = torch.flatten(t2_joint_probs)
        sorted_t2_joint_probs, sorted_t2_joint_prob_indices = torch.sort(flattened_t2_joint_probs, descending = True)
        
        flattened_t3_joint_probs = torch.flatten(t3_joint_probs)
        sorted_t3_joint_probs, sorted_t3_joint_prob_indices = torch.sort(flattened_t3_joint_probs, descending = True)
        
        flattened_t4_joint_probs = torch.flatten(t4_joint_probs)
        sorted_t4_joint_probs, sorted_t4_joint_prob_indices = torch.sort(flattened_t4_joint_probs, descending = True)

        example_predictions = []
        for _ in range(k):
            if sorted_t1_joint_probs[0] >= sorted_t2_joint_probs[0]\
                    and sorted_t1_joint_probs[0] >= sorted_t3_joint_probs[0]\
                    and sorted_t1_joint_probs[0] >= sorted_t4_joint_probs[0]:
                flattened_index = int(sorted_t1_joint_prob_indices[0].item())
                
                verb_skip_interval = object_probs.shape[0]

                verb_index = flattened_index // verb_skip_interval
                verb_skip_total = verb_index * verb_skip_interval
                flattened_index -= verb_skip_total
                
                object_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((0, verb_index + 1, object_index + 1), sorted_t1_joint_probs[0]))
                
                sorted_t1_joint_probs = sorted_t1_joint_probs[1:]
                sorted_t1_joint_prob_indices = sorted_t1_joint_prob_indices[1:]
                
            elif sorted_t2_joint_probs[0] >= sorted_t3_joint_probs[0]\
                    and sorted_t2_joint_probs[0] >= sorted_t4_joint_probs[0]:
                flattened_index = int(sorted_t2_joint_prob_indices[0].item())
                
                subject_skip_interval = object_probs.shape[0]
                
                subject_index = flattened_index // subject_skip_interval
                subject_skip_total = subject_index * subject_skip_interval
                flattened_index -= subject_skip_total
                
                object_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, 0, object_index + 1), sorted_t2_joint_probs[0]))
                
                sorted_t2_joint_probs = sorted_t2_joint_probs[1:]
                sorted_t2_joint_prob_indices = sorted_t2_joint_prob_indices[1:]
                
            elif sorted_t3_joint_probs[0] >= sorted_t4_joint_probs[0]:
                flattened_index = int(sorted_t3_joint_prob_indices[0].item())
                
                subject_skip_interval = verb_probs.shape[0]

                subject_index = flattened_index // subject_skip_interval
                subject_skip_total = subject_index * subject_skip_interval
                flattened_index -= subject_skip_total
                
                verb_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, verb_index + 1, 0), sorted_t3_joint_probs[0]))
                
                sorted_t3_joint_probs = sorted_t3_joint_probs[1:]
                sorted_t3_joint_prob_indices = sorted_t3_joint_prob_indices[1:]
                
            else:
                flattened_index = int(sorted_t4_joint_prob_indices[0].item())
                
                subject_skip_interval = verb_probs.shape[0] * object_probs.shape[0]
                verb_skip_interval = object_probs.shape[0]

                subject_index = flattened_index // subject_skip_interval
                subject_skip_total = subject_index * subject_skip_interval
                flattened_index -= subject_skip_total
                
                verb_index = flattened_index // verb_skip_interval
                verb_skip_total = verb_index * verb_skip_interval
                flattened_index -= verb_skip_total
                
                object_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, verb_index + 1, object_index + 1), sorted_t4_joint_probs[0]))
                
                sorted_t4_joint_probs = sorted_t4_joint_probs[1:]
                sorted_t4_joint_prob_indices = sorted_t4_joint_prob_indices[1:]

        return example_predictions, t67_joint_probs

    def _case_2(self, subject_probs, verb_probs, p_type, k):
        t1_raw_joint_probs = verb_probs
        t1_joint_probs = t1_raw_joint_probs * p_type[0]
        
        t2_raw_joint_probs = subject_probs
        t2_joint_probs = t2_raw_joint_probs * p_type[1]

        t4_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        t4_unknown_joint_probs = (1 - self.known_sv_combinations.to(torch.int)) * t4_raw_joint_probs
        t4_unknown_joint_probs_sum = t4_unknown_joint_probs.sum()
        if t4_unknown_joint_probs_sum.item() == 0:
            t4_conditional_joint_probs = t4_unknown_joint_probs
        else:
            t4_conditional_joint_probs = t4_unknown_joint_probs / t4_unknown_joint_probs_sum
        t4_joint_probs = t4_conditional_joint_probs * p_type[3]

        t67_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        t67_known_joint_probs = self.known_sv_combinations.to(torch.int) * t67_raw_joint_probs
        t67_known_joint_probs_sum = t67_known_joint_probs.sum()
        if t67_known_joint_probs_sum.item() == 0:
            t67_conditional_joint_probs = t67_known_joint_probs
        else:
            t67_conditional_joint_probs = t67_known_joint_probs / t67_known_joint_probs_sum
        t67_joint_probs = t67_conditional_joint_probs * p_type[4]
        
        type_normalizer = p_type[0] + p_type[1] + p_type[3] + p_type[4]
        
        if type_normalizer > 0:
            t1_joint_probs /= type_normalizer
            t2_joint_probs /= type_normalizer
            t4_joint_probs /= type_normalizer
            t67_joint_probs /= type_normalizer
        
        flattened_t1_joint_probs = torch.flatten(t1_joint_probs)
        sorted_t1_joint_probs, sorted_t1_joint_prob_indices = torch.sort(flattened_t1_joint_probs, descending = True)
        
        flattened_t2_joint_probs = torch.flatten(t2_joint_probs)
        sorted_t2_joint_probs, sorted_t2_joint_prob_indices = torch.sort(flattened_t2_joint_probs, descending = True)

        flattened_t4_joint_probs = torch.flatten(t4_joint_probs)
        sorted_t4_joint_probs, sorted_t4_joint_prob_indices = torch.sort(flattened_t4_joint_probs, descending = True)

        example_predictions = []
        for _ in range(k):
            if sorted_t1_joint_probs[0] >= sorted_t2_joint_probs[0]\
                    and sorted_t1_joint_probs[0] >= sorted_t4_joint_probs[0]:
                flattened_index = int(sorted_t1_joint_prob_indices[0].item())
                
                verb_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((0, verb_index + 1, -1), sorted_t1_joint_probs[0]))
                
                sorted_t1_joint_probs = sorted_t1_joint_probs[1:]
                sorted_t1_joint_prob_indices = sorted_t1_joint_prob_indices[1:]
                
            elif sorted_t2_joint_probs[0] >= sorted_t4_joint_probs[0]:
                flattened_index = int(sorted_t2_joint_prob_indices[0].item())
                
                subject_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, 0, -1), sorted_t2_joint_probs[0]))
                
                sorted_t2_joint_probs = sorted_t2_joint_probs[1:]
                sorted_t2_joint_prob_indices = sorted_t2_joint_prob_indices[1:]
            else:
                flattened_index = int(sorted_t4_joint_prob_indices[0].item())
                
                subject_skip_interval = verb_probs.shape[0]

                subject_index = flattened_index // subject_skip_interval
                subject_skip_total = subject_index * subject_skip_interval
                flattened_index -= subject_skip_total
                
                verb_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, verb_index + 1, -1), sorted_t4_joint_probs[0]))
                
                sorted_t4_joint_probs = sorted_t4_joint_probs[1:]
                sorted_t4_joint_prob_indices = sorted_t4_joint_prob_indices[1:]
        
        return example_predictions, t67_joint_probs

    def _case_3(self, object_probs, p_type, k):
        t3_joint_prob = p_type[2]
        
        t67_conditional_joint_probs = object_probs
        t67_joint_probs = t67_conditional_joint_probs * p_type[4]

        type_normalizer = p_type[2] + p_type[4]
        if type_normalizer > 0:
            t3_joint_prob /= type_normalizer
            t67_joint_probs /= type_normalizer

        return ((-1, 0, torch.tensor(0, dtype=torch.long)), t3_joint_prob), t67_joint_probs

    def _known_case_1(self, subject_probs, object_probs, verb_probs, k):
        joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
        
        flattened_joint_probs = torch.flatten(joint_probs)
        sorted_joint_probs, sorted_joint_prob_indices = torch.sort(flattened_joint_probs, descending = True)

        # TODO tensorize
        example_predictions = []
        for i in range(k):
            flattened_index = int(sorted_joint_prob_indices[i].item())
            
            subject_skip_interval = verb_probs.shape[0] * object_probs.shape[0]
            verb_skip_interval = object_probs.shape[0]
            
            subject_index = flattened_index // subject_skip_interval
            subject_skip_total = subject_index * subject_skip_interval
            flattened_index -= subject_skip_total
            
            verb_index = flattened_index // verb_skip_interval
            verb_skip_total = verb_index * verb_skip_interval
            flattened_index -= verb_skip_total
            
            object_index = flattened_index
            
            # Shift labels forward, to allow for anomaly = 0
            example_predictions.append(((subject_index + 1, verb_index + 1, object_index + 1), sorted_joint_probs[i]))
        
        return example_predictions

    def _known_case_2(self, subject_probs, verb_probs, k):
        joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        
        flattened_joint_probs = torch.flatten(joint_probs)
        sorted_joint_probs, sorted_joint_prob_indices = torch.sort(flattened_joint_probs, descending = True)
        
        # TODO tensorize
        example_predictions = []
        for i in range(k):
            flattened_index = int(sorted_joint_prob_indices[i].item())
            
            subject_skip_interval = verb_probs.shape[0]
            
            subject_index = flattened_index // subject_skip_interval
            subject_skip_total = subject_index * subject_skip_interval
            flattened_index -= subject_skip_total

            verb_index = flattened_index
            
            # Shift labels forward, to allow for anomaly = 0
            example_predictions.append(((subject_index + 1, verb_index + 1, -1), sorted_joint_probs[i]))
        
        return example_predictions
    
    def _known_case_3(self, object_probs, k):
        sorted_object_probs, sorted_object_prob_indices = torch.sort(object_probs, descending = True)

        # TODO tensorize
        example_predictions = []
        for i in range(k):
            object_index = int(sorted_object_prob_indices[i].item())
            
            # Shift labels forward, to allow for anomaly = 0
            example_predictions.append(((-1, 0, object_index + 1), sorted_object_probs[i]))
        
        return example_predictions
    
    def _compute_p_known_svo(self, subject_probs, verb_probs, object_probs):
        svo_raw_joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
        svo_known_joint_probs = (self.known_svo_combinations.to(torch.int)) * svo_raw_joint_probs
        return svo_known_joint_probs.sum()

    def _compute_p_known_sv(self, subject_probs, verb_probs):
        sv_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        sv_known_joint_probs = (self.known_sv_combinations.to(torch.int)) * sv_raw_joint_probs
        return sv_known_joint_probs.sum()

    def _compute_p_known_so(self, subject_probs, object_probs):
        so_raw_joint_probs = subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        so_known_joint_probs = (self.known_so_combinations.to(torch.int)) * so_raw_joint_probs
        return so_known_joint_probs.sum()

    def _compute_p_known_vo(self, verb_probs, object_probs):
        vo_raw_joint_probs = verb_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        vo_known_joint_probs = (self.known_vo_combinations.to(torch.int)) * vo_raw_joint_probs
        return vo_known_joint_probs.sum()

    def _compute_p_known_svo_2(self, subject_probs, verb_probs, object_probs):
        svo_raw_joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
        flattened_svo_raw_joint_probs = torch.flatten(svo_raw_joint_probs)
        flattened_known_svo_combinations = torch.flatten(self.known_svo_combinations)
        max_prob, max_prob_idx = torch.max(flattened_svo_raw_joint_probs, dim = 0)
        max_prob_is_known = flattened_known_svo_combinations[max_prob_idx]
        if max_prob_is_known:
            return max_prob * 0 + 1 # i.e. return 1
        else:
            return 1 - max_prob # And we'll take 1 - this in compute_probability_novelty

    def _compute_p_known_sv_2(self, subject_probs, verb_probs):
        sv_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        flattened_sv_raw_joint_probs = torch.flatten(sv_raw_joint_probs)
        flattened_known_sv_combinations = torch.flatten(self.known_sv_combinations)
        max_prob, max_prob_idx = torch.max(flattened_sv_raw_joint_probs, dim = 0)
        max_prob_is_known = flattened_known_sv_combinations[max_prob_idx]
        if max_prob_is_known:
            return max_prob * 0 + 1 # i.e. return 1
        else:
            return 1 - max_prob # And we'll take 1 - this in compute_probability_novelty

    def _compute_p_known_so_2(self, subject_probs, object_probs):
        so_raw_joint_probs = subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        flattened_so_raw_joint_probs = torch.flatten(so_raw_joint_probs)
        flattened_known_so_combinations = torch.flatten(self.known_so_combinations)
        max_prob, max_prob_idx = torch.max(flattened_so_raw_joint_probs, dim = 0)
        max_prob_is_known = flattened_known_so_combinations[max_prob_idx]
        if max_prob_is_known:
            return max_prob * 0 + 1 # i.e. return 1
        else:
            return 1 - max_prob # And we'll take 1 - this in compute_probability_novelty

    def _compute_p_known_vo_2(self, verb_probs, object_probs):
        vo_raw_joint_probs = verb_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        flattened_vo_raw_joint_probs = torch.flatten(vo_raw_joint_probs)
        flattened_known_vo_combinations = torch.flatten(self.known_vo_combinations)
        max_prob, max_prob_idx = torch.max(flattened_vo_raw_joint_probs, dim = 0)
        max_prob_is_known = flattened_known_vo_combinations[max_prob_idx]
        if max_prob_is_known:
            return max_prob * 0 + 1 # i.e. return scalar tensor 1
        else:
            return 1 - max_prob # And we'll take 1 - this in compute_probability_novelty

    # Note: p_type should be a tensor of size [N, 4]; it's per-image p_type
    def top3(self, spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, p_type, trial_level_p_type_strategy = None):
        # Revise per-instance p_type using trial_level_p_type_strategy, if not
        # None
        if trial_level_p_type_strategy is not None:
            p_type = trial_level_p_type_strategy.revise_p_type(p_type)
        
        predictions = []
        t67_joint_probs = []
        cases = []
        results = {}
        p_known_svo = []
        p_known_sv = []
        p_known_so = []
        p_known_vo = []
        for idx in range(len(spatial_features)):
            example_spatial_features = spatial_features[idx]
            example_subject_appearance_features = subject_appearance_features[idx]
            example_object_appearance_features = object_appearance_features[idx]
            example_verb_appearance_features = verb_appearance_features[idx]
            cur_p_type = p_type[idx]
            
            if example_subject_appearance_features is not None:
                example_subject_features = torch.flatten(example_subject_appearance_features).to(self.device)

                subject_logits, subject_score = self.classifier.predict_score_subject(example_subject_features.unsqueeze(0))
                
                subject_score = subject_score.squeeze(0)

                subject_probs = self.confidence_calibrator.calibrate_subject(subject_logits)
                subject_probs = subject_probs.squeeze(0)

            if example_object_appearance_features is not None:
                example_object_features = torch.flatten(example_object_appearance_features).to(self.device)
                
                object_logits, object_score = self.classifier.predict_score_object(example_object_features.unsqueeze(0))
                
                object_score = object_score.squeeze(0)
                
                object_probs = self.confidence_calibrator.calibrate_object(object_logits)
                object_probs = object_probs.squeeze(0)
            
            if example_verb_appearance_features is not None:
                example_verb_features = torch.cat((torch.flatten(example_spatial_features), torch.flatten(example_verb_appearance_features))).to(self.device)
                
                verb_logits, verb_score = self.classifier.predict_score_verb(example_verb_features.unsqueeze(0))
                
                verb_score = verb_score.squeeze(0)

                verb_probs = self.confidence_calibrator.calibrate_verb(verb_logits)
                verb_probs = verb_probs.squeeze(0)
                
            cur_p_known_svo = torch.tensor(0, dtype = torch.float, device = self.device)
            cur_p_known_sv = torch.tensor(0, dtype = torch.float, device = self.device)
            cur_p_known_so = torch.tensor(0, dtype = torch.float, device = self.device)
            cur_p_known_vo = torch.tensor(0, dtype = torch.float, device = self.device)
            if example_subject_appearance_features is not None and example_object_appearance_features is not None:
                # Case 1, S/V/O
                example_case = 1
                example_predictions, example_t67_joint_probs = self._case_1(subject_probs, object_probs, verb_probs, cur_p_type, 3)
                cur_p_known_svo = self._compute_p_known_svo_2(subject_probs, verb_probs, object_probs)
                cur_p_known_sv = self._compute_p_known_sv_2(subject_probs, verb_probs)
                cur_p_known_so = self._compute_p_known_so_2(subject_probs, object_probs)
                cur_p_known_vo = self._compute_p_known_vo_2(verb_probs, object_probs)
            elif example_subject_appearance_features is not None and example_object_appearance_features is None:
                # Case 2, S/V/None
                example_case = 2
                example_predictions, example_t67_joint_probs = self._case_2(subject_probs, verb_probs, cur_p_type, 3)
                cur_p_known_sv = self._compute_p_known_sv_2(subject_probs, verb_probs)
            elif example_subject_appearance_features is None and example_object_appearance_features is not None:
                # Case 3, None/None/O
                example_case = 3
                example_predictions, example_t67_joint_probs = self._case_3(object_probs, cur_p_type, 3)
            else:
                return NotImplemented
            
            predictions.append(example_predictions)
            t67_joint_probs.append(example_t67_joint_probs)
            cases.append(example_case)
            p_known_svo.append(cur_p_known_svo)
            p_known_sv.append(cur_p_known_sv)
            p_known_so.append(cur_p_known_so)
            p_known_vo.append(cur_p_known_vo)
        
        p_known_svo = torch.stack(p_known_svo, dim = 0)
        p_known_sv = torch.stack(p_known_sv, dim = 0)
        p_known_so = torch.stack(p_known_so, dim = 0)
        p_known_vo = torch.stack(p_known_vo, dim = 0)

        results['p_known_svo'] = p_known_svo
        results['p_known_sv'] = p_known_sv
        results['p_known_so'] = p_known_so
        results['p_known_vo'] = p_known_vo
        # Excludes type 6/7 tuples
        results['top3'] = predictions
        # ALL type 6/7 tuples; needed for merging with SCG tuples
        results['t67'] = t67_joint_probs 
        # Example cases each in {1, 2, 3}. Needed for merging with SCG tuples
        results['cases'] = cases

        return results

    def known_top3(self, spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features):
        predictions = []
        for idx in range(len(spatial_features)):
            example_spatial_features = spatial_features[idx]
            example_subject_appearance_features = subject_appearance_features[idx]
            example_object_appearance_features = object_appearance_features[idx]
            example_verb_appearance_features = verb_appearance_features[idx]
            
            if example_subject_appearance_features is not None:
                example_subject_features = torch.flatten(example_subject_appearance_features).to(self.device)

                subject_logits = self.classifier.predict_subject(example_subject_features.unsqueeze(0))

                subject_probs = self.confidence_calibrator.calibrate_subject(subject_logits)
                subject_probs = subject_probs.squeeze(0)

            if example_object_appearance_features is not None:
                example_object_features = torch.flatten(example_object_appearance_features).to(self.device)
                
                object_logits = self.classifier.predict_object(example_object_features.unsqueeze(0))
                
                object_probs = self.confidence_calibrator.calibrate_object(object_logits)
                object_probs = object_probs.squeeze(0)
            
            if example_verb_appearance_features is not None:
                example_verb_features = torch.cat((torch.flatten(example_spatial_features), torch.flatten(example_verb_appearance_features))).to(self.device)
                
                verb_logits = self.classifier.predict_verb(example_verb_features.unsqueeze(0))
                
                verb_probs = self.confidence_calibrator.calibrate_verb(verb_logits)
                verb_probs = verb_probs.squeeze(0)
            
            if example_subject_appearance_features is not None and example_object_appearance_features is not None:
                # Case 1, S/V/O
                example_predictions = self._known_case_1(subject_probs, object_probs, verb_probs, 3)
            elif example_subject_appearance_features is not None and example_object_appearance_features is None:
                # Case 2, S/V/None
                example_predictions = self._known_case_2(subject_probs, verb_probs, 3)
            elif example_subject_appearance_features is None and example_object_appearance_features is not None:
                # Case 3, None/None/O
                example_predictions = self._known_case_3(object_probs, 3)
            else:
                return NotImplemented
            
            predictions.append(example_predictions)

        return predictions

    def score_subject(self, subject_appearance_features):
        novelty_scores = []
        for example_subject_appearance_features in subject_appearance_features:
            if example_subject_appearance_features is not None:
                example_features = torch.flatten(example_subject_appearance_features).to(self.device)
                
                score = self.classifier.score_subject(example_features.unsqueeze(0))
                score = score.squeeze(0)
                novelty_scores.append(score)
            else:
                novelty_scores.append(None)
            
        return novelty_scores

    def score_object(self, object_appearance_features):
        novelty_scores = []
        for example_object_appearance_features in object_appearance_features:
            if example_object_appearance_features is not None:
                example_features = torch.flatten(example_object_appearance_features).to(self.device)
                
                score = self.classifier.score_object(example_features.unsqueeze(0))
                score = score.squeeze(0)
                novelty_scores.append(score)
            else:
                novelty_scores.append(None)
            
        return novelty_scores

    def score_verb(self, spatial_features, verb_appearance_features):
        novelty_scores = []
        for idx in range(len(verb_appearance_features)):
            example_spatial_features = spatial_features[idx]
            example_verb_appearance_features = verb_appearance_features[idx]
            
            if example_verb_appearance_features is not None and example_spatial_features is not None:
                example_features = torch.cat((torch.flatten(example_spatial_features), torch.flatten(example_verb_appearance_features)))
                example_features = example_features.to(self.device)
                
                score = self.classifier.score_verb(example_features.unsqueeze(0))
                score = score.squeeze(0)
                novelty_scores.append(score)
            else:
                novelty_scores.append(None)
            
        return novelty_scores
    
    def score(self, spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features):
        results = {}
        results['subject_novelty_score'] = self.score_subject(subject_appearance_features)
        results['object_novelty_score'] = self.score_object(object_appearance_features)
        results['verb_novelty_score'] = self.score_verb(spatial_features, verb_appearance_features)
        return results

    def scores_and_p_t4(self, spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features):
        subject_scores = []
        subject_probs = []
        verb_scores = []
        verb_probs = []
        object_scores = []
        object_probs = []
        for idx in range(len(spatial_features)):
            example_spatial_features = spatial_features[idx]
            example_subject_appearance_features = subject_appearance_features[idx]
            example_verb_appearance_features = verb_appearance_features[idx]
            example_object_appearance_features = object_appearance_features[idx]
            
            if example_subject_appearance_features is not None:
                example_subject_features = torch.flatten(example_subject_appearance_features).to(self.device)
                
                subject_logits, subject_score = self.classifier.predict_score_subject(example_subject_features.unsqueeze(0))
                
                subject_score = subject_score.squeeze(0)
                subject_scores.append(subject_score)

                example_subject_probs = self.confidence_calibrator.calibrate_subject(subject_logits)
                example_subject_probs = example_subject_probs.squeeze(0)
                subject_probs.append(example_subject_probs)
            else:
                subject_scores.append(None)
                subject_probs.append(None)

            if example_object_appearance_features is not None:
                example_object_features = torch.flatten(example_object_appearance_features).to(self.device)
                
                object_logits, object_score = self.classifier.predict_score_object(example_object_features.unsqueeze(0))
                
                object_score = object_score.squeeze(0)
                object_scores.append(object_score)
                
                example_object_probs = self.confidence_calibrator.calibrate_object(object_logits)
                example_object_probs = example_object_probs.squeeze(0)
                object_probs.append(example_object_probs)
            else:
                object_scores.append(None)
                object_probs.append(None)
            
            if example_verb_appearance_features is not None:
                example_verb_features = torch.cat((torch.flatten(example_spatial_features), torch.flatten(example_verb_appearance_features))).to(self.device)
                
                verb_logits, verb_score = self.classifier.predict_score_verb(example_verb_features.unsqueeze(0))
                
                verb_score = verb_score.squeeze(0)
                verb_scores.append(verb_score)

                example_verb_probs = self.confidence_calibrator.calibrate_verb(verb_logits)
                example_verb_probs = example_verb_probs.squeeze(0)
                verb_probs.append(example_verb_probs)
            else:
                verb_scores.append(None)
                verb_probs.append(None)
        
        # Now compute t4 probabilities based on each example's case
        p_t4 = []
        for idx in range(len(subject_probs)):
            assert ((example_subject_probs is not None and example_verb_probs is not None) or example_object_probs is not None)

            example_subject_probs = subject_probs[idx]
            example_verb_probs = verb_probs[idx]
            example_object_probs = object_probs[idx]
            
            if example_subject_probs is not None and example_object_probs is not None:
                # Case 1. P(type = 4 | no novel boxes) = P(predicted SVO is a
                # novel combination) = 1 - P(predicted SVO is a known
                # combination)
                p_t4.append(1 - self._compute_p_known_svo(example_subject_probs, example_verb_probs, example_object_probs))
            elif example_subject_probs is not None:
                # Case 2. P(type = 4 | no novel boxes) = P(predicted SV is a
                # novel combination) = 1 - P(predicted SV is a known
                # combination)
                p_t4.append(1 - self._compute_p_known_sv(example_subject_probs, example_verb_probs))
            elif example_object_probs is not None:
                # Case 3. P(type = 4 | no novel boxes) = 0; there is just an
                # object (no subject or verb), and so a novel combination would
                # actually mean a novel object, which is considered a type 3
                # novelty instead of type 4.
                p_t4.append(torch.tensor(0.0, device = example_object_probs.device))
            else:
                # Subject and object boxes cannot both be missing. Shouldn't
                # have made it past earlier assert. If this error is raised,
                # then there is a bug; investigate.
                raise ValueError
        
        results = {}
        results['subject_novelty_score'] = subject_scores
        results['verb_novelty_score'] = verb_scores
        results['object_novelty_score'] = object_scores
        results['p_t4'] = p_t4
        
        return results
