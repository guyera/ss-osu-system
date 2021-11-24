import torch

import unsupervisednoveltydetection.common

class UnsupervisedNoveltyDetector:
    # num_appearance_features is the number of features after ROI pooling
    # and flattening the resulting representation. It should be 12544 for the
    # resnet50 backbone and the typical pooling parameters.

    # num_verb_features is the number of appearance features, plus the number
    # of spatial features. It should be 12616, since there are 72 additional
    # spatial features for the verb box classifier.

    # num_hidden_nodes is the number of hidden nodes used in each box classifier
    # MLP. It should be 1024 for the current box classifiers.
    def __init__(self, num_appearance_features, num_verb_features, num_hidden_nodes, num_subj_cls, num_obj_cls, num_action_cls):
        self.device = 'cpu'

        self.classifier = unsupervisednoveltydetection.common.Classifier(
            num_appearance_features,
            num_verb_features,
            num_hidden_nodes,
            num_subj_cls,
            num_obj_cls,
            num_action_cls
        )
    
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
        t1_joint_probs = t1_conditional_joint_probs * p_type[1]
        
        t2_raw_joint_probs = subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        t2_known_joint_probs = self.known_so_combinations.to(torch.int) * t2_raw_joint_probs
        t2_known_joint_probs_sum = t2_known_joint_probs.sum()
        if t2_known_joint_probs_sum.item() == 0:
            t2_conditional_joint_probs = t2_known_joint_probs
        else:
            t2_conditional_joint_probs = t2_known_joint_probs / t2_known_joint_probs_sum
        t2_joint_probs = t2_conditional_joint_probs * p_type[2]
        
        t3_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        t3_known_joint_probs = self.known_sv_combinations.to(torch.int) * t3_raw_joint_probs
        t3_known_joint_probs_sum = t3_known_joint_probs.sum()
        if t3_known_joint_probs_sum.item() == 0:
            t3_conditional_joint_probs = t3_known_joint_probs
        else:
            t3_conditional_joint_probs = t3_known_joint_probs / t3_known_joint_probs_sum
        t3_joint_probs = t3_conditional_joint_probs * p_type[3]
        
        t4_raw_joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
        t4_unknown_joint_probs = (1 - self.known_svo_combinations.to(torch.int)) * t4_raw_joint_probs
        t4_unknown_joint_probs_sum = t4_unknown_joint_probs.sum()
        if t4_unknown_joint_probs_sum.item() == 0:
            t4_conditional_joint_probs = t4_unknown_joint_probs
        else:
            t4_conditional_joint_probs = t4_unknown_joint_probs / t4_unknown_joint_probs_sum
        t4_joint_probs = t4_conditional_joint_probs * p_type[4]
        
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
        
        type_normalizer = p_type[1] + p_type[2] + p_type[3] + p_type[4]
        
        if type_normalizer > 0:
            t1_joint_probs /= type_normalizer
            t2_joint_probs /= type_normalizer
            t3_joint_probs /= type_normalizer
            t4_joint_probs /= type_normalizer
        
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

        return example_predictions

    def _case_2(self, subject_probs, verb_probs, p_type, k):
        t1_raw_joint_probs = verb_probs
        t1_joint_probs = t1_raw_joint_probs * p_type[1]
        
        t2_raw_joint_probs = subject_probs
        # Type 2 and 5 instances are indistinguishable in case 2; a tuple of
        # the form <S, 0, None> can be either type 2 or type 5. Thus, its
        # probability of being correct is
        #   P(Correct | valid) * P(valid)
        # = P(Correct | valid) * P(type = 2 or type = 5)
        # = P(Correct | valid) * (P(type = 2) + P(type = 5))
        t2_joint_probs = t2_raw_joint_probs * p_type[2]
        
        # We've computed P(Correct | N_i), but we also have the case information.
        # We need to compute:
        #   P(Correct | N_i, case = 2)
        # = P(Correct | N_i, type in [1, 2, 5])
        # = P(Correct, type in [1, 2, 5] | N_i) / P(type in [1, 2, 5] | N_i)
        ####### Correctness of a novel tuple implies the trial type, so type
        ####### P(Correct, type in [1, 2, 5]) = P(Correct). Type is also
        ####### independent of N_i. So P(type in [1, 2, 5] | N_i) =
        ####### P(type in [1, 2, 5]).
        # = P(Correct | N_i) / P(type in [1, 2, 5])
        # = P(Correct | N_i) / (\sum_{t in [1, 2, 5]} P(type = t)).
        
        # So divide each joint probability by the type normalizer, or sum of
        # probabilities of possible types.
        
        type_normalizer = p_type[1] + p_type[2]
        
        if type_normalizer > 0:
            t1_joint_probs /= type_normalizer
            t2_joint_probs /= type_normalizer
        
        flattened_t1_joint_probs = torch.flatten(t1_joint_probs)
        sorted_t1_joint_probs, sorted_t1_joint_prob_indices = torch.sort(flattened_t1_joint_probs, descending = True)
        
        flattened_t2_joint_probs = torch.flatten(t2_joint_probs)
        sorted_t2_joint_probs, sorted_t2_joint_prob_indices = torch.sort(flattened_t2_joint_probs, descending = True)
        
        example_predictions = []
        for _ in range(k):
            if sorted_t1_joint_probs[0] >= sorted_t2_joint_probs[0]:
                flattened_index = int(sorted_t1_joint_prob_indices[0].item())
                
                verb_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((0, verb_index + 1, -1), sorted_t1_joint_probs[0]))
                
                sorted_t1_joint_probs = sorted_t1_joint_probs[1:]
                sorted_t1_joint_prob_indices = sorted_t1_joint_prob_indices[1:]
                
            else:
                flattened_index = int(sorted_t2_joint_prob_indices[0].item())
                
                subject_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, 0, -1), sorted_t2_joint_probs[0]))
                
                sorted_t2_joint_probs = sorted_t2_joint_probs[1:]
                sorted_t2_joint_prob_indices = sorted_t2_joint_prob_indices[1:]
        
        return example_predictions

    def _case_3(self, p_type):
        return [((-1, 0, torch.tensor(0, dtype = torch.long)), torch.tensor(1.0))]

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

    def __call__(self, spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, p_type):
        predictions = []
        results = {}
        subject_novelty_scores = []
        object_novelty_scores = []
        verb_novelty_scores = []
        p_n_t4 = []
        for idx in range(len(spatial_features)):
            example_spatial_features = spatial_features[idx]
            example_subject_appearance_features = subject_appearance_features[idx]
            example_object_appearance_features = object_appearance_features[idx]
            example_verb_appearance_features = verb_appearance_features[idx]
            
            if example_subject_appearance_features is not None:
                example_subject_features = torch.flatten(example_subject_appearance_features).to(self.device)

                subject_logits, subject_score = self.classifier.predict_score_subject(example_subject_features.unsqueeze(0))
                
                subject_score = subject_score.squeeze(0)

                subject_probs = self.confidence_calibrator.calibrate_subject(subject_logits)
                subject_probs = subject_probs.squeeze(0)

                subject_novelty_scores.append(subject_score)
            else:
                subject_novelty_scores.append(None)

            if example_object_appearance_features is not None:
                example_object_features = torch.flatten(example_object_appearance_features).to(self.device)
                
                object_logits, object_score = self.classifier.predict_score_object(example_object_features.unsqueeze(0))
                
                object_score = object_score.squeeze(0)
                
                object_probs = self.confidence_calibrator.calibrate_object(object_logits)
                object_probs = object_probs.squeeze(0)
                
                object_novelty_scores.append(object_score)
            else:
                object_novelty_scores.append(None)
            
            if example_verb_appearance_features is not None:
                example_verb_features = torch.cat((torch.flatten(example_spatial_features), torch.flatten(example_verb_appearance_features))).to(self.device)
                
                verb_logits, verb_score = self.classifier.predict_score_verb(example_verb_features.unsqueeze(0))
                
                verb_score = verb_score.squeeze(0)

                verb_probs = self.confidence_calibrator.calibrate_verb(verb_logits)
                verb_probs = verb_probs.squeeze(0)
                
                verb_novelty_scores.append(verb_score)
            else:
                verb_novelty_scores.append(None)
            
            p_ni_t4 = torch.tensor(0, dtype = torch.float)
            p_ni_t4 = p_ni_t4.to(self.device)
            if example_subject_appearance_features is not None and example_object_appearance_features is not None:
                # Case 1, S/V/O
                example_predictions = self._case_1(subject_probs, object_probs, verb_probs, p_type, 3)
                
                t4_raw_joint_probs = (subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)).unsqueeze(2) * object_probs.unsqueeze(0).unsqueeze(1)
                t4_unknown_joint_probs = (1 - self.known_svo_combinations.to(torch.int)) * t4_raw_joint_probs
                p_ni_t4 = t4_unknown_joint_probs.sum()
            elif example_subject_appearance_features is not None and example_object_appearance_features is None:
                # Case 2, S/V/None
                example_predictions = self._case_2(subject_probs, verb_probs, p_type, 3)
            elif example_subject_appearance_features is None and example_object_appearance_features is not None:
                # Case 3, None/None/O
                example_predictions = self._case_3(p_type)
            else:
                return NotImplemented
            
            predictions.append(example_predictions)
            p_n_t4.append(p_ni_t4)
        
        p_n_t4 = torch.stack(p_n_t4, dim = 0)

        assert len(subject_novelty_scores) == len(verb_novelty_scores) == len(object_novelty_scores)

        results['subject_novelty_score'] = subject_novelty_scores
        results['verb_novelty_score'] = verb_novelty_scores
        results['object_novelty_score'] = object_novelty_scores
        results['p_n_t4'] = p_n_t4
        results['top3'] = predictions

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
