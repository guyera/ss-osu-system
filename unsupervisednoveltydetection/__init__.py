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

        self.known_combinations = torch.zeros(num_subj_cls - 1, num_obj_cls - 1, num_action_cls - 1, dtype = torch.bool)
    
    def to(self, device):
        self.device = device
        self.classifier = self.classifier.to(device)
        self.confidence_calibrator = self.confidence_calibrator.to(device)
        self.known_combinations = self.known_combinations.to(device)
        return self

    def load_state_dict(self, state_dict):
        classifier_state_dict = state_dict['classifier']
        confidence_calibrator_state_dict = state_dict['confidence_calibrator']
        known_combinations = state_dict['known_combinations']

        self.classifier.load_state_dict(classifier_state_dict)
        self.confidence_calibrator.load_state_dict(confidence_calibrator_state_dict)
        
        self.known_combinations[:] = False
        subject_indices = []
        object_indices = []
        verb_indices = []
        for subject_index, object_index, verb_index in known_combinations:
            subject_indices.append(subject_index)
            object_indices.append(object_index)
            verb_indices.append(verb_index)
        self.known_combinations[(subject_indices, object_indices, verb_indices)] = True
    
    def _case_1(self, subject_probs, object_probs, verb_probs, p_type, k):
        t1_known_combinations = self.known_combinations.to(torch.int).sum(dim = 0) > 0
        t1_raw_joint_probs = object_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        t1_known_joint_probs = t1_known_combinations.to(torch.int) * t1_raw_joint_probs
        t1_known_joint_probs_sum = t1_known_joint_probs.sum()
        if t1_known_joint_probs_sum.item() == 0:
            t1_conditional_joint_probs = t1_known_joint_probs
        else:
            t1_conditional_joint_probs = t1_known_joint_probs / (t1_known_joint_probs_sum)
        t1_joint_probs = t1_conditional_joint_probs * p_type[0]
        
        t2_known_combinations = self.known_combinations.to(torch.int).sum(dim = 2) > 0
        t2_raw_joint_probs = subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)
        t2_known_joint_probs = t2_known_combinations.to(torch.int) * t2_raw_joint_probs
        t2_known_joint_probs_sum = t2_known_joint_probs.sum()
        if t2_known_joint_probs_sum.item() == 0:
            t2_conditional_joint_probs = t2_known_joint_probs
        else:
            t2_conditional_joint_probs = t2_known_joint_probs / (t2_known_joint_probs_sum)
        t2_joint_probs = t2_conditional_joint_probs * p_type[1]
        
        t3_known_combinations = self.known_combinations.to(torch.int).sum(dim = 1) > 0
        t3_raw_joint_probs = subject_probs.unsqueeze(1) * verb_probs.unsqueeze(0)
        t3_known_joint_probs = t3_known_combinations.to(torch.int) * t3_raw_joint_probs
        t3_known_joint_probs_sum = t3_known_joint_probs.sum()
        if t3_known_joint_probs_sum.item() == 0:
            t3_conditional_joint_probs = t3_known_joint_probs
        else:
            t3_conditional_joint_probs = t3_known_joint_probs / (t3_known_joint_probs_sum)
        t3_joint_probs = t3_conditional_joint_probs * p_type[2]
        
        t4_raw_joint_probs = (subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)).unsqueeze(2) * verb_probs.unsqueeze(0).unsqueeze(1)
        t4_unknown_joint_probs = (1 - self.known_combinations.to(torch.int)) * t4_raw_joint_probs
        t4_unknown_joint_probs_sum = t4_unknown_joint_probs.sum()
        if t4_unknown_joint_probs_sum.item() == 0:
            t4_conditional_joint_probs = t4_unknown_joint_probs
        else:
            t4_conditional_joint_probs = t4_unknown_joint_probs / (t4_unknown_joint_probs_sum)
        t4_joint_probs = t4_conditional_joint_probs * p_type[3]
        
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
        
        type_normalizer = p_type[0] + p_type[1] + p_type[2] + p_type[3]
        
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
                
                object_skip_interval = verb_probs.shape[0]

                object_index = flattened_index // object_skip_interval
                object_skip_total = object_index * object_skip_interval
                flattened_index -= object_skip_total
                
                verb_index = flattened_index
                
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
                
                subject_skip_interval = object_probs.shape[0] * verb_probs.shape[0]
                object_skip_interval = verb_probs.shape[0]

                subject_index = flattened_index // subject_skip_interval
                subject_skip_total = subject_index * subject_skip_interval
                flattened_index -= subject_skip_total
                
                object_index = flattened_index // object_skip_interval
                object_skip_total = object_index * object_skip_interval
                flattened_index -= object_skip_total
                
                verb_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, verb_index + 1, object_index + 1), sorted_t4_joint_probs[0]))
                
                sorted_t4_joint_probs = sorted_t4_joint_probs[1:]
                sorted_t4_joint_prob_indices = sorted_t4_joint_prob_indices[1:]

        return example_predictions

    def _case_2(self, subject_probs, verb_probs, p_type, k):
        t1_raw_joint_probs = verb_probs
        t1_joint_probs = t1_raw_joint_probs * p_type[0]
        
        t2_5_raw_joint_probs = subject_probs
        # Type 2 and 5 instances are indistinguishable in case 2; a tuple of
        # the form <S, 0, None> can be either type 2 or type 5. Thus, its
        # probability of being correct is
        #   P(Correct | valid) * P(valid)
        # = P(Correct | valid) * P(type = 2 or type = 5)
        # = P(Correct | valid) * (P(type = 2) + P(type = 5))
        t2_5_joint_probs = t2_5_raw_joint_probs * (p_type[1] + p_type[4])
        
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
        
        type_normalizer = p_type[0] + p_type[1] + p_type[4]
        
        t1_joint_probs /= type_normalizer
        t2_5_joint_probs /= type_normalizer
        
        flattened_t1_joint_probs = torch.flatten(t1_joint_probs)
        sorted_t1_joint_probs, sorted_t1_joint_prob_indices = torch.sort(flattened_t1_joint_probs, descending = True)
        
        flattened_t2_5_joint_probs = torch.flatten(t2_5_joint_probs)
        sorted_t2_5_joint_probs, sorted_t2_5_joint_prob_indices = torch.sort(flattened_t2_5_joint_probs, descending = True)
        
        example_predictions = []
        for _ in range(k):
            if sorted_t1_joint_probs[0] >= sorted_t2_5_joint_probs[0]:
                flattened_index = int(sorted_t1_joint_prob_indices[0].item())
                
                verb_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((0, verb_index + 1, None), sorted_t1_joint_probs[0]))
                
                sorted_t1_joint_probs = sorted_t1_joint_probs[1:]
                sorted_t1_joint_prob_indices = sorted_t1_joint_prob_indices[1:]
                
            else:
                flattened_index = int(sorted_t2_5_joint_prob_indices[0].item())
                
                subject_index = flattened_index
                
                # Shift labels forward, to allow for anomaly = 0
                example_predictions.append(((subject_index + 1, 0, None), sorted_t2_5_joint_probs[0]))
                
                sorted_t2_5_joint_probs = sorted_t2_5_joint_probs[1:]
                sorted_t2_5_joint_prob_indices = sorted_t2_5_joint_prob_indices[1:]
                
        return example_predictions

    def _case_3(self, p_type):
        return [((None, None, torch.tensor(0, dtype = torch.long)), torch.tensor(1.0))]

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
                object_logits, object_scores = self.classifier.predict_score_object(example_object_features.unsqueeze(0))
                object_probs = self.confidence_calibrator.calibrate_object(object_logits)
                object_probs = object_probs.squeeze(0)
                object_novelty_scores.append(object_scores)
            else:
                object_novelty_scores.append(None)
            
            if example_verb_appearance_features is not None:
                example_verb_features = torch.cat((torch.flatten(example_spatial_features), torch.flatten(example_verb_appearance_features)))
                example_verb_features = example_verb_features.to(self.device)
                verb_logits, verb_scores = self.classifier.predict_score_verb(example_verb_features.unsqueeze(0))
                verb_probs = self.confidence_calibrator.calibrate_verb(verb_logits)
                verb_probs = verb_probs.squeeze(0)
                verb_novelty_scores.append(verb_scores)
            else:
                verb_novelty_scores.append(None)
            
            p_ni_t4 = torch.tensor(0) 
            p_ni_t4 = p_ni_t4.to(subject_probs.device)
            if example_subject_appearance_features is not None and example_object_appearance_features is not None:
                # Case 1, S/V/O
                example_predictions = self._case_1(subject_probs, object_probs, verb_probs, p_type, 3)
                
                t4_raw_joint_probs = (subject_probs.unsqueeze(1) * object_probs.unsqueeze(0)).unsqueeze(2) * verb_probs.unsqueeze(0).unsqueeze(1)
                t4_unknown_joint_probs = (1 - self.known_combinations.to(torch.int)) * t4_raw_joint_probs
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

        results['subject_novelty_score'] = subject_novelty_scores
        results['verb_novelty_score'] = verb_novelty_scores
        results['object_novelty_score'] = object_novelty_scores
        results['p_n_t4'] = p_n_t4
        results['top3'] = predictions

        return results

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
