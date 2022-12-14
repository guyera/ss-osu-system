import torch

from unsupervisednoveltydetection._utils import _state_dict, _load_state_dict
from unsupervisednoveltydetection._confidencecalibrator import\
    ConfidenceCalibrator

class ClassifierV2:
    def __init__(
            self,
            bottleneck_dim,
            num_subj_cls,
            num_obj_cls,
            num_action_cls,
            spatial_encoding_dim):
        self.device = 'cpu'
        self.bottleneck_dim = bottleneck_dim
        self.num_subj_cls = num_subj_cls
        self.num_obj_cls = num_obj_cls
        self.num_action_cls = num_action_cls
        self.spatial_encoding_dim = spatial_encoding_dim
        self.reset()

    def reset(self):
        self.subject_classifier = torch.nn.Linear(self.bottleneck_dim, self.num_subj_cls - 1).to(self.device)
        self.object_classifier = torch.nn.Linear(self.bottleneck_dim, self.num_obj_cls - 1).to(self.device)
        self.verb_classifier = torch.nn.Linear(self.bottleneck_dim + self.spatial_encoding_dim, self.num_action_cls - 1).to(self.device)
        self.confidence_calibrator = ConfidenceCalibrator()
    
    def predict(self, subject_features, object_features, verb_features):
        self.subject_classifier.eval()
        self.object_classifier.eval()
        self.verb_classifier.eval()
        subject_logits = self.subject_classifier(subject_features)
        object_logits = self.object_classifier(object_features)
        verb_logits = self.verb_classifier(verb_features)
        return subject_logits, object_logits, verb_logits

    def predict_score(self, spatial_features, subject_box_features, verb_box_features, object_box_features):
        # Ensure equal numbers of all features
        n = len(spatial_features)
        assert len(subject_box_features) == n
        assert len(verb_box_features) == n
        assert len(object_box_features) == n

        self.subject_classifier.eval()
        self.object_classifier.eval()
        self.verb_classifier.eval()

        # Find non-none indices for each feature
        non_none_subject_indices = []
        non_none_verb_indices = []
        non_none_object_indices = []
        for idx in range(n):
            if subject_box_features[idx] is not None:
                non_none_subject_indices.append(idx)
            if verb_box_features[idx] is not None:
                non_none_verb_indices.append(idx)
            if object_box_features[idx] is not None:
                non_none_object_indices.append(idx)

        # Construct feature tensors
        non_none_spatial_features = torch.cat([
            spatial_features[idx] for idx in non_none_verb_indices
        ])
        non_none_subject_box_features = torch.cat([
            subject_box_features[idx] for idx in non_none_subject_indices
        ])
        non_none_verb_box_features = torch.cat([
            verb_box_features[idx] for idx in non_none_verb_indices
        ])
        non_none_object_box_features = torch.cat([
            object_box_features[idx] for idx in non_none_object_indices
        ])
        
        subject_features = torch.flatten(
            non_none_subject_box_features,
            start_dim=1
        )
        verb_features = torch.cat(
            (
                torch.flatten(non_none_spatial_features, start_dim=1),
                torch.flatten(non_none_verb_box_features, start_dim=1)
            ),
            dim=1
        )
        object_features = torch.flatten(
            non_none_object_box_features,
            start_dim=1
        )

        # Make predictions
        subject_logits = self.subject_classifier(subject_features)
        max_subject_logits, _ = torch.max(subject_logits, dim=1)
        subject_scores = -max_subject_logits

        verb_logits = self.verb_classifier(verb_features)
        max_verb_logits, _ = torch.max(verb_logits, dim=1)
        verb_scores = -max_verb_logits

        object_logits = self.object_classifier(object_features)
        max_object_logits, _ = torch.max(object_logits, dim=1)
        object_scores = -max_object_logits

        # Calibrate predictions
        subject_probs, object_probs, verb_probs = \
            self.confidence_calibrator.calibrate(
                subject_logits,
                object_logits,
                verb_logits
            )

        # Expand predictions back to list form, inserting "Nones" where
        # necessary
        final_subject_probs = [None] * n
        final_subject_scores = [None] * n
        for idx, non_none_subject_index in enumerate(non_none_subject_indices):
            final_subject_probs[non_none_subject_index] = subject_probs[idx]
            final_subject_scores[non_none_subject_index] = subject_scores[idx]

        final_verb_probs = [None] * n
        final_verb_scores = [None] * n
        for idx, non_none_verb_index in enumerate(non_none_verb_indices):
            final_verb_probs[non_none_verb_index] = verb_probs[idx]
            final_verb_scores[non_none_verb_index] = verb_scores[idx]

        final_object_probs = [None] * n
        final_object_scores = [None] * n
        for idx, non_none_object_index in enumerate(non_none_object_indices):
            final_object_probs[non_none_object_index] = object_probs[idx]
            final_object_scores[non_none_object_index] = object_scores[idx]

        return final_subject_probs,\
            final_subject_scores,\
            final_verb_probs,\
            final_verb_scores,\
            final_object_probs,\
            final_object_scores

    def predict_score_subject(self, features):
        self.subject_classifier.eval()
        logits = self.subject_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_score_object(self, features):
        self.object_classifier.eval()
        logits = self.object_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_score_verb(self, features):
        self.verb_classifier.eval()
        logits = self.verb_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logits, logit_scores

    def predict_subject(self, features):
        self.subject_classifier.eval()
        logits = self.subject_classifier(features)
        return logits

    def predict_object(self, features):
        self.object_classifier.eval()
        logits = self.object_classifier(features)
        return logits

    def predict_verb(self, features):
        self.verb_classifier.eval()
        logits = self.verb_classifier(features)
        return logits

    def score_subject(self, features):
        self.subject_classifier.eval()
        logits = self.subject_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores

    def score_object(self, features):
        self.object_classifier.eval()
        logits = self.object_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores

    def score_verb(self, features):
        self.verb_classifier.eval()
        logits = self.verb_classifier(features)
        max_logits, _ = torch.max(logits, dim = 1)
        logit_scores = -max_logits
        return logit_scores
    
    def to(self, device):
        self.device = device
        self.subject_classifier = self.subject_classifier.to(device)
        self.object_classifier = self.object_classifier.to(device)
        self.verb_classifier = self.verb_classifier.to(device)
        return self
    
    def state_dict(self):
        state_dict = {}
        state_dict['subject_classifier'] = _state_dict(self.subject_classifier)
        state_dict['object_classifier'] = _state_dict(self.object_classifier)
        state_dict['verb_classifier'] = _state_dict(self.verb_classifier)
        return state_dict

    def load_state_dict(self, state_dict):
        _load_state_dict(
            self.subject_classifier,
            state_dict['subject_classifier']
        )
        _load_state_dict(
            self.object_classifier,
            state_dict['object_classifier']
        )
        _load_state_dict(
            self.verb_classifier,
            state_dict['verb_classifier']
        )
