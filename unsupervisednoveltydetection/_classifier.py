import torch

from unsupervisednoveltydetection._utils import _state_dict, _load_state_dict

class ClassifierV2:
    def __init__(
            self,
            bottleneck_dim,
            num_subj_cls,
            num_obj_cls,
            num_action_cls,
            spatial_encoding_dim):
        self.device = 'cpu'
        self.subject_classifier = torch.nn.Linear(bottleneck_dim, num_subj_cls - 1)
        self.object_classifier = torch.nn.Linear(bottleneck_dim, num_obj_cls - 1)
        self.verb_classifier = torch.nn.Linear(bottleneck_dim + spatial_encoding_dim, num_action_cls - 1)
    
    def predict(self, subject_features, object_features, verb_features):
        self.subject_classifier.eval()
        self.object_classifier.eval()
        self.verb_classifier.eval()
        subject_logits = self.subject_classifier(subject_features)
        object_logits = self.object_classifier(object_features)
        verb_logits = self.verb_classifier(verb_features)
        return subject_logits, object_logits, verb_logits

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
