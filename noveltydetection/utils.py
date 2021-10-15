import torch
import sklearn.metrics

from enum import Enum

def compute_partial_auc(nominal_scores, novel_scores):
    nominal_trues = torch.zeros_like(nominal_scores)
    novel_trues = torch.ones_like(novel_scores)

    trues = torch.cat((nominal_trues, novel_trues), dim = 0)\
        .data.cpu().numpy()
    scores = torch.cat((nominal_scores, novel_scores), dim = 0)\
        .data.cpu().numpy()
    
    auc = sklearn.metrics.roc_auc_score(trues, scores, max_fpr = 0.25)

    return auc

class ScoreContext:
    class Source(Enum):
        UNSUPERVISED = 1
        SUPERVISED = 2
    
    def __init__(self, source, nominal_scores = None, novel_scores = None):
        self.device = 'cpu'
        self.source = source
        self.nominal_scores = nominal_scores
        self.novel_scores = novel_scores

    def add_nominal_scores(self, nominal_scores):
        if self.nominal_scores is None:
            self.nominal_scores = nominal_scores
        else:
            self.nominal_scores =\
                torch.cat((self.nominal_scores, nominal_scores), dim = 0)

    def add_novel_scores(self, novel_scores):
        if self.novel_scores is None:
            self.novel_scores = novel_scores
        else:
            self.novel_scores =\
                torch.cat((self.novel_scores, novel_scores), dim = 0)

    def compute_partial_auc(self):
        return compute_partial_auc(self.nominal_scores, self.novel_scores)

    def state_dict(self):
        state_dict = {}
        state_dict['source'] = self.source
        state_dict['nominal_scores'] = self.nominal_scores.data.cpu()\
            if self.nominal_scores is not None else None
        state_dict['novel_scores'] = self.novel_scores.data.cpu()\
            if self.novel_scores is not None else None
        return state_dict

    def load_state_dict(self, state_dict):
        self.source = state_dict['source']
        self.nominal_scores = state_dict['nominal_scores']
        self.novel_scores = state_dict['novel_scores']
        
        self.nominal_scores = self.nominal_scores.to(self.device)\
            if self.nominal_scores is not None else self.nominal_scores
        self.novel_scores = self.novel_scores.to(self.device)\
            if self.novel_scores is not None else self.novel_scores

    def to(self, device):
        self.device = device
        self.nominal_scores = self.nominal_scores.to(device)
        self.novel_scores = self.novel_scores.to(device)
        return self

def compute_probability_novelty(
        subject_scores,
        verb_scores,
        object_scores,
        p_n_t4,
        subject_score_ctx,
        verb_score_ctx,
        object_score_ctx, p_type):
    # TODO. Will require more data before it actually works well, but post-
    # red-button the scores for supervised feedback novelty can be added to the
    # score contexts, and then kernel density estimation can be used to
    # compute P(N).
    return 0.5
