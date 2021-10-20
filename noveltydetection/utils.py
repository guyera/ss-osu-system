import torch
import sklearn.metrics
import sklearn.neighbors
import math
import sys

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
        if nominal_scores.device != novel_scores.device:
            print(('Warning: nominal_scores and novel_scores are on '
                'different devices. Moving novel_scores to '
                'nominal_scores.device.'),
                file = sys.stderr)
            self.novel_scores = self.novel_scores.to(nominal_scores.device)

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

    def _compute_bandwidth(self, scores):
        std = torch.std(scores, dim = 0, unbiased = True)
        q3 = torch.quantile(scores, 0.75, dim = 0)
        q1 = torch.quantile(scores, 0.25, dim = 0)
        iqr = q3 - q1
        a = torch.min(std, iqr / 1.34)
        bw = 0.9 * a * math.pow(len(scores), -0.2)
        return bw
        
    
    def _nominal_bandwidth(self):
        return self._compute_bandwidth(self.nominal_scores)

    def _novel_bandwidth(self):
        return self._compute_bandwidth(self.novel_scores)
    
    def fit_kdes(self):
        nominal_kde = sklearn.neighbors.KernelDensity(kernel = 'gaussian', bandwidth = self._nominal_bandwidth())
        nominal_kde.fit(self.nominal_scores.unsqueeze(1).detach().cpu().numpy())
        
        novel_kde = sklearn.neighbors.KernelDensity(kernel = 'gaussian', bandwidth = self._novel_bandwidth())
        novel_kde.fit(self.novel_scores.unsqueeze(1).detach().cpu().numpy())

        return nominal_kde, novel_kde

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
        object_score_ctx,
        p_type):
    nominal_subject_kde, novel_subject_kde = subject_score_ctx.fit_kdes()
    nominal_object_kde, novel_object_kde = object_score_ctx.fit_kdes()
    nominal_verb_kde, novel_verb_kde = verb_score_ctx.fit_kdes()
    
    novel_subject_probs = []
    novel_object_probs = []
    novel_verb_probs = []
    case_2 = []
    for score in subject_scores:
        if score is None:
            novel_subject_probs.append(torch.tensor(0, device = p_type.device))
            continue
        
        nominal_log_prob = nominal_subject_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        novel_log_prob = novel_subject_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )

        log_probs = torch.tensor(
            [nominal_log_prob, novel_log_prob],
            device = p_type.device
        )
        novel_prob = torch.nn.functional.softmax(log_probs, dim = 0)[1]
        novel_subject_probs.append(novel_prob)
    novel_subject_probs = torch.stack(novel_subject_probs, dim = 0)

    for score in object_scores:
        if score is None:
            novel_object_probs.append(torch.tensor(0, device = p_type.device))
            # Object box is missing, so case is 2, and novel t5 instance is
            # possible.
            case_2.append(torch.tensor(1, device = p_type.device))
            continue
        
        # Object box present. Novel t5 instance is impossible in anything
        # but case 2 (object box must be missing).
        case_2.append(torch.tensor(0, device = p_type.device))
        
        nominal_log_prob = nominal_object_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        novel_log_prob = novel_object_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )

        log_probs = torch.tensor(
            [nominal_log_prob, novel_log_prob],
            device = p_type.device
        )
        novel_prob = torch.nn.functional.softmax(log_probs, dim = 0)[1]
        novel_object_probs.append(novel_prob)
    novel_object_probs = torch.stack(novel_object_probs, dim = 0)
    case_2 = torch.stack(case_2, dim = 0)
    
    for score in verb_scores:
        if score is None:
            novel_verb_probs.append(torch.tensor(0, device = p_type.device))
            continue
        
        nominal_log_prob = nominal_verb_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        novel_log_prob = novel_verb_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )

        log_probs = torch.tensor(
            [nominal_log_prob, novel_log_prob],
            device = p_type.device
        )
        novel_prob = torch.nn.functional.softmax(log_probs, dim = 0)[1]
        novel_verb_probs.append(novel_prob)
    novel_verb_probs = torch.stack(novel_verb_probs, dim = 0)
    
    # P(subject is novel | subject can be novel) * P(subject exists) * P(type = 1)
    p_novel_subject = novel_subject_probs * p_type[0]
    # P(object is novel | object can be novel) * P(object exists) * P(type = 3)
    p_novel_object = novel_object_probs * p_type[2]
    # P(verb is novel | verb can be novel) * P(verb exists) * (P(type = 2) + P(type = 5) * P(case = 2))
    p_novel_verb = novel_verb_probs * (p_type[1] + p_type[4] * case_2)
    # P(combination is novel | combination can be novel) * P(all boxes exists) * P(type = 4)
    p_novel_combination = p_n_t4 * p_type[3]
    
    p_novel = p_novel_subject + p_novel_object + p_novel_verb + p_novel_combination
    
    return p_novel
    # return 0.5
