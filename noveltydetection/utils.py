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

    def data_available(self):
        return self.nominal_scores is not None and self.novel_scores is not None

    def compute_partial_auc(self):
        return compute_partial_auc(self.nominal_scores, self.novel_scores)

    # Use silverman's rule of thumb to choose bandwidth.
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

    def get_mean_scores(self):
        return torch.mean(self.nominal_scores), torch.mean(self.novel_scores)

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
        p_known_svo,
        p_known_sv,
        p_known_so,
        p_known_vo,
        subject_score_ctx,
        verb_score_ctx,
        object_score_ctx):
    if not subject_score_ctx.data_available() or\
            not verb_score_ctx.data_available() or\
            not object_score_ctx.data_available():
        # Missing nominal or novel data for one or more KDEs. Return 0.5.
        return 0.5
    
    nominal_subject_kde, novel_subject_kde = subject_score_ctx.fit_kdes()
    nominal_object_kde, novel_object_kde = object_score_ctx.fit_kdes()
    nominal_verb_kde, novel_verb_kde = verb_score_ctx.fit_kdes()
    
    novel_subject_probs = []
    subject_boxes_present = []
    mean_nominal_subject_score, mean_novel_subject_score = subject_score_ctx.get_mean_scores()
    for score in subject_scores:
        if score is None:
            novel_subject_probs.append(torch.tensor(0.0, device = p_known_svo.device))
            subject_boxes_present.append(torch.tensor(0.0, device = p_known_svo.device))
            continue
        
        nominal_log_prob = nominal_subject_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        novel_log_prob = novel_subject_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )

        if nominal_log_prob == 0 and novel_log_prob == 0:
            # Both probabilities are 0. The score is in a low-probability
            # region. If it's below the mean nominal score, then it's very
            # nominal, and the output should be zero. If it's above the mean
            # novel score, then it's very novel, and the output should be 1.
            # Otherwise, it's in between the two score distributions, and we
            # should output 0.5.
            if score < mean_nominal_subject_score:
                novel_prob = torch.tensor(0.0, device = p_known_svo.device)
            elif score > mean_novel_subject_score:
                novel_prob = torch.tensor(1.0, device = p_known_svo.device)
            else:
                novel_prob = torch.tensor(0.5, device = p_known_svo.device)
        else:
            # We need to compute the probability densities from the log probs
            # (by exponentiation), and then divide the novel prob by the sum
            # of the two probs. This is equivalent to the novel component of
            # the softmax of the two log probs.
            log_probs = torch.tensor(
                [nominal_log_prob, novel_log_prob],
                device = p_known_svo.device
            )
            novel_prob = torch.nn.functional.softmax(log_probs, dim = 0)[1]
        
        novel_subject_probs.append(novel_prob)
        subject_boxes_present.append(torch.tensor(1.0, device = p_known_svo.device))
    
    novel_subject_probs = torch.stack(novel_subject_probs, dim = 0)
    subject_boxes_present = torch.stack(subject_boxes_present, dim = 0)

    novel_object_probs = []
    object_boxes_present = []
    mean_nominal_object_score, mean_novel_object_score = object_score_ctx.get_mean_scores()
    for score in object_scores:
        if score is None:
            novel_object_probs.append(torch.tensor(0.0, device = p_known_svo.device))
            object_boxes_present.append(torch.tensor(0.0, device = p_known_svo.device))
            continue
        
        nominal_log_prob = nominal_object_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        novel_log_prob = novel_object_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )

        if nominal_log_prob == 0 and novel_log_prob == 0:
            # Both probabilities are 0. The score is in a low-probability
            # region. If it's below the mean nominal score, then it's very
            # nominal, and the output should be zero. If it's above the mean
            # novel score, then it's very novel, and the output should be 1.
            # Otherwise, it's in between the two score distributions, and we
            # should output 0.5.
            if score < mean_nominal_object_score:
                novel_prob = torch.tensor(0.0, device = p_known_svo.device)
            elif score > mean_novel_object_score:
                novel_prob = torch.tensor(1.0, device = p_known_svo.device)
            else:
                novel_prob = torch.tensor(0.5, device = p_known_svo.device)
        else:
            # We need to compute the probability densities from the log probs
            # (by exponentiation), and then divide the novel prob by the sum
            # of the two probs. This is equivalent to the novel component of
            # the softmax of the two log probs.
            log_probs = torch.tensor(
                [nominal_log_prob, novel_log_prob],
                device = p_known_svo.device
            )
            novel_prob = torch.nn.functional.softmax(log_probs, dim = 0)[1]
        
        novel_object_probs.append(novel_prob)
        object_boxes_present.append(torch.tensor(1.0, device = p_known_svo.device))
    
    novel_object_probs = torch.stack(novel_object_probs, dim = 0)
    object_boxes_present = torch.stack(object_boxes_present, dim = 0)
    
    novel_verb_probs = []
    mean_nominal_verb_score, mean_novel_verb_score = verb_score_ctx.get_mean_scores()
    for score in verb_scores:
        if score is None:
            novel_verb_probs.append(torch.tensor(0.0, device = p_known_svo.device))
            continue
        
        nominal_log_prob = nominal_verb_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        novel_log_prob = novel_verb_kde.score(
            score.unsqueeze(0).unsqueeze(1).detach().cpu().numpy()
        )
        
        if nominal_log_prob == 0 and novel_log_prob == 0:
            # Both probabilities are 0. The score is in a low-probability
            # region. If it's below the mean nominal score, then it's very
            # nominal, and the output should be zero. If it's above the mean
            # novel score, then it's very novel, and the output should be 1.
            # Otherwise, it's in between the two score distributions, and we
            # should output 0.5.
            if score < mean_nominal_verb_score:
                novel_prob = torch.tensor(0.0, device = p_known_svo.device)
            elif score > mean_novel_verb_score:
                novel_prob = torch.tensor(1.0, device = p_known_svo.device)
            else:
                novel_prob = torch.tensor(0.5, device = p_known_svo.device)
        else:
            # We need to compute the probability densities from the log probs
            # (by exponentiation), and then divide the novel prob by the sum
            # of the two probs. This is equivalent to the novel component of
            # the softmax of the two log probs.
            log_probs = torch.tensor(
                [nominal_log_prob, novel_log_prob],
                device = p_known_svo.device
            )
            novel_prob = torch.nn.functional.softmax(log_probs, dim = 0)[1]
        
        novel_verb_probs.append(novel_prob)
    
    novel_verb_probs = torch.stack(novel_verb_probs, dim = 0)
    
    # P(Case = 1)
    p_c1 = subject_boxes_present * object_boxes_present

    # P(Case = 2)
    p_c2 = subject_boxes_present * (1 - object_boxes_present)

    # P(Case = 3)
    p_c3 = (1 - subject_boxes_present) * object_boxes_present

    # For invalid cases, p_known_XXX are all 0. Otherwise, they are non-zero.
    # p_t4 is (1 - p_known_svo) in case 1, (1 - p_known_sv) in case 2,
    # and 0 otherwise. This can be computed as follows:
    p_t4 = p_c1 * (1 - p_known_svo) + p_c2 * (1 - p_known_sv)
    
    # Now compute P_Ni for each i as
    # max(p_t1, p_t2, p_t3, p_t4).
    p_n, max_indices = torch.max(torch.stack((novel_subject_probs, novel_verb_probs, novel_object_probs, p_t4), dim = 1), dim = 1)

    # We want the max novelty probability prior to normalization to also be
    # the max novelty type in p_type after normalization. If the max is less
    # than 0.25, then this won't be the case. So we have to consider two cases.
    # First, consider the case where it is greater than 0.25:
    
    # Compute remainder after accounting for p_Ni for each i, i.e. 1 - p_n
    remainder = 1 - p_n

    # Distribute remainder among the three non-max probabilities
    distributed_remainder = remainder / 3.0

    # Construct p_type as tensor of size [N, 4], full of distributed_remainder values
    p_type_gt = distributed_remainder.unsqueeze(1).repeat(1, 4)
    
    # Set the appropriate value of p_type for each data point to be equal to
    # the corresponding p_ni value.
    data_indices = torch.arange(len(p_n), device = p_known_svo.device)
    # Use tuple for this kind of indexing
    idx_tuple = (data_indices, max_indices)
    p_type_gt[idx_tuple] = p_n

    # Next, we have to consider the case where the greatest novelty type 
    # probability is less than 0.25. In that case, we'll just set p_type to
    # be the uniform distribution, i.e. [0.25, 0.25, 0.25, 0.25]. So let's
    # combine them now:
    gt_mask = (p_n >= 0.25).to(torch.int)
    p_type = gt_mask.unsqueeze(1) * p_type_gt + (1 - gt_mask).unsqueeze(1) * 0.25

    return p_type, p_n

'''
def compute_probability_novelty(
        subject_scores,
        verb_scores,
        object_scores,
        p_n_t4,
        subject_score_ctx,
        verb_score_ctx,
        object_score_ctx,
        p_type):
    if not subject_score_ctx.data_available() or\
            not verb_score_ctx.data_available() or\
            not object_score_ctx.data_available():
        # Missing nominal or novel data for one or more KDEs. Return 0.5.
        return 0.5

    nominal_subject_kde, novel_subject_kde = subject_score_ctx.fit_kdes()
    nominal_object_kde, novel_object_kde = object_score_ctx.fit_kdes()
    nominal_verb_kde, novel_verb_kde = verb_score_ctx.fit_kdes()
    
    novel_subject_probs = []
    novel_object_probs = []
    novel_verb_probs = []
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
            continue
        
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
    p_novel_subject = novel_subject_probs * p_type[1]
    # P(verb is novel | verb can be novel) * P(verb exists) * P(type = 2)
    p_novel_verb = novel_verb_probs * (p_type[2])
    # P(object is novel | object can be novel) * P(object exists) * P(type = 3)
    p_novel_object = novel_object_probs * p_type[3]
    # P(combination is novel | combination can be novel) * P(all boxes exists) * P(type = 4)
    p_novel_combination = p_n_t4 * p_type[4]
    
    p_novel = p_novel_subject + p_novel_verb + p_novel_object + p_novel_combination
    
    return p_novel
'''
