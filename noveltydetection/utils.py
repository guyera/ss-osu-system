import torch
import sklearn.metrics
import sklearn.neighbors
import math
import sys
from enum import Enum

from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from backbone import Backbone

def compute_partial_auc(nominal_scores, novel_scores):
    nominal_trues = torch.zeros_like(nominal_scores)
    novel_trues = torch.ones_like(novel_scores)

    trues = torch.cat((nominal_trues, novel_trues), dim = 0)\
        .data.cpu().numpy()
    scores = torch.cat((nominal_scores, novel_scores), dim = 0)\
        .data.cpu().numpy()

    auc = sklearn.metrics.roc_auc_score(trues, scores, max_fpr = 0.25)

    return auc


class ActivationStatisticalModel:
    _bandwidths = {
        Backbone.Architecture.swin_t: 50,
        Backbone.Architecture.resnet50: 5
    }
    def __init__(self, backbone_architecture):
        self._bandwidth = self._bandwidths[backbone_architecture]
        self._kde = KernelDensity(kernel='gaussian', bandwidth=self._bandwidth)
        self._device = None
        self._v = None

    def forward_hook(self, module, inputs, outputs):
        self._features = outputs

    def compute_features(self, backbone, batch):
        handle = backbone.register_feature_hook(self.forward_hook)
        backbone(batch)
        handle.remove()
        return self._features.view(self._features.shape[0], -1)

    def pca_reduce(self, v, features, k):
        return torch.matmul(features, v[:, :k])

    def fit(self, features):
        # Fit PCA
        _, _, v = torch.svd(features)
        self._v = v

        # PCA-project features
        projected_features = self.pca_reduce(v, features, 64)

        # Fit self._kde to pca-projected features
        self._kde.fit(projected_features.cpu().numpy())

    def score(self, features):
        # Compute and return negative log likelihood under self._kde
        projected_features = self.pca_reduce(self._v, features, 64)
        return -self._kde.score_samples(projected_features.cpu().numpy())

    def reset(self):
        self._v = None
        self._kde = KernelDensity(kernel='gaussian', bandwidth=self._bandwidth)

    def to(self, device):
        self._device = device
        if self._v is not None:
            self._v = self._v.to(device)
        return self

    def state_dict(self):
        sd = {}
        if self._v is not None:
            sd['v'] = self._v.to('cpu')
        else:
            sd['v'] = None
        sd['kde'] = self._kde
        return sd

    def load_state_dict(self, sd):
        self._v = sd['v']
        self._kde = sd['kde']
        if self._device is not None and self._v is not None:
            self._v = self._v.to(self._device)

class Case1LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
        self.reset()

    def reset(self):
        self.fc = torch.nn.Linear(4, 6).to(self.device)
        self.mean = torch.nn.Parameter(
            torch.zeros(4, device=self.device),
            requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.ones(4, device=self.device),
            requires_grad=False
        )

    def forward(self, x):
        normalized = (x - self.mean) / self.std
        normalized[torch.isnan(normalized)] = 0
        return self.fc(normalized)

    def fit_standardization_statistics(self, scores):
        self.mean[:] = scores.mean(dim=0).detach()
        self.std[:] = scores.std(dim=0).detach()

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def state_dict(self):
        raw_state_dict = super().state_dict()
        return {k: v.cpu() for k, v in raw_state_dict.items()}

    def load_state_dict(self, cpu_state_dict):
        moved_state_dict = {
            k: v.to(self.device) for k, v in cpu_state_dict.items()
        }
        super().load_state_dict(moved_state_dict)

class Case2LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
        self.reset()

    def reset(self):
        self.fc = torch.nn.Linear(3, 5).to(self.device)
        self.mean = torch.nn.Parameter(
            torch.zeros(3, device=self.device),
            requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.ones(3, device=self.device),
            requires_grad=False
        )

    def forward(self, x):
        normalized = (x - self.mean) / self.std
        normalized[torch.isnan(normalized)] = 0
        return self.fc(normalized)

    def fit_standardization_statistics(self, scores):
        self.mean[:] = scores.mean(dim=0).detach()
        self.std[:] = scores.std(dim=0).detach()

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def state_dict(self):
        raw_state_dict = super().state_dict()
        return {k: v.cpu() for k, v in raw_state_dict.items()}

    def load_state_dict(self, cpu_state_dict):
        moved_state_dict = {
            k: v.to(self.device) for k, v in cpu_state_dict.items()
        }
        super().load_state_dict(moved_state_dict)

class Case3LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = 'cpu'
        self.reset()

    def reset(self):
        self.fc = torch.nn.Linear(2, 4).to(self.device)
        self.mean = torch.nn.Parameter(
            torch.zeros(2, device=self.device),
            requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.ones(2, device=self.device),
            requires_grad=False
        )

    def forward(self, x):
        normalized = (x - self.mean) / self.std
        normalized[torch.isnan(normalized)] = 0
        return self.fc(normalized)

    def fit_standardization_statistics(self, scores):
        self.mean[:] = scores.mean(dim=0).detach()
        self.std[:] = scores.std(dim=0).detach()

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def state_dict(self):
        raw_state_dict = super().state_dict()
        return {k: v.cpu() for k, v in raw_state_dict.items()}

    def load_state_dict(self, cpu_state_dict):
        moved_state_dict = {
            k: v.to(self.device) for k, v in cpu_state_dict.items()
        }
        super().load_state_dict(moved_state_dict)

# If p_t4 is None, then it sets p_type[3] to zero and forfeits type 4 novelty.
def compute_probability_novelty(
        subject_scores,
        verb_scores,
        object_scores,
        activation_statistical_scores,
        case_1_logistic_regression,
        case_2_logistic_regression,
        case_3_logistic_regression,
        ignore_t2_in_pni = True,
        p_t4 = None,
        hint_a = None,
        hint_b = None):
    '''
    Parameters:
        subject_scores: List of N scalars, some of which may be None
            Subject novelty scores (negative max logits from the subject box
            classifier, returned by
            UnsupervisedNoveltyDetector.scores_and_p_t4()
        verb_scores: List of N scalars, some of which may be None
            Subject novelty scores (negative max logits from the verb box
            classifier, returned by
            UnsupervisedNoveltyDetector.scores_and_p_t4()
        object_scores: List of N scalars, some of which may be None
            Subject novelty scores (negative max logits from the object box
            classifier, returned by
            UnsupervisedNoveltyDetector.scores_and_p_t4()
        activation_statistical_scores: numpy.ndarray of shape [N]
            Whole-image novelty scores computed from a statistical model fit to
            PCA projections of early-layer activations extracted from whole
            images. Returned by ActivationStatisticalModel.score()
        case_1_logistic_regression: Case1LogisticRegression
            Predicts novelty type probability tensor for case 1 instances
        case_2_logistic_regression: Case2LogisticRegression
            Predicts novelty type probability tensor for case 2 instances
        case_3_logistic_regression: Case3LogisticRegression
            Predicts novelty type probability tensor for case 3 instances
        ignore_t2_in_pni: bool
            Specifies whether the type 2 (novel verb) probability should be
            ignored when computing p_n -- a hack for preventing false alarms
            when type 0 is frequently confused for type 2
        p_t4: Tensor of shape [N] or None
            p_t4[i] represents P(type = 4 | type = 0 or 4) for image i.
            Returned by UnsupervisedNoveltyDetector.scores_and_p_t4(). If None,
            type 4 probabilities are set to zero.
        hint_a: int in {0, 1, 2, 3, 4, 5, 6, 7} or None
            Specifies the trial-level novelty type. 0 means the trial doesn't
            contain any novelty. The rest correspond to their respective
            novelty types. If None, then no hint is specified.
        hint_b: Boolean tensor of shape [N] or None
            hint_b[i] specifies whether image i is novel (True) or non-novel
            (False).
    '''
    p_type = []
    p_n = []
    separated_p_type = []
       
    for idx in range(len(subject_scores)):
        subject_score = subject_scores[idx]
        object_score = object_scores[idx]
        verb_score = verb_scores[idx]
        activation_statistical_score = torch.from_numpy(activation_statistical_scores[idx])
        
        if hint_b is not None:
            cur_hint_b = hint_b[idx]
        else:
            cur_hint_b = None
        if subject_score is not None:
            device = subject_score.device
        else:
            device = object_score.device

        if p_t4 is None:
            cur_cond_p_t4 = 0
        else:
            cur_cond_p_t4 = p_t4[idx]
        
        zero_out = False
        possible_nov_types = torch.ones(7, dtype=torch.bool, device=device)
        # Rule out novelty types that aren't allowed in phase 3
        possible_nov_types[1] = False
        possible_nov_types[2] = False
        possible_nov_types[4] = False
        # TODO Any others? And are these ones correct?
        if cur_hint_b is not None and not cur_hint_b:
            possible_nov_types[1:] = False  
            cur_p_type = possible_nov_types.to(torch.float)
            
        else:
            if cur_hint_b is not None and cur_hint_b:
                possible_nov_types[0] = False
            
            if hint_a is not None:
                if hint_a in [6, 7]:
                    nov_type_idx = hint_a - 1
                elif hint_a == 5:
                    nov_type_idx = 0 # TODO maybe set this to 4 instead?
                else:
                    nov_type_idx = hint_a
                mask = torch.ones_like(possible_nov_types)
                mask[0] = False
                mask[nov_type_idx] = False
                possible_nov_types[mask] = False
                
            n_possible_types = possible_nov_types.to(torch.int).sum()
            assert n_possible_types > 0
            if n_possible_types == 1:
                cur_p_type = possible_nov_types.to(torch.float)
            elif subject_score is not None and object_score is not None:
                # Case 1                
                if not torch.any(possible_nov_types[[1, 2, 3, 5, 6]]):
                    cur_p_type = torch.zeros(7, dtype=torch.float, device=device)
                    cur_p_type[0] = 1 - cur_cond_p_t4
                    cur_p_type[4] = cur_cond_p_t4
                else:                    
                    scores = torch.stack((subject_score, verb_score, object_score, activation_statistical_score.to(torch.float).to(device)), dim = 0).unsqueeze(0)
                    logits = case_1_logistic_regression(scores).squeeze(0)
                    softmax = torch.nn.functional.softmax(logits, dim = 0)
                    cur_p_t0 = (1 - cur_cond_p_t4) * softmax[0]
                    cur_p_t4 = cur_cond_p_t4 * softmax[0]
                    cur_p_type = torch.cat(
                        (
                            cur_p_t0.unsqueeze(0),
                            softmax[1:-2],
                            cur_p_t4.unsqueeze(0),
                            softmax[-2:]
                        ),
                        dim=0
                    )
                    zero_out = True
            elif subject_score is not None:
                # Case 2
                possible_nov_types[3] = False
                n_possible_types = possible_nov_types.to(torch.int).sum()
                assert n_possible_types > 0                
                if n_possible_types == 1:
                    cur_p_type = possible_nov_types.to(torch.float)
                elif not torch.any(possible_nov_types[[1, 2, 3, 5, 6]]):
                    cur_p_type = torch.zeros(7, dtype=torch.float, device=device)
                    cur_p_type[0] = 1 - cur_cond_p_t4
                    cur_p_type[4] = cur_cond_p_t4
                else:
                    
                    scores = torch.stack((subject_score, verb_score, activation_statistical_score.to(torch.float).to(device)), dim = 0).unsqueeze(0)
                    logits = case_2_logistic_regression(scores).squeeze(0)
                    softmax = torch.nn.functional.softmax(logits, dim = 0)
                    cur_p_t0 = (1 - cur_cond_p_t4) * softmax[0]
                    cur_p_t4 = cur_cond_p_t4 * softmax[0]
                    cur_p_type = torch.cat(
                        (
                            cur_p_t0.unsqueeze(0),
                            softmax[1:-2],
                            torch.zeros(1, dtype=torch.float, device=device),
                            cur_p_t4.unsqueeze(0),
                            softmax[-2:]
                        ),
                        dim=0
                    )
                    zero_out = True
            else:
                # Case 3
                possible_nov_types[[1, 2, 4]] = False
                n_possible_types = possible_nov_types.to(torch.int).sum()
                assert n_possible_types > 0
                if n_possible_types == 1:
                    cur_p_type = possible_nov_types.to(torch.float)
                else:
                    scores = torch.stack((object_score, activation_statistical_score.to(torch.float).to(object_score.device)), dim = 0).unsqueeze(0)
                    logits = case_3_logistic_regression(scores).squeeze(0)
                    softmax = torch.nn.functional.softmax(logits, dim = 0)
                    cur_p_type = torch.cat(
                        (
                            softmax[0:1],
                            torch.zeros(2, dtype=torch.float, device=device),
                            softmax[1:-2],
                            torch.zeros(1, dtype=torch.float, device=device),
                            softmax[-2:]
                        ),
                        dim=0
                    )
                    zero_out = True

        # Some impossible novelty types still have predicted non-zero
        # probabilities associated with them; zero them out and renormalize
        # the predictions
        if zero_out:
            cur_p_type[~possible_nov_types] = 0.0
            normalizer = cur_p_type.sum()
            if normalizer == 0:
                # All of the possible types were assigned probabilities of
                # zero; set them all to 1 / K, where K is the number of
                # possible novelty types
                cur_p_type = possible_nov_types.to(torch.float)
                cur_p_type = cur_p_type / cur_p_type.sum()
            else:
                cur_p_type = cur_p_type / normalizer
        # Normalize the NOVEL novelty types; i.e., types 1, 2, 3, 4, 6, and 7.
        # This normalized partial p-type vector represents the probability
        # of each novelty type given that some novelty is present. This is
        # what's passed to the novel tuple classifier (after combining types
        # 6/7 by adding their probabilities together)
        normalizer = cur_p_type[1:].sum()
        
        if normalizer == 0:
            cur_novel_p_type = possible_nov_types[1:].to(torch.float)
            normalizer = cur_novel_p_type.sum()
            if normalizer == 0:
                cur_novel_p_type = torch.full_like(cur_p_type[1:], 1.0 / len(cur_p_type[1:]))
            else:
                cur_novel_p_type = cur_novel_p_type / normalizer
        else:
            cur_novel_p_type = cur_p_type[1:] / normalizer

        # Combine type 6/7 by adding their probabilities together
        cur_combined_novel_p_type = torch.cat(
            (
                cur_novel_p_type[0:-2],
                cur_novel_p_type[-2:-1] + cur_novel_p_type[-1:]
            ),
            dim=0
        )

        # Append the 6/7 combined vector to p_type (it's the main per-image
        # p-type vector used in all computations)
        p_type.append(cur_combined_novel_p_type)

        # But we also need to keep track of the 6/7 separated vectors because
        # the characterization output must treat them separately.
        separated_p_type.append(cur_novel_p_type)

        # P(novel) = P(type = 1, 2, 3, 4, 6, or 7), or the sum of these
        # probabilities. Alternatively, 1 - P(type = 0)
        cur_p_n = 1.0 - cur_p_type[0]
        p_n.append(cur_p_n)

    p_type = torch.stack(p_type, dim = 0)
    p_n = torch.stack(p_n, dim = 0)
    separated_p_type = torch.stack(separated_p_type, dim = 0)

    # print('Given Hint', hint_a)
    # print('p_n', p_n)
    # print('p_type', p_type)
    # print('separated_p_type', separated_p_type)
    # import ipdb; ipdb.set_trace()
    
    return p_type, p_n, separated_p_type
