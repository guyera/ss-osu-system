import torch
import sklearn.metrics
import sklearn.neighbors
import math
import sys
from enum import Enum

from tqdm import tqdm
from sklearn.neighbors import KernelDensity

from backbone import Backbone
from tupleprediction._tuplepredictor import TuplePredictor
import tupleprediction.training as training

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

class NoveltyTypeClassifier(torch.nn.Module):
    def __init__(self, n_signals):
        super().__init__()
        self.device = 'cpu'
        self.n_signals = n_signals
        self.reset()

    def reset(self):
        self.fc = torch.nn.Linear(self.n_signals, 6).to(self.device)
        self.mean = torch.nn.Parameter(
            torch.zeros(self.n_signals, device=self.device),
            requires_grad=False
        )
        self.std = torch.nn.Parameter(
            torch.ones(self.n_signals, device=self.device),
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

# TODO tensorize
def compute_probability_novelty(
        scores,
        novelty_type_classifier,
        ignore_t2_in_pni = True,
        hint_a = None,
        hint_b = None):
    '''
    Parameters:
        scores: Tensor of shape [N, L], where N is the number of images and
            L is the number of novelty signals.
        novelty_type_classifier: NoveltyTypeClassifier
            Predicts novelty type probability tensor
        ignore_t2_in_pni: bool
            Specifies whether the type 2 (novel verb) probability should be
            ignored when computing p_n -- a hack for preventing false alarms
            when type 0 is frequently confused for type 2
        hint_a: int in {0, 1, 2, 3, 4, 5} or None
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
    device = scores.device

    for idx in range(len(scores)):
        img_scores = scores[idx]
        
        if hint_b is not None:
            cur_hint_b = hint_b[idx]
        else:
            cur_hint_b = None

        possible_nov_types = torch.ones(7, dtype=torch.bool, device=device)
        # Rule out novelty types that aren't allowed in phase 3
        possible_nov_types[1] = False
        possible_nov_types[2] = False
        possible_nov_types[4] = False
        if cur_hint_b is not None and not cur_hint_b:
            possible_nov_types[1:] = False  
            cur_p_type = possible_nov_types.to(torch.float)
        else:
            if cur_hint_b is not None and cur_hint_b:
                possible_nov_types[0] = False

            if hint_a is not None:
                mask = torch.ones_like(possible_nov_types)
                mask[0] = False
                mask[hint_a] = False
                possible_nov_types[mask] = False
                
            n_possible_types = possible_nov_types.to(torch.int).sum()
            assert n_possible_types > 0
            if n_possible_types == 1:
                cur_p_type = possible_nov_types.to(torch.float)
            else:
                logits = \
                    novelty_type_classifier(img_scores.unsqueeze(0)).squeeze(0)
                softmax = torch.nn.functional.softmax(logits, dim = 0)
                cur_p_type = softmax

                # Some impossible novelty types still have predicted non-zero
                # probabilities associated with them; zero them out and renormalize
                # the predictions
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

        # Normalize the NOVEL novelty types; i.e., non-type-0.
        # This normalized partial p-type vector represents the probability
        # of each novelty type given that some novelty is present. This is
        # what's passed to the novel tuple predictor
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

        # Append the 6/7 combined vector to p_type (it's the main per-image
        # p-type vector used in all computations)
        p_type.append(cur_p_type)

        # P(novel) = P(type = 1, 2, 3, 4, 6, or 7), or the sum of these
        # probabilities. Alternatively, 1 - P(type = 0)
        cur_p_n = 1.0 - cur_p_type[0]
        p_n.append(cur_p_n)

    p_type = torch.stack(p_type, dim = 0)
    p_n = torch.stack(p_n, dim = 0)

    return p_type, p_n
