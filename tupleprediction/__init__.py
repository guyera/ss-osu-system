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

class NoveltyTypeClassifier(torch.nn.Module):
    def __init__(self, n_signals):
        super().__init__()
        self.device = 'cpu'
        self.n_signals = n_signals
        self.reset()

    def reset(self):
        self.bn = torch.nn.BatchNorm1d(self.n_signals)
        self.fc = torch.nn.Linear(self.n_signals, 6).to(self.device)

    def forward(self, x):
        return self.fc(self.bn(x))

    def to(self, device):
        super().to(device)
        self.device = device
        return self

# TODO tensorize
def compute_probability_novelty(
        scores,
        box_counts,
        novelty_type_classifier,
        device,
        hint_a = None,
        hint_b = None):
    '''
    Parameters:
        scores: Tensor of shape [N, L], where N is the number of images and
            L is the number of novelty signals.
        novelty_type_classifier: NoveltyTypeClassifier
            Predicts novelty type probability tensor
        hint_a: int in {0, 1, 2, 3, 4, 5} or None
            Specifies the trial-level novelty type. 0 means the trial doesn't
            contain any novelty. The rest correspond to novelty types 2, 3, 4,
            5, and 6, respectively. If None, then no hint is specified.
        hint_b: Boolean tensor of shape [N] or None
            hint_b[i] specifies whether image i is novel (True) or non-novel
            (False).
    '''
    p_type = []
    p_n = []

    for idx in range(len(scores)):
        img_scores = scores[idx]
        box_count = box_counts[idx]
        if img_scores is None:
            # Image was empty. Assume non-novel.
            # TODO Maybe we should allow novel empty images. But in that case
            # we have to train a separate logistic regression for empty images
            # since there are still activation statistical scores but no
            # box logit scores
            cur_p_type = torch.zeros(6, device=device)
            cur_p_type[0] = 1.0
            p_type.append(
                cur_p_type
            )
            cur_p_n = 1.0 - cur_p_type[0]
            p_n.append(cur_p_n)
            continue
        
        if hint_b is not None:
            cur_hint_b = hint_b[idx]
        else:
            cur_hint_b = None

        possible_nov_types = torch.ones(6, dtype=torch.bool, device=device)
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
                # Edge case: Images with one box cannot have type 4 or type 5
                # novelties.
                if box_count == 1:
                    possible_nov_types[[3, 4]] = False

                n_possible_types = possible_nov_types.to(torch.int).sum()
                assert n_possible_types > 0
                if n_possible_types == 1:
                    cur_p_type = possible_nov_types.to(torch.float)
                else:
                    logits = \
                        novelty_type_classifier(img_scores[None]).squeeze(0)
                    cur_p_type = torch.nn.functional.softmax(logits, dim = 0)

                    # Some impossible novelty types still have predicted non-zero
                    # probabilities associated with them; zero them out and
                    # renormalize the predictions
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

        p_type.append(cur_p_type)

        # P(novel) = 1 - P(type = 0)
        cur_p_n = 1.0 - cur_p_type[0]
        p_n.append(cur_p_n)

    p_type = torch.stack(p_type, dim = 0)
    p_n = torch.stack(p_n, dim = 0)

    return p_type, p_n
