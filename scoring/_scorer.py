# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

from abc import ABC, abstractmethod

import torch

'''
Produces novelty scores for a single image

Notation:
    N: Number of boxes
    S: Number of unique species classes
    A: Number of unique activity classes
    L: Number of scores produced for a given image
'''
class ImageScorer(ABC):
    '''
    Produces novelty scores for a single image

    Parameters:
        species_logits: Tensor of shape NxS
        activity_logits: Tensor of shape NxA

    Returns:
        scores: Tensor of shape L
            Scores for the given image
    '''
    @abstractmethod
    def score(self, species_logits, activity_logits):
        return NotImplemented

    '''
    Returns:
        L: int
    '''
    @abstractmethod
    def n_scores(self):
        return NotImplemented

    @abstractmethod
    def to(self, device):
        return NotImplemented

'''
Notation:
    M: Total number of boxes in the batch
    N: Number of images in the batch
    S: Number of unique species classes
    A: Number of unique activity classes
    D: Number of whole-image features produced for each image
    L: Number of scores produced for each image
'''
class Scorer(ABC):
    '''
    Produces novelty signals in the form of scalar scores for each image,
    given the box classifier's logit predictions.

    Parameters:
        species_logits: Tensor of shape MxS
        activity_logits: Tensor of shape MxA
        whole_image_features: Tensor of shape NxD
        box_counts: List[int] of length N
            box_counts[i] specifies the number of boxes in image i

    Returns:
        Tensor of shape NxL, where L is the number of scores produced for
            each image.
    '''
    @abstractmethod
    def score(
            self,
            species_logits,
            activity_logits,
            whole_image_features,
            box_counts):
        return NotImplemented

    '''
    Returns:
        L: int
    '''
    @abstractmethod
    def n_scores(self):
        return NotImplemented

    @abstractmethod
    def to(self, device):
        return NotImplemented

'''
Composes multiple ImageScorer objects into one, concatenating their score
tensors in-order
'''
class CompositeImageScorer(ImageScorer):
    def __init__(self, image_scorers):
        self._image_scorers = list(image_scorers)
        score_counts = [img_scorer.n_scores() for img_scorer in image_scorers]
        self._n_scores = sum(score_counts)

    def score(self, species_logits, activity_logits):
        img_scores = [
            img_scorer.score(species_logits, activity_logits)\
                for img_scorer in self._image_scorers
        ]
        return torch.cat(img_scores, dim=0)

    def n_scores(self):
        return self._n_scores

    def to(self, device):
        for idx, image_scorer in enumerate(self._image_scorers):
            self._image_scorers[idx] = image_scorer.to(device)
        return self

'''
Converts an ImageScorer into a Scorer by splitting the logit tensors by image
and concatenating image score tensors in image-order
'''
class ScorerFromImageScorer(Scorer):
    def __init__(self, image_scorer):
        self._image_scorer = image_scorer

    def score(
            self,
            species_logits,
            activity_logits,
            whole_image_features,
            box_counts):
        scores = []
        split_species_logits = torch.split(species_logits, box_counts, dim=0)
        split_activity_logits = torch.split(activity_logits, box_counts, dim=0)
        for img_species_logits, img_activity_logits in\
                zip(split_species_logits, split_activity_logits):
            img_scores = self._image_scorer.score(
                img_species_logits,
                img_activity_logits
            )
            scores.append(img_scores)
        scores = torch.stack(scores, dim=0)
        return scores

    def n_scores(self):
        return self._image_scorer.n_scores()

    def to(self, device):
        self._image_scorer = self._image_scorer.to(device)
        return self

'''
Composes multiple Scorer objects into one, concatenating their score tensors
in-order
'''
class CompositeScorer(Scorer):
    def __init__(self, scorers):
        self._scorers = list(scorers)
        score_counts = [scorer.n_scores() for scorer in scorers]
        self._n_scores = sum(score_counts)

    def score(
            self,
            species_logits,
            activity_logits,
            whole_image_features,
            box_counts):
        scores = [
            scorer.score(
                species_logits,
                activity_logits,
                whole_image_features,
                box_counts
            ) for scorer in self._scorers
        ]
        return torch.cat(scores, dim=1)

    def n_scores(self):
        return self._n_scores

    def to(self, device):
        for idx, scorer in enumerate(self._scorers):
            self._scorers[idx] = scorer.to(device)
        return self
