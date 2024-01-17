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
class Scorer(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    '''
    Returns:
        L: int
    '''
    @abstractmethod
    def n_scores(self):
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
class BatchScorer(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()

    '''
    Returns:
        L: int
    '''
    @abstractmethod
    def n_scores(self):
        return NotImplemented

'''
Composes multiple Scorer objects into one, concatenating their score
tensors in-order
'''
class CompositeScorer(Scorer):
    def __init__(self, scorers):
        super().__init__()
        self.scorers = torch.nn.ModuleList(scorers)
        score_counts = [cur_scorer.n_scores() for cur_scorer in scorers]
        self._n_scores = sum(score_counts)

    def forward(self, species_logits, activity_logits):
        scores = [
            cur_scorer(species_logits, activity_logits)\
                for cur_scorer in self.scorers
        ]
        return torch.cat(scores, dim=0)

    def n_scores(self):
        return self._n_scores

'''
Converts an Scorer into a BatchScorer by splitting the logit tensors by image
and concatenating image score tensors in image-order
'''
class BatchScorerFromScorer(BatchScorer):
    def __init__(self, scorer):
        super().__init__()
        self.scorer = scorer

    def forward(
            self,
            species_logits,
            activity_logits,
            box_counts):
        scores = []
        split_species_logits = torch.split(species_logits, box_counts, dim=0)
        split_activity_logits = torch.split(activity_logits, box_counts, dim=0)
        for img_species_logits, img_activity_logits in\
                zip(split_species_logits, split_activity_logits):
            img_scores = self.scorer(
                img_species_logits,
                img_activity_logits
            )
            scores.append(img_scores)
        scores = torch.stack(scores, dim=0)
        return scores

    def n_scores(self):
        return self.scorer.n_scores()

'''
Composes multiple BatchScorer objects into one, concatenating their score tensors
in-order
'''
class CompositeBatchScorer(BatchScorer):
    def __init__(self, scorers):
        super().__init__()
        self.scorers = torch.nn.ModuleList(scorers)
        score_counts = [scorer.n_scores() for scorer in scorers]
        self._n_scores = sum(score_counts)

    def forward(
            self,
            species_logits,
            activity_logits,
            box_counts):
        scores = [
            scorer(
                species_logits,
                activity_logits,
                box_counts
            ) for scorer in self.scorers
        ]
        return torch.cat(scores, dim=1)

    def n_scores(self):
        return self._n_scores
