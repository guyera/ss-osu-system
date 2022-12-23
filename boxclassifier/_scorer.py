from abc import ABC, abstractmethod

class ImageScorer(ABC):
    '''
    Produces a single novelty signal for a single image with N boxes.

    Parameters:
        species_logits: Tensor of shape NxS, where S is the number of
            unique species classes
        activity_logits: Tensor of shape NxA, where A is the number of
            unique activity classes

    Returns:
        Scalar tensor of novelty signal
    '''
    @abstractmethod
    def score(self, species_logits, activity_logits):
        return NotImplemented

class Scorer(ABC):
    '''
    Produces novelty signals in the form of scalar scores for each image,
    given the box classifier's logit predictions.

    Parameters:
        species_logits: Tensor of shape MxS, where M is the total number of
            boxes across the entire batch and S is the number of species
            classes.
        activity_logits: Tensor of shape MxA, where A is the number of activity
            classes.
        box_counts: Iterable[Int] of length N, where N is the number of
            images in the batch and box_counts[i] specifies the number of boxes
            for image i. Logits are presented in image-order; that is, the
            logits for the boxes of an image are all consecutive.

    Returns:
        Tensor of shape MxL, where L is the number of scores produced for
            each image.
    '''
    @abstractmethod
    def score(self, species_logits, activity_logits, box_counts):
        return NotImplemented

    '''
    Returns L, the number of scores produced by the scorer. Necessary for
    constructing the novelty type logistic regressions (novelty
    characterization + detection)
    '''
    @abstractmethod
    def n_scores(self):
        return NotImplemented

class CompositeImageScorer(Scorer):
    def __init__(self, image_scorers):
        self._image_scorers = image_scorers

    def score(self, species_logits, activity_logits, box_counts):
        split_species_logits = torch.split(species_logits, box_counts, dim=0)
        split_activity_logits = torch.split(activity_logits, box_counts, dim=0)
        scores = []
        for img_species_logits, img_activity_logits in\
                zip(split_species_logits, split_activity_logits):
            img_scores = [
                img_scorer.score(img_species_logits, img_activity_logits)\
                    for img_scorer in self._image_scorers
            ]
            img_scores = torch.stack(img_scores, dim=0)
            scores.append(img_scores)
        scores = torch.stack(scores, dim=0)
        return scores

    def n_scores(self):
        return len(self._image_scorers)

def _avg_max_logit_image_score(self, logits):
    max_logits, _ = torch.max(logits, dim=1)
    avg_max_logit = max_logits.mean()
    return avg_max_logit

def _max_avg_logit_image_score(self, logits):
    avg_logits = logits.mean(dim=0)
    max_avg_logit, _ = torch.max(avg_logits, dim=0)
    return max_avg_logit

class AvgMaxSpeciesLogitImageScorer(ImageScorer):
    def score(self, species_logits, activity_logits):
        return _avg_max_logit_image_score(species_logits)

class MaxAvgSpeciesLogitImageScorer(ImageScorer):
    def score(self, species_logits, activity_logits):
        return _max_avg_logit_image_score(species_logits)

class AvgMaxActivityLogitImageScorer(ImageScorer):
    def score(self, species_logits, activity_logits):
        return _avg_max_logit_image_score(activity_logits)

class MaxAvgActivityLogitImageScorer(ImageScorer):
    def score(self, species_logits, activity_logits):
        return _max_avg_logit_image_score(activity_logits)
