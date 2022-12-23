from abc import ABC, abstractmethod

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
        box_counts: Iterable[Int] of length N, where box_counts[i] specifies the
            number of boxes for image i. Logits are presented in image-order;
            that is, the logits for the boxes of an image are all consecutive.

    Returns:
        Tensor of shape MxL, where L is the number of scores produced for
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

'''
Composes multiple ImageScorer objects into one, concatenating their score
tensors in-order
'''
class CompositeImageScorer(ImageScorer):
    def __init__(self, image_scorers):
        self._image_scorers = image_scorers
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
        split_species_logits = torch.split(species_logits, box_counts, dim=0)
        split_activity_logits = torch.split(activity_logits, box_counts, dim=0)
        scores = []
        for img_species_logits, img_activity_logits in\
                zip(split_species_logits, split_activity_logits):
            img_scores = self._image_scorer.score(
                img_species_logits,
                img_activity_logits
            )
            scores.append(img_scores)
        scores = torch.stack(scores, dim=0)
        return scores

'''
Composes multiple Scorer objects into one, concatenating their score tensors
in-order
'''
class CompositeScorer(Scorer):
    def __init__(self, scorers):
        self._scorers = scorers
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
