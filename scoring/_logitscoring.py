import torch

def _avg_max_logit_image_score(self, logits):
    max_logits, _ = torch.max(logits, dim=1)
    avg_max_logit = max_logits.mean()
    return avg_max_logit

def _max_avg_logit_image_score(self, logits):
    avg_logits = logits.mean(dim=0)
    max_avg_logit, _ = torch.max(avg_logits, dim=0)
    return max_avg_logit

class AvgMaxSpeciesLogitImageScorer(ImageScorer):
    def __init__(self, n_known_species_cls):
        self._n_known_species_cls = n_known_species_cls

    def score(self, species_logits, activity_logits):
        return _avg_max_logit_image_score(
            species_logits[:, :self._n_known_species_cls]
        )

class MaxAvgSpeciesLogitImageScorer(ImageScorer):
    def __init__(self, n_known_species_cls):
        self._n_known_species_cls = n_known_species_cls

    def score(self, species_logits, activity_logits):
        return _max_avg_logit_image_score(
            species_logits[:, :self._n_known_species_cls]
        )

class AvgMaxActivityLogitImageScorer(ImageScorer):
    def __init__(self, n_known_activity_cls):
        self._n_known_activity_cls = n_known_activity_cls

    def score(self, activity_logits, activity_logits):
        return _avg_max_logit_image_score(
            activity_logits[:, :self._n_known_activity_cls]
        )

class MaxAvgActivityLogitImageScorer(ImageScorer):
    def __init__(self, n_known_activity_cls):
        self._n_known_activity_cls = n_known_activity_cls

    def score(self, activity_logits, activity_logits):
        return _max_avg_logit_image_score(
            activity_logits[:, :self._n_known_activity_cls]
        )
