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
