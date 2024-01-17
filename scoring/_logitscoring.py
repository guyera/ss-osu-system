import torch

from scoring._scorer import\
    Scorer,\
    CompositeScorer,\
    BatchScorerFromScorer

def _avg_max_logit_image_score(logits):
    max_logits, _ = torch.max(logits, dim=1)
    avg_max_logit = max_logits.mean(dim=0, keepdim=True)
    return avg_max_logit

def _max_avg_logit_image_score(logits):
    avg_logits = logits.mean(dim=0)
    max_avg_logit, _ = torch.max(avg_logits, dim=0, keepdim=True)
    return max_avg_logit

def _continuous_presence_count(logits, sigmoid_slope, threshold, temperature):
    # The goal is to compute, for each class, some continuous
    # approximation of the box-wise max logit and compare it against a
    # threshold. We then sum up the number of classes that exceed the threshold.
    # This reflects the number of classes present in the image.  We use
    # the boltzmann operator with a small temperature to approximate the max
    # continuously, and we use sigmoid with a learned slope and bias to
    # approximate the thresholding continuously.
    scaled_logits = logits / temperature
    scaled_softmax = torch.nn.functional.softmax(scaled_logits)
    boltzmann = (scaled_softmax * logits).sum(dim=0)
    threshold_exceedances = \
        torch.nn.functional.sigmoid(sigmoid_slope * (boltzmann - threshold))
    # Num classes present is approximated as the sum of the exceedances
    num_classes_present = threshold_exceedances.sum(dim=0, keepdim=True)
    return num_classes_present

class AvgMaxSpeciesLogitScorer(Scorer):
    def __init__(self, n_known_species_cls):
        super().__init__()
        self._n_known_species_cls = n_known_species_cls

    def forward(self, species_logits, activity_logits):
        return _avg_max_logit_image_score(
            species_logits[:, :self._n_known_species_cls]
        )

    def n_scores(self):
        return 1

    def to(self, device):
        return self

class MaxAvgSpeciesLogitScorer(Scorer):
    def __init__(self, n_known_species_cls):
        super().__init__()
        self._n_known_species_cls = n_known_species_cls

    def forward(self, species_logits, activity_logits):
        return _max_avg_logit_image_score(
            species_logits[:, :self._n_known_species_cls]
        )

    def n_scores(self):
        return 1

    def to(self, device):
        return self

class ContinuousPresenceCountSpeciesLogitScorer(Scorer):
    def __init__(self, n_known_species_cls, temperature):
        super().__init__()
        self._n_known_species_cls = n_known_species_cls
        self._temperature = temperature
        self.sigmoid_slope = torch.nn.Parameter(torch.tensor(1.0))
        self.threshold = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, species_logits, activity_logits):
        return _continuous_presence_count(
            species_logits[:, :self._n_known_species_cls],
            self.sigmoid_slope,
            self.threshold,
            self._temperature
        )

    def n_scores(self):
        return 1

    def to(self, device):
        return self

class AvgMaxActivityLogitScorer(Scorer):
    def __init__(self, n_known_activity_cls):
        super().__init__()
        self._n_known_activity_cls = n_known_activity_cls

    def forward(self, species_logits, activity_logits):
        return _avg_max_logit_image_score(
            activity_logits[:, :self._n_known_activity_cls]
        )

    def n_scores(self):
        return 1

    def to(self, device):
        return self


class ContinuousPresenceCountActivityLogitScorer(Scorer):
    def __init__(self, n_known_activity_cls, temperature):
        super().__init__()
        self._n_known_activity_cls = n_known_activity_cls
        self._temperature = temperature
        self.sigmoid_slope = torch.nn.Parameter(torch.tensor(1.0))
        self.threshold = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, species_logits, activity_logits):
        return _continuous_presence_count(
            activity_logits[:, :self._n_known_activity_cls],
            self.sigmoid_slope,
            self.threshold,
            self._temperature
        )

    def n_scores(self):
        return 1

    def to(self, device):
        return self

class MaxAvgActivityLogitScorer(Scorer):
    def __init__(self, n_known_activity_cls):
        super().__init__()
        self._n_known_activity_cls = n_known_activity_cls

    def forward(self, species_logits, activity_logits):
        return _max_avg_logit_image_score(
            activity_logits[:, :self._n_known_activity_cls]
        )

    def n_scores(self):
        return 1

    def to(self, device):
        return self

def make_logit_scorer(n_known_species_cls, n_known_activity_cls):
    continuous_presence_count_temperature = 0.1
    avg_max_species_logit_image_scorer = AvgMaxSpeciesLogitScorer(
        n_known_species_cls
    )
    continuous_presence_count_species_logit_image_scorer = \
        ContinuousPresenceCountSpeciesLogitScorer(
            n_known_species_cls,
            continuous_presence_count_temperature
        )
    avg_max_activity_logit_image_scorer = AvgMaxActivityLogitScorer(
        n_known_activity_cls
    )
    continuous_presence_count_activity_logit_image_scorer = \
        ContinuousPresenceCountSpeciesLogitScorer(
            n_known_activity_cls,
            continuous_presence_count_temperature
        )
    composite_logit_image_scorer = CompositeScorer((
        avg_max_species_logit_image_scorer,
        continuous_presence_count_species_logit_image_scorer,
        avg_max_activity_logit_image_scorer,
        continuous_presence_count_activity_logit_image_scorer
    ))
    composite_logit_scorer = BatchScorerFromScorer(composite_logit_image_scorer)
    return composite_logit_scorer
