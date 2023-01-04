from scoring._scorer import\
    Scorer,\
    ImageScorer,\
    CompositeScorer,\
    CompositeImageScorer,\
    ScorerFromImageScorer
from scoring._logitscoring import\
    AvgMaxSpeciesLogitImageScorer,\
    MaxAvgSpeciesLogitImageScorer,\
    AvgMaxActivityLogitImageScorer,\
    MaxAvgActivityLogitImageScorer,\
    make_logit_scorer
from scoring._wholeimagescoring import ActivationStatisticalModel
