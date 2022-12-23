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
    MaxAvgActivityLogitImageScorer
from scoring._wholeimagescoring import ActivationStatisticalModel
