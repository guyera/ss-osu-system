import torch
import sklearn.metrics
import sklearn.neighbors
import math
import sys

from tqdm import tqdm

from enum import Enum

def compute_partial_auc(nominal_scores, novel_scores):
    nominal_trues = torch.zeros_like(nominal_scores)
    novel_trues = torch.ones_like(novel_scores)

    trues = torch.cat((nominal_trues, novel_trues), dim = 0)\
        .data.cpu().numpy()
    scores = torch.cat((nominal_scores, novel_scores), dim = 0)\
        .data.cpu().numpy()

    auc = sklearn.metrics.roc_auc_score(trues, scores, max_fpr = 0.25)

    return auc

class Case1LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 5)
        self.mean = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(4), requires_grad=False)

    def forward(self, x):
        return self.fc((x - self.mean) / self.std)

    def fit_standardization_statistics(self, scores):
        self.mean[:] = scores.mean(dim=0).detach()
        self.std[:] = scores.std(dim=0).detach()

class Case2LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(3, 4)
        self.mean = torch.nn.Parameter(torch.zeros(3), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    def forward(self, x):
        return self.fc((x - self.mean) / self.std)

    def fit_standardization_statistics(self, scores):
        self.mean[:] = scores.mean(dim=0).detach()
        self.std[:] = scores.std(dim=0).detach()

class Case3LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 3)
        self.mean = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
        self.std = torch.nn.Parameter(torch.ones(2), requires_grad=False)

    def forward(self, x):
        return self.fc((x - self.mean) / self.std)

    def fit_standardization_statistics(self, scores):
        self.mean[:] = scores.mean(dim=0).detach()
        self.std[:] = scores.std(dim=0).detach()

class ScoreContext:
    class Source(Enum):
        UNSUPERVISED = 1
        SUPERVISED = 2
    
    def __init__(self, source, nominal_scores = None, novel_scores = None):
        self.device = 'cpu'
        self.source = source
        self.nominal_scores = nominal_scores
        self.novel_scores = novel_scores

    def add_nominal_scores(self, nominal_scores):
        if self.nominal_scores is None:
            self.nominal_scores = nominal_scores
        else:
            self.nominal_scores =\
                torch.cat((self.nominal_scores, nominal_scores), dim = 0)

    def add_novel_scores(self, novel_scores):
        if self.novel_scores is None:
            self.novel_scores = novel_scores
        else:
            self.novel_scores =\
                torch.cat((self.novel_scores, novel_scores), dim = 0)

    def data_available(self):
        return self.nominal_scores is not None and self.novel_scores is not None

    def compute_partial_auc(self):
        return compute_partial_auc(self.nominal_scores, self.novel_scores)

    # Use silverman's rule of thumb to choose bandwidth.
    def _compute_bandwidth(self, scores):
        std = torch.std(scores, dim = 0, unbiased = True)
        q3 = torch.quantile(scores, 0.75, dim = 0)
        q1 = torch.quantile(scores, 0.25, dim = 0)
        iqr = q3 - q1
        a = torch.min(std, iqr / 1.34)
        bw = 0.9 * a * math.pow(len(scores), -0.2)
        return bw
    
    def _nominal_bandwidth(self):
        return self._compute_bandwidth(self.nominal_scores)

    def _novel_bandwidth(self):
        return self._compute_bandwidth(self.novel_scores)
    
    def fit_kdes(self):
        nominal_kde = sklearn.neighbors.KernelDensity(kernel = 'gaussian', bandwidth = self._nominal_bandwidth())
        nominal_kde.fit(self.nominal_scores.unsqueeze(1).detach().cpu().numpy())
        
        novel_kde = sklearn.neighbors.KernelDensity(kernel = 'gaussian', bandwidth = self._novel_bandwidth())
        novel_kde.fit(self.novel_scores.unsqueeze(1).detach().cpu().numpy())

        return nominal_kde, novel_kde

    def get_mean_scores(self):
        return torch.mean(self.nominal_scores), torch.mean(self.novel_scores)

    def state_dict(self):
        state_dict = {}
        state_dict['source'] = self.source
        state_dict['nominal_scores'] = self.nominal_scores.data.cpu()\
            if self.nominal_scores is not None else None
        state_dict['novel_scores'] = self.novel_scores.data.cpu()\
            if self.novel_scores is not None else None
        return state_dict

    def load_state_dict(self, state_dict):
        self.source = state_dict['source']
        self.nominal_scores = state_dict['nominal_scores']
        self.novel_scores = state_dict['novel_scores']
        
        self.nominal_scores = self.nominal_scores.to(self.device)\
            if self.nominal_scores is not None else self.nominal_scores
        self.novel_scores = self.novel_scores.to(self.device)\
            if self.novel_scores is not None else self.novel_scores

    def to(self, device):
        self.device = device
        self.nominal_scores = self.nominal_scores.to(device)
        self.novel_scores = self.novel_scores.to(device)
        return self

# If p_t4 is None, then it sets p_type[3] to zero and forfeits type 4 novelty.
def compute_probability_novelty(
        subject_scores,
        verb_scores,
        object_scores,
        activation_statistical_scores,
        case_1_logistic_regression,
        case_2_logistic_regression,
        case_3_logistic_regression,
        ignore_t2_in_pni = True,
        p_t4 = None,
        hint_a = None,
        hint_b = None):
    '''
    Parameters:
        subject_scores: List of N scalars, some of which may be None
            Subject novelty scores (negative max logits from the subject box
            classifier, returned by
            UnsupervisedNoveltyDetector.scores_and_p_t4()
        verb_scores: List of N scalars, some of which may be None
            Subject novelty scores (negative max logits from the verb box
            classifier, returned by
            UnsupervisedNoveltyDetector.scores_and_p_t4()
        object_scores: List of N scalars, some of which may be None
            Subject novelty scores (negative max logits from the object box
            classifier, returned by
            UnsupervisedNoveltyDetector.scores_and_p_t4()
        activation_statistical_scores: numpy.ndarray of shape [N]
            Whole-image novelty scores computed from a statistical model fit to
            PCA projections of early-layer activations extracted from whole
            images. Returned by ActivationStatisticalModel.score()
        case_1_logistic_regression: Case1LogisticRegression
            Predicts novelty type probability tensor for case 1 instances
        case_2_logistic_regression: Case2LogisticRegression
            Predicts novelty type probability tensor for case 2 instances
        case_3_logistic_regression: Case3LogisticRegression
            Predicts novelty type probability tensor for case 3 instances
        ignore_t2_in_pni: bool
            Specifies whether the type 2 (novel verb) probability should be
            ignored when computing p_n -- a hack for preventing false alarms
            when type 0 is frequently confused for type 2
        p_t4: Tensor of shape [N] or None
            p_t4[i] represents P(type = 4 | type = 0 or 4) for image i.
            Returned by UnsupervisedNoveltyDetector.scores_and_p_t4(). If None,
            type 4 probabilities are set to zero.
        hint_a: int in {0, 1, 2, 3, 4, 5, 6, 7} or None
            Specifies the trial-level novelty type. 0 means the trial doesn't
            contain any novelty. The rest correspond to their respective
            novelty types. If None, then no hint is specified.
        hint_b: Boolean tensor of shape [N] or None
            hint_b[i] specifies whether image i is novel (True) or non-novel
            (False).
    '''
    p_type = []
    p_n = []
    for idx in range(len(subject_scores)):
        subject_score = subject_scores[idx]
        object_score = object_scores[idx]
        verb_score = verb_scores[idx]
        activation_statistical_score = torch.from_numpy(activation_statistical_scores[idx])
        if hint_b is not None:
            cur_hint_b = hint_b[idx]
        else:
            cur_hint_b = None
        if subject_score is not None:
            device = subject_score.device
        else:
            device = object_score.device

        if p_t4 is None:
            cur_cond_p_t4 = 0
        else:
            cur_cond_p_t4 = p_t4[idx]
        
        zero_out = False
        possible_nov_types = torch.ones(6, dtype=torch.bool, device=device)
        if cur_hint_b is not None and not cur_hint_b:
            possible_nov_types[1:] = False
            cur_p_type = possible_nov_types.to(torch.float)
        else:
            if cur_hint_b is not None and cur_hint_b:
                possible_nov_types[0] = False
            
            if hint_a is not None:
                if hint_a in [6, 7]:
                    nov_type_idx = 5
                elif hint_a == 5:
                    nov_type_idx = 0
                else:
                    nov_type_idx = hint_a
                mask = torch.ones_like(possible_nov_types)
                mask[0] = False
                mask[nov_type_idx] = False
                possible_nov_types[mask] = False
                
            n_possible_types = possible_nov_types.to(torch.int).sum()
            assert n_possible_types > 0
            if n_possible_types == 1:
                cur_p_type = possible_nov_types.to(torch.float)
            elif subject_score is not None and object_score is not None:
                # Case 1
                if not torch.any(possible_nov_types[[1, 2, 3, 5]]):
                    cur_p_type = torch.zeros(6, dtype=torch.float, device=device)
                    cur_p_type[0] = 1 - cur_cond_p_t4
                    cur_p_type[4] = cur_cond_p_t4
                else:
                    scores = torch.stack((subject_score, verb_score, object_score, activation_statistical_score.to(torch.float).to(device)), dim = 0).unsqueeze(0)
                    logits = case_1_logistic_regression(scores).squeeze(0)
                    softmax = torch.nn.functional.softmax(logits, dim = 0)
                    cur_p_t0 = (1 - cur_cond_p_t4) * softmax[0]
                    cur_p_t4 = cur_cond_p_t4 * softmax[0]
                    cur_p_type = torch.cat(
                        (
                            cur_p_t0.unsqueeze(0),
                            softmax[1:-1],
                            cur_p_t4.unsqueeze(0),
                            softmax[-1:]
                        ),
                        dim=0
                    )
                    zero_out = True
            elif subject_score is not None:
                # Case 2
                possible_nov_types[3] = False
                n_possible_types = possible_nov_types.to(torch.int).sum()
                assert n_possible_types > 0
                if n_possible_types == 1:
                    cur_p_type = possible_nov_types.to(torch.float)
                elif not torch.any(possible_nov_types[[1, 2, 3, 5]]):
                    cur_p_type = torch.zeros(6, dtype=torch.float, device=device)
                    cur_p_type[0] = 1 - cur_cond_p_t4
                    cur_p_type[4] = cur_cond_p_t4
                else:
                    scores = torch.stack((subject_score, verb_score, activation_statistical_score.to(torch.float).to(device)), dim = 0).unsqueeze(0)
                    logits = case_2_logistic_regression(scores).squeeze(0)
                    softmax = torch.nn.functional.softmax(logits, dim = 0)
                    cur_p_t0 = (1 - cur_cond_p_t4) * softmax[0]
                    cur_p_t4 = cur_cond_p_t4 * softmax[0]
                    cur_p_type = torch.cat(
                        (
                            cur_p_t0.unsqueeze(0),
                            softmax[1:-1],
                            torch.zeros(1, dtype=torch.float, device=device),
                            cur_p_t4.unsqueeze(0),
                            softmax[-1:]
                        ),
                        dim=0
                    )
                    zero_out = True
            else:
                # Case 3
                possible_nov_types[[1, 2, 4]] = False
                n_possible_types = possible_nov_types.to(torch.int).sum()
                assert n_possible_types > 0
                if n_possible_types == 1:
                    cur_p_type = possible_nov_types.to(torch.float)
                else:
                    scores = torch.stack((object_score, activation_statistical_score.to(torch.float).to(object_score.device)), dim = 0).unsqueeze(0)
                    logits = case_3_logistic_regression(scores).squeeze(0)
                    softmax = torch.nn.functional.softmax(logits, dim = 0)
                    cur_p_type = torch.cat(
                        (
                            softmax[0:1],
                            torch.zeros(2, dtype=torch.float, device=device),
                            softmax[1:-1],
                            torch.zeros(1, dtype=torch.float, device=device),
                            softmax[-1:]
                        ),
                        dim=0
                    )
                    zero_out = True

        # Some impossible novelty types still have predicted non-zero
        # probabilities associated with them; zero them out and renormalize
        # the predictions
        if zero_out:
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

        # Normalize the NOVEL novelty types; i.e., types 1, 2, 3, 4, and 6/7.
        # This normalized partial p-type vector represents the probability
        # of each novelty type given that some novelty is present. This is
        # what's passed to the novel tuple classifier
        normalizer = cur_p_type[1:].sum()
        if normalizer == 0:
            cur_novel_p_type = possible_nov_types[1:].to(torch.float)
            normalizer = cur_novel_p_type.sum()
            if normalizer == 0:
                cur_novel_p_type = torch.full_like(cur_p_type[1:], 1.0 / len(cur_p_type[1:]))
            else:
                cur_novel_p_type = cur_novel_p_type / normalizer
        else:
            cur_novel_p_type = cur_p_type[1:] / normalizer

        p_type.append(cur_novel_p_type)

        # P(novel) = P(type = 1, 2, 3, 4, or 6/7), or the sum of these
        # probabilities. Alternatively, 1 - P(type = 0)
        cur_p_n = 1.0 - cur_p_type[0]
        p_n.append(cur_p_n)

    p_type = torch.stack(p_type, dim = 0)
    p_n = torch.stack(p_n, dim = 0)
    
    return p_type, p_n

def separate_scores_and_labels(subject_scores, object_scores, verb_scores, subject_labels, object_labels, verb_labels):
    case_1_type_0_scores = []
    case_1_type_1_scores = []
    case_1_type_2_scores = []
    case_1_type_3_scores = []
    case_2_type_0_scores = []
    case_2_type_1_scores = []
    case_2_type_2_scores = []
    case_3_type_0_scores = []
    case_3_type_3_scores = []

    for idx in range(len(subject_scores)):
        subject_score = subject_scores[idx]
        object_score = object_scores[idx]
        verb_score = verb_scores[idx]
        subject_label = subject_labels[idx]
        object_label = object_labels[idx]
        verb_label = verb_labels[idx]
        
        # Determine the novelty type
        if subject_label == 0:
            if object_label == 0 or verb_label == 0:
                # Invalid novel example; multiple novelty types. Filter it out.
                continue
            # Type 1
            type_label = 1
        elif subject_label is not None and verb_label == 0:
            if subject_label == 0 or object_label == 0:
                # Invalid novel example; multiple novelty types. Filter it out.
                continue
            # Type 2
            type_label = 2
        elif object_label == 0:
            if subject_label == 0 or (subject_label is not None and verb_label == 0):
                # Invalid novel example; multiple novelty types. Filter it out.
                continue
            # Type 3
            type_label = 3
        else:
            # Type 0
            type_label = 0

        # Add the scores and labels to the case inputs
        if subject_score is not None and object_score is not None:
            # All boxes present; append scores to case 1
            if type_label == 0:
                case_1_type_0_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
            elif type_label == 1:
                case_1_type_1_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
            elif type_label == 2:
                case_1_type_2_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
            elif type_label == 3:
                case_1_type_3_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
        if subject_score is not None:
            # At least the subject box is present; append subject and verb scores
            # to case 2
            if type_label == 0:
                case_2_type_0_scores.append(torch.stack((subject_score, verb_score), dim = 0))
            elif type_label == 1:
                case_2_type_1_scores.append(torch.stack((subject_score, verb_score), dim = 0))
            elif type_label == 2:
                case_2_type_2_scores.append(torch.stack((subject_score, verb_score), dim = 0))
        if object_score is not None:
            # At least the object box is present; append object score to case 3
            if type_label == 0:
                case_3_type_0_scores.append(object_score.unsqueeze(0))
            if type_label == 3:
                case_3_type_3_scores.append(object_score.unsqueeze(0))

    case_1_type_0_scores = torch.stack(case_1_type_0_scores, dim = 0)
    case_1_type_1_scores = torch.stack(case_1_type_1_scores, dim = 0)
    case_1_type_2_scores = torch.stack(case_1_type_2_scores, dim = 0)
    case_1_type_3_scores = torch.stack(case_1_type_3_scores, dim = 0)
    case_2_type_0_scores = torch.stack(case_2_type_0_scores, dim = 0)
    case_2_type_1_scores = torch.stack(case_2_type_1_scores, dim = 0)
    case_2_type_2_scores = torch.stack(case_2_type_2_scores, dim = 0)
    case_3_type_0_scores = torch.stack(case_3_type_0_scores, dim = 0)
    case_3_type_3_scores = torch.stack(case_3_type_3_scores, dim = 0)

    device = case_1_type_0_scores.device

    # Randomly shuffle score rows
    case_1_type_0_scores = case_1_type_0_scores[torch.randperm(len(case_1_type_0_scores), device = device)]
    case_1_type_1_scores = case_1_type_1_scores[torch.randperm(len(case_1_type_1_scores), device = device)]
    case_1_type_2_scores = case_1_type_2_scores[torch.randperm(len(case_1_type_2_scores), device = device)]
    case_1_type_3_scores = case_1_type_3_scores[torch.randperm(len(case_1_type_3_scores), device = device)]
    case_2_type_0_scores = case_2_type_0_scores[torch.randperm(len(case_2_type_0_scores), device = device)]
    case_2_type_1_scores = case_2_type_1_scores[torch.randperm(len(case_2_type_1_scores), device = device)]
    case_2_type_2_scores = case_2_type_2_scores[torch.randperm(len(case_2_type_2_scores), device = device)]
    case_3_type_0_scores = case_3_type_0_scores[torch.randperm(len(case_3_type_0_scores), device = device)]
    case_3_type_3_scores = case_3_type_3_scores[torch.randperm(len(case_3_type_3_scores), device = device)]

    # Now, we need to balance classes. Compute novelty type with fewest instances in
    # each case
    case_1_n_per_class = min(len(case_1_type_0_scores), len(case_1_type_1_scores), len(case_1_type_2_scores), len(case_1_type_3_scores))
    case_2_n_per_class = min(len(case_2_type_0_scores), len(case_2_type_1_scores), len(case_2_type_2_scores))
    case_3_n_per_class = min(len(case_3_type_0_scores), len(case_3_type_3_scores))

    # Balance the classes
    case_1_type_0_scores = case_1_type_0_scores[:case_1_n_per_class]
    case_1_type_1_scores = case_1_type_1_scores[:case_1_n_per_class]
    case_1_type_2_scores = case_1_type_2_scores[:case_1_n_per_class]
    case_1_type_3_scores = case_1_type_3_scores[:case_1_n_per_class]
    case_2_type_0_scores = case_2_type_0_scores[:case_2_n_per_class]
    case_2_type_1_scores = case_2_type_1_scores[:case_2_n_per_class]
    case_2_type_2_scores = case_2_type_2_scores[:case_2_n_per_class]
    case_3_type_0_scores = case_3_type_0_scores[:case_3_n_per_class]
    case_3_type_3_scores = case_3_type_3_scores[:case_3_n_per_class]

    # Construct concatenated score tensors and label tensors
    case_1_scores = torch.cat((case_1_type_0_scores, case_1_type_1_scores, case_1_type_2_scores, case_1_type_3_scores), dim = 0).detach()
    case_2_scores = torch.cat((case_2_type_0_scores, case_2_type_1_scores, case_2_type_2_scores), dim = 0).detach()
    case_3_scores = torch.cat((case_3_type_0_scores, case_3_type_3_scores), dim = 0).detach()
    case_1_labels = torch.cat((torch.full(size = (len(case_1_type_0_scores),), fill_value = 0, dtype = torch.long, device = device), torch.full(size = (len(case_1_type_1_scores),), fill_value = 1, dtype = torch.long, device = device), torch.full(size = (len(case_1_type_2_scores),), fill_value = 2, dtype = torch.long, device = device), torch.full(size = (len(case_1_type_3_scores),), fill_value = 3, dtype = torch.long, device = device)), dim = 0)
    case_2_labels = torch.cat((torch.full(size = (len(case_2_type_0_scores),), fill_value = 0, dtype = torch.long, device = device), torch.full(size = (len(case_2_type_1_scores),), fill_value = 1, dtype = torch.long, device = device), torch.full(size = (len(case_2_type_2_scores),), fill_value = 2, dtype = torch.long, device = device)), dim = 0)
    case_3_labels = torch.cat((torch.full(size = (len(case_3_type_0_scores),), fill_value = 0, dtype = torch.long, device = device), torch.full(size = (len(case_3_type_3_scores),), fill_value = 1, dtype = torch.long, device = device)), dim = 0)
    
    return case_1_scores, case_2_scores, case_3_scores, case_1_labels, case_2_labels, case_3_labels

def fit_logistic_regression(logistic_regression, scores, labels, epochs = 3000, quiet = True):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(logistic_regression.parameters(), lr = 0.01, momentum = 0.9)
    logistic_regression.fit_standardization_statistics(scores)
    
    progress = None
    if not quiet:
        progress = tqdm(total = epochs)
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = logistic_regression(scores)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        if not quiet:
            progress.set_description(f'Loss: {loss.detach().cpu().item()}')
            progress.update()
    if not quiet:
        progress.close()

'''
Parameters:
    case_1_logistic_regression:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_1_logistic_regression'
    case_2_logistic_regression:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_2_logistic_regression'
    case_3_logistic_regression:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_3_logistic_regression'
    case_1_scores:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_1_scores'
    case_2_scores:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_2_scores'
    case_3_scores:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_3_scores'
    case_1_labels:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_1_labels'
    case_2_labels:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_2_labels'
    case_3_labels:
        Loaded from unsupervised_novelty_detection_module.pth via key
        'case_3_labels'
    first_60_subject_novelty_scores:
        List of length 60, containing the 60 subject novelty scores output by
        UnsupervisedNoveltyDetectionModule.score() or
        UnsupervisedNoveltyDetectionModule.__call__() on the first 60 examples
        of the trial, guaranteed to be nominal (type 0).
    first_60_verb_novelty_scores:
        List of length 60, containing the 60 verb novelty scores output by
        UnsupervisedNoveltyDetectionModule.score() or
        UnsupervisedNoveltyDetectionModule.__call__() on the first 60 examples
        of the trial, guaranteed to be nominal (type 0).
    first_60_object_novelty_scores:
        List of length 60, containing the 60 object novelty scores output by
        UnsupervisedNoveltyDetectionModule.score() or
        UnsupervisedNoveltyDetectionModule.__call__() on the first 60 examples
        of the trial, guaranteed to be nominal (type 0).

Preconditions:
    All arguments must be located on the same device. This includes
    all logistic regression models and tensors.
'''
def tune_logistic_regressions(
        case_1_logistic_regression,
        case_2_logistic_regression,
        case_3_logistic_regression,
        case_1_scores,
        case_2_scores,
        case_3_scores,
        case_1_labels,
        case_2_labels,
        case_3_labels,
        first_60_subject_novelty_scores,
        first_60_verb_novelty_scores,
        first_60_object_novelty_scores,
        first_60_activation_statistical_scores,
        epochs = 3000,
        quiet = True):
    # Separate first 60 score tuples into case 1, 2, and 3.
    first_60_case_1_scores = []
    first_60_case_2_scores = []
    first_60_case_3_scores = []
    for idx in range(len(first_60_subject_novelty_scores)):
        subject_score = first_60_subject_novelty_scores[idx]
        object_score = first_60_object_novelty_scores[idx]
        verb_score = first_60_verb_novelty_scores[idx]
        activation_statistical_score = torch.from_numpy(first_60_activation_statistical_scores[idx])
        
        # Add the scores to the appropriate case tensors
        if subject_score is not None and object_score is not None:
            # All boxes present; append scores to case 1
            first_60_case_1_scores.append(torch.stack((subject_score, verb_score, object_score, activation_statistical_score).to(torch.float).to(subject_score.device), dim = 0))
        if subject_score is not None:
            # At least the subject box is present; append subject and verb scores
            # to case 2
            first_60_case_2_scores.append(torch.stack((subject_score, verb_score, activation_statistical_score).to(torch.float).to(subject_score.device), dim = 0))
        if object_score is not None:
            # At least the object box is present; append object score to case 3
            first_60_case_3_scores.append(torch.stack((object_score, activation_statistical_score).to(torch.float).to(object_score.device), dim = 0))
    first_60_case_1_scores = torch.stack(first_60_case_1_scores, dim = 0)
    first_60_case_2_scores = torch.stack(first_60_case_2_scores, dim = 0)
    first_60_case_3_scores = torch.stack(first_60_case_3_scores, dim = 0)
    
    # Concatenate new (first 60) scores with saved scores (from the validation set)
    all_case_1_scores = torch.cat((case_1_scores, first_60_case_1_scores), dim = 0)
    all_case_2_scores = torch.cat((case_2_scores, first_60_case_2_scores), dim = 0)
    all_case_3_scores = torch.cat((case_3_scores, first_60_case_3_scores), dim = 0)

    # Concatenate on the new labels. Since the first 60 examples are all nominal,
    # their labels are all 0 (for type 0)
    all_case_1_labels = torch.cat((case_1_labels, torch.zeros(len(first_60_case_1_scores), dtype = torch.long, device = case_1_labels.device)), dim = 0)
    all_case_2_labels = torch.cat((case_2_labels, torch.zeros(len(first_60_case_2_scores), dtype = torch.long, device = case_2_labels.device)), dim = 0)
    all_case_3_labels = torch.cat((case_3_labels, torch.zeros(len(first_60_case_3_scores), dtype = torch.long, device = case_3_labels.device)), dim = 0)

    # And retrain the logistic regressions
    fit_logistic_regression(case_1_logistic_regression, all_case_1_scores, all_case_1_labels, epochs = epochs, quiet = quiet)
    fit_logistic_regression(case_2_logistic_regression, all_case_2_scores, all_case_2_labels, epochs = epochs, quiet = quiet)
    fit_logistic_regression(case_3_logistic_regression, all_case_3_scores, all_case_3_labels, epochs = epochs, quiet = quiet)
