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
        self.fc = torch.nn.Linear(3, 4)

    def forward(self, x):
        return self.fc(x)

class Case2LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x)

class Case3LogisticRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.fc(x)

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

def compute_probability_novelty(
        subject_scores,
        verb_scores,
        object_scores,
        case_1_logistic_regression,
        case_2_logistic_regression,
        case_3_logistic_regression,
        ignore_t2_in_pni = True):
    p_type = []
    p_n = []
    for idx in range(len(subject_scores)):
        subject_score = subject_scores[idx]
        object_score = object_scores[idx]
        verb_score = verb_scores[idx]
        
        if subject_score is not None and object_score is not None:
            # Case 1
            scores = torch.stack((subject_score, verb_score, object_score), dim = 0).unsqueeze(0)
            logits = case_1_logistic_regression(scores).squeeze(0)
            # To compute p_n, skip P(type = 0) and sum remaining probabilities
            if ignore_t2_in_pni:
                cur_p_n = torch.nn.functional.softmax(logits, dim = 0)[[1, 3]].sum()
            else:
                cur_p_n = torch.nn.functional.softmax(logits, dim = 0)[1:].sum()
            # To compute p_type, we'll remove the type = 0 logit, and then
            # normalize to get p_type[i] = P(type = i) for i in {1, 2, 3}.
            cur_partial_p_type = torch.nn.functional.softmax(logits[1:], dim = 0)
            # And, of course, we'll set p_type[3] to zero
            cur_p_type = torch.cat((cur_partial_p_type, torch.zeros(1, device = cur_partial_p_type.device)), dim = 0)
        elif subject_score is not None:
            # Case 2
            scores = torch.stack((subject_score, verb_score), dim = 0).unsqueeze(0)
            logits = case_2_logistic_regression(scores).squeeze(0)
            # To compute p_n, skip P(type = 0) and sum remaining probabilities
            if ignore_t2_in_pni:
                cur_p_n = torch.nn.functional.softmax(logits, dim = 0)[1]
            else:
                cur_p_n = torch.nn.functional.softmax(logits, dim = 0)[1:].sum()
            # To compute p_type, we'll remove the type = 0 logit, and then
            # normalize to get p_type[i] = P(type = i) for i in {1, 2}.
            cur_partial_p_type = torch.nn.functional.softmax(logits[1:], dim = 0)
            # And, of course, we'll set p_type[2] and p_type[3] to zero
            cur_p_type = torch.cat((cur_partial_p_type, torch.zeros(2, device = cur_partial_p_type.device)), dim = 0)
        else:
            # Case 3
            scores = object_score.unsqueeze(0).unsqueeze(0)
            logits = case_3_logistic_regression(scores).squeeze(0)
            # To compute p_n, skip P(type = 0), setting p_n equal to the type 3
            # probability
            cur_p_n = torch.nn.functional.softmax(logits, dim = 0)[1]
            # In case 3, novelty type can only be 3 or 4. Since we're ignoring
            # type 4, that leaves 100% probability for type 3.
            cur_p_type = torch.zeros(4, device = cur_p_n.device)
            cur_p_type[2] = 1.0

        p_type.append(cur_p_type)
        p_n.append(cur_p_n)

    p_type = torch.stack(p_type, dim = 0)
    p_n = torch.stack(p_n, dim = 0)
    
    return p_type, p_n

def fit_logistic_regression(logistic_regression, scores, labels, epochs = 3000, quiet = True):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(logistic_regression.parameters(), lr = 0.01, momentum = 0.9)
    
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
        
        # Add the scores to the appropriate case tensors
        if subject_score is not None and object_score is not None:
            # All boxes present; append scores to case 1
            first_60_case_1_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
        if subject_score is not None:
            # At least the subject box is present; append subject and verb scores
            # to case 2
            first_60_case_2_scores.append(torch.stack((subject_score, verb_score), dim = 0))
        if object_score is not None:
            # At least the object box is present; append object score to case 3
            first_60_case_3_scores.append(object_score.unsqueeze(0))
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
