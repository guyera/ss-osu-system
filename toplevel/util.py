import torch
from unsupervisednoveltydetection import UnsupervisedNoveltyDetector
from noveltydetection.utils import ScoreContext
from adaptation.supervised_anomaly_detectors import train_supervised_models, eval_supervised


class BatchContext:
    def __init__(self):
        self.query_mask = None
        self.feedback_mask = None
        self.p_ni = None
        self.subject_novelty_scores_u = None
        self.verb_novelty_scores_u = None
        self.object_novelty_scores_u = None
        self.subject_novelty_scores_best = None
        self.verb_novelty_scores_best = None
        self.object_novelty_scores_best = None
        self.image_paths = None
        self.novelty_dataset = None
        self.query_indices = None
        self.top_1 = None
        self.cases = None
        self.preds_is_nc = None

    def reset(self):
        self.query_mask = None
        self.feedback_mask = None
        self.p_ni = None
        self.subject_novelty_scores_u = None
        self.verb_novelty_scores_u = None
        self.object_novelty_scores_u = None
        self.subject_novelty_scores_best = None
        self.verb_novelty_scores_best = None
        self.object_novelty_scores_best = None
        self.image_paths = None
        self.novelty_dataset = None
        self.query_indices = None
        self.top_1 = None
        self.cases = None
        self.preds_is_nc = None

    def is_set(self):
        return self.p_ni is not None and self.subject_novelty_scores_u is not None and \
            self.verb_novelty_scores_u is not None and self.object_novelty_scores_u is not None and \
            self.image_paths is not None and self.novelty_dataset is not None


class UnsupervisedNoveltyDetectionManager:
    def __init__(self, 
        pretrained_path, 
        num_subject_classes, 
        num_object_classes, 
        num_verb_classes,
        num_appearance_features, 
        num_verb_features):

        self.num_appearance_features = num_appearance_features
        self.num_verb_features = num_verb_features

        self.detector = UnsupervisedNoveltyDetector(num_appearance_features=self.num_appearance_features, 
            num_verb_features=self.num_verb_features, 
            num_hidden_nodes=1024, 
            num_subj_cls=num_subject_classes, 
            num_obj_cls=num_object_classes, 
            num_action_cls=num_verb_classes)

        self.detector = self.detector.to('cuda:0')

        state_dict = torch.load(pretrained_path)
        self.detector.load_state_dict(state_dict['module'])

        self.subject_score_context = ScoreContext(ScoreContext.Source.UNSUPERVISED, None, None)
        self.object_score_context = ScoreContext(ScoreContext.Source.UNSUPERVISED, None, None)
        self.verb_score_context = ScoreContext(ScoreContext.Source.UNSUPERVISED, None, None)
        
        self.subject_score_context.load_state_dict(state_dict['subject_score_context'])
        self.object_score_context.load_state_dict(state_dict['object_score_context'])
        self.verb_score_context.load_state_dict(state_dict['verb_score_context'])

    def get_svo_detectors_auc(self):
        subj_auc = self.subject_score_context.compute_partial_auc()
        verb_auc = self.verb_score_context.compute_partial_auc()
        obj_auc = self.object_score_context.compute_partial_auc()

        return subj_auc, verb_auc, obj_auc

    def get_score_contexts(self):
        return self.subject_score_context, self.verb_score_context, self.object_score_context

    def feedback_callback(self, subject_novelty_scores_non_novel, subject_novelty_scores_novel, 
        verb_novelty_scores_non_novel, verb_novelty_scores_novel,
        object_novelty_scores_non_novel, object_novelty_scores_novel):

        self.subject_score_context.add_nominal_scores(subject_novelty_scores_non_novel)
        self.subject_score_context.add_novel_scores(subject_novelty_scores_novel)

        self.verb_score_context.add_nominal_scores(verb_novelty_scores_non_novel)
        self.verb_score_context.add_novel_scores(verb_novelty_scores_novel)

        self.object_score_context.add_nominal_scores(object_novelty_scores_non_novel)
        self.object_score_context.add_novel_scores(object_novelty_scores_novel)

    def score(self, dataset, p_type_dist):
        all_spatial_features = []
        all_subject_appearance_features = []
        all_object_appearance_features = []
        all_verb_appearance_features = []

        for spatial_features, subj_app_features, obj_app_features, verb_app_features, _, _, _ in dataset:
            all_spatial_features.append(spatial_features)
            all_subject_appearance_features.append(subj_app_features)
            all_object_appearance_features.append(obj_app_features)
            all_verb_appearance_features.append(verb_app_features)

        with torch.no_grad():
            results = self.detector(all_spatial_features, all_subject_appearance_features, 
                all_verb_appearance_features, all_object_appearance_features, p_type_dist)

        for i, p in enumerate(results['top3']):
            new_p = []
            for j in range(3):
                e = (p[j][0], p[j][1].item()) if j < len(p) else ((-100, -100, -100), -100)
                new_p.append(e)

            results['top3'][i] = new_p

        return results


class SupervisedNoveltyDetectionManager:
    def __init__(self, num_appearance_features, num_verb_features):
        self.subject_features = None
        self.object_features = None
        self.verb_features = None
        self.subject_labels = None
        self.object_labels = None
        self.verb_labels = None
        self.subject_novelty_scores = None
        self.verb_novelty_scores = None
        self.object_novelty_scores = None
        self.models = []
        self.latest_AUC_scores = None
        self.num_appearance_features = num_appearance_features
        self.num_verb_features = num_verb_features

    def train(self):
        # print(f'\nsubj_f: {self.subject_features}\n, verb_f: {self.verb_features}\n, obj_f: {self.object_features}\n')
        # print(f'subj_n: {self.subject_novelty_scores}, verb:N: {self.verb_novelty_scores}, obj_n: {self.object_novelty_scores}\n')            
        # print(f'subj_l: {self.subject_labels}, verb_l: {self.verb_labels}, obj_l: {self.object_labels}\n')

        # torch.save({'S_X': self.subject_features, 
        #     'V_X': self.verb_features, 
        #     'O_X': self.object_features, 
        #     'S_a': self.subject_novelty_scores,
        #     'V_a': self.verb_novelty_scores,
        #     'O_a': self.object_novelty_scores, 
        #     'S_y': self.subject_labels, 
        #     'V_y': self.verb_labels, 
        #     'O_y': self.object_labels}, 'inputs.pth')

        # print('save inputs.')

        aucs, novelty_scores, models = train_supervised_models(self.subject_features, 
            self.verb_features, 
            self.object_features, 
            self.subject_novelty_scores.reshape((-1, 1)),
            self.verb_novelty_scores.reshape((-1, 1)),
            self.object_novelty_scores.reshape((-1, 1)),
            self.subject_labels.reshape((-1, 1)),
            self.verb_labels.reshape((-1, 1)),
            self.object_labels.reshape((-1, 1)))

        print('qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq')

        self.models = models
        self.latest_AUC_scores = aucs

        return novelty_scores

    def get_svo_detectors_auc(self):
        return self.latest_AUC_scores

    def score(self, dataset, subject_novelty_scores, verb_novelty_scores, object_novelty_scores):
        subject_features = []
        verb_features = []
        object_features = []

        for spatial_features, subject_appearance_features,  \
            object_appearance_features, verb_appearance_features, _, _, _ in dataset:

            if subject_appearance_features is not None:
                subject_features.append(subject_appearance_features)
            else:
                subject_features.append(None)

            if verb_appearance_features is not None:
                verb_features.append(torch.hstack([spatial_features, verb_appearance_features]))
            else:
                verb_features.append(None)

            if object_appearance_features is not None:
                object_features.append(object_appearance_features)
            else:
                object_features.append(None)

        subject_nov_scores_s, verb_nov_scores_s, object_nov_scores_s = eval_supervised(subject_features, 
            verb_features, 
            object_features, 
            subject_novelty_scores, 
            verb_novelty_scores, 
            object_novelty_scores, 
            self.models)

        return subject_nov_scores_s, verb_nov_scores_s, object_nov_scores_s

    def feedback_callback(self, subject_features, verb_features, object_features, 
        subject_novelty_scores, verb_novelty_scores, object_novelty_scores,
        subject_labels, verb_labels, object_labels):
        
        assert subject_features.shape[0] == subject_labels.shape[0], "Number of subject instances must be equal to number of subject labels"
        assert verb_features.shape[0] == verb_labels.shape[0], "Number of verb instances must be equal to number of verb labels"
        assert object_features.shape[0] == object_labels.shape[0], "Number of object instances must be equal to number of object labels"

        assert subject_features.shape[1] == self.num_appearance_features, "Number of appearance features is invalid."
        assert verb_features.shape[1] == self.num_verb_features, "Number of verb features is invalid."
        assert object_features.shape[1] == self.num_appearance_features, "Number of object features is invalid."

        self.subject_features = torch.vstack([self.subject_features, subject_features]) if self.subject_features is not None else subject_features
        self.verb_features = torch.vstack([self.verb_features, verb_features]) if self.verb_features is not None else verb_features
        self.object_features = torch.vstack([self.object_features, object_features]) if self.object_features is not None else object_features

        self.subject_labels = torch.vstack([self.subject_labels, subject_labels]) if self.subject_labels is not None else subject_labels 
        self.verb_labels = torch.vstack([self.verb_labels, verb_labels]) if self.verb_labels is not None else verb_labels
        self.object_labels = torch.vstack([self.object_labels, object_labels]) if self.object_labels is not None else object_labels

        self.subject_novelty_scores = torch.vstack([self.subject_novelty_scores, subject_novelty_scores]) \
            if self.subject_novelty_scores is not None else torch.tensor(subject_novelty_scores)
        self.verb_novelty_scores = torch.vstack([self.verb_novelty_scores, verb_novelty_scores]) \
            if self.verb_novelty_scores is not None else torch.tensor(verb_novelty_scores)
        self.object_novelty_scores = torch.vstack([self.object_novelty_scores, object_novelty_scores]) \
            if self.object_novelty_scores is not None else torch.tensor(object_novelty_scores)