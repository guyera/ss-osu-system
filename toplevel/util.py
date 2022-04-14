import torch
from unsupervisednoveltydetection import UnsupervisedNoveltyDetector
from noveltydetection.utils import Case1LogisticRegression, Case2LogisticRegression, Case3LogisticRegression
from unsupervisednoveltydetection.common import ClassifierV2
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
        self.top_3 = None
        self.cases = None
        self.preds_is_nc = None
        self.s_unknown = None
        self.v_unknown = None
        self.o_unknown = None
        self.p_type = None

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
        self.top_3 = None
        self.cases = None
        self.preds_is_nc = None
        self.s_unknown = None
        self.v_unknown = None
        self.o_unknown = None
        self.p_type = None

    def is_set(self):
        return self.p_ni is not None and self.subject_novelty_scores_u is not None and \
            self.verb_novelty_scores_u is not None and self.object_novelty_scores_u is not None and \
            self.image_paths is not None and self.novelty_dataset is not None


class UnsupervisedNoveltyDetectionManager:
    def __init__(self, 
        pretrained_path, 
        num_subject_classes, 
        num_verb_classes,
        num_object_classes, 
        num_spatial_features):
       
        classifier = ClassifierV2(256, num_subject_classes, num_object_classes, num_verb_classes, num_spatial_features)
        self.detector = UnsupervisedNoveltyDetector(classifier, num_subject_classes, num_object_classes, num_verb_classes)
        state_dict = torch.load(pretrained_path)
        self.detector.load_state_dict(state_dict['module'])
        self.detector = self.detector.to('cuda:0')
        
        self.case_1_logistic_regression = Case1LogisticRegression()
        self.case_1_logistic_regression.load_state_dict(state_dict['case_1_logistic_regression'])
        self.case_1_logistic_regression = self.case_1_logistic_regression.to('cuda:0')
        
        self.case_2_logistic_regression = Case2LogisticRegression()
        self.case_2_logistic_regression.load_state_dict(state_dict['case_2_logistic_regression'])
        self.case_2_logistic_regression = self.case_2_logistic_regression.to('cuda:0')
        
        self.case_3_logistic_regression = Case3LogisticRegression()
        self.case_3_logistic_regression.load_state_dict(state_dict['case_3_logistic_regression'])
        self.case_3_logistic_regression = self.case_3_logistic_regression.to('cuda:0')
        
        self.case_1_scores = state_dict['case_1_scores'].to('cuda:0')
        self.case_1_labels = state_dict['case_1_labels'].to('cuda:0')

        self.case_2_scores = state_dict['case_2_scores'].to('cuda:0')
        self.case_2_labels = state_dict['case_2_labels'].to('cuda:0')

        self.case_3_scores = state_dict['case_3_scores'].to('cuda:0')
        self.case_3_labels = state_dict['case_3_labels'].to('cuda:0')
        
    # def reset_lr_calibrators(self):
    #     state_dict = torch.load(self.pretrained_path)

    #     self.case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression()
    #     self.case_1_logistic_regression.load_state_dict(state_dict['case_1_logistic_regression'])
    #     self.case_1_logistic_regression = self.case_1_logistic_regression.to('cuda:0')
        
    #     self.case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression()
    #     self.case_2_logistic_regression.load_state_dict(state_dict['case_2_logistic_regression'])
    #     self.case_2_logistic_regression = self.case_2_logistic_regression.to('cuda:0')
        
    #     self.case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression()
    #     self.case_3_logistic_regression.load_state_dict(state_dict['case_3_logistic_regression'])
    #     self.case_3_logistic_regression = self.case_3_logistic_regression.to('cuda:0')
        
    #     self.case_1_scores = state_dict['case_1_scores'].to('cuda:0')
    #     self.case_1_labels = state_dict['case_1_labels'].to('cuda:0')

    #     self.case_2_scores = state_dict['case_2_scores'].to('cuda:0')
    #     self.case_2_labels = state_dict['case_2_labels'].to('cuda:0')

    #     self.case_3_scores = state_dict['case_3_scores'].to('cuda:0')
    #     self.case_3_labels = state_dict['case_3_labels'].to('cuda:0')
            
    # def tune_lr_calibrators(self, subj_scores, verb_scores, obj_scores):
    #     assert len(subj_scores) == 60
    #     assert len(verb_scores) == 60
    #     assert len(obj_scores) == 60
    
    #     subj_scores = [s.clone().to('cuda:0') if s is not None else None for s in subj_scores]
    #     verb_scores = [s.clone().to('cuda:0') if s is not None else None for s in verb_scores]
    #     obj_scores = [s.clone().to('cuda:0') if s is not None else None for s in obj_scores]
        
    #     tune_logistic_regressions(self.case_1_logistic_regression, 
    #         self.case_2_logistic_regression,
    #         self.case_3_logistic_regression, 
    #         self.case_1_scores, 
    #         self.case_2_scores, 
    #         self.case_3_scores, 
    #         self.case_1_labels, 
    #         self.case_2_labels, 
    #         self.case_3_labels, 
    #         subj_scores, 
    #         verb_scores, 
    #         obj_scores)            
            
    def get_calibrators(self):
        return self.case_1_logistic_regression, self.case_2_logistic_regression, self.case_3_logistic_regression

    def top3(self, backbone, dataset, batch_p_type):
        spatial_features = []
        subject_box_features = []
        object_box_features = []
        verb_box_features = []
        
        for example_spatial_features, _, _, _, _, _, _, example_subject_images, example_object_images, example_verb_images in dataset:
            spatial_feature = example_spatial_features if example_spatial_features is not None else None
            subject_feature = backbone(example_subject_images.unsqueeze(0)).squeeze(0) if example_subject_images is not None else None
            object_feature = backbone(example_object_images.unsqueeze(0)).squeeze(0) if example_object_images is not None else None
            verb_feature = backbone(example_verb_images.unsqueeze(0)).squeeze(0) if example_verb_images is not None else None
            
            spatial_features.append(spatial_feature)
            subject_box_features.append(subject_feature)
            object_box_features.append(object_feature)
            verb_box_features.append(verb_feature)

        with torch.no_grad():
            top3 = self.detector.top3(spatial_features, subject_box_features, verb_box_features, 
                object_box_features, batch_p_type)

        assert not any([any([torch.isnan(p[1]).item() for p in preds]) for preds in top3['top3']]), "NaNs in unsupervied detector's top-3"
                
        for i, p in enumerate(top3['top3']):
            new_p = []
            for j in range(3):
                if j < len(p):
                    e = (p[j][0], 0.0 if torch.isnan(p[j][1]) or torch.isinf(p[j][1]) else p[j][1].item())
                    new_svo = list(e[0])

                    for k in range(3):
                        if new_svo[k] is not None and type(new_svo[k]) == torch.Tensor:
                            new_svo[k] = new_svo[k].item()
                    
                    e = (tuple(new_svo), e[1])

                else: 
                    e = ((int(-100), int(-100), int(-100)), -100)

                new_p.append(e)

            top3['top3'][i] = new_p
            
        return top3['top3']    

    def score(self, backbone, dataset):
        spatial_features = []
        subject_box_features = []
        object_box_features = []
        verb_box_features = []
        
        for example_spatial_features, _, _, _, _, _, _, example_subject_images, example_object_images, example_verb_images in dataset:
            spatial_feature = example_spatial_features if example_spatial_features is not None else None
            subject_feature = backbone(example_subject_images.unsqueeze(0)).squeeze(0) if example_subject_images is not None else None
            object_feature = backbone(example_object_images.unsqueeze(0)).squeeze(0) if example_object_images is not None else None
            verb_feature = backbone(example_verb_images.unsqueeze(0)).squeeze(0) if example_verb_images is not None else None
            
            spatial_features.append(spatial_feature)
            subject_box_features.append(subject_feature)
            object_box_features.append(object_feature)
            verb_box_features.append(verb_feature)

        with torch.no_grad():
            results = self.detector.scores_and_p_t4(spatial_features, subject_box_features, verb_box_features, 
                object_box_features)

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

    def reset(self):
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

    def train(self):
        assert self.subject_features is not None, "subject_features was None"
        assert self.verb_features is not None, "verb_features was None"
        assert self.object_features is not None, "object_features was None"
        
        aucs, novelty_scores, models = train_supervised_models(self.subject_features, 
            self.verb_features, 
            self.object_features, 
            self.subject_novelty_scores.reshape((-1, 1)),
            self.verb_novelty_scores.reshape((-1, 1)),
            self.object_novelty_scores.reshape((-1, 1)),
            self.subject_labels.reshape((-1, 1)),
            self.verb_labels.reshape((-1, 1)),
            self.object_labels.reshape((-1, 1)))

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
                subject_features.append(subject_appearance_features.view(-1))
            else:
                subject_features.append(None)

            if verb_appearance_features is not None:
                verb_features.append(torch.hstack([spatial_features.view(-1), verb_appearance_features.view(-1)]))
            else:
                verb_features.append(None)

            if object_appearance_features is not None:
                object_features.append(object_appearance_features.view(-1))
            else:
                object_features.append(None)

        assert len(subject_features) == len(object_features) == len(verb_features)
        assert len(self.models) > 0, "no supervised models"

        subject_nov_scores_s, verb_nov_scores_s, object_nov_scores_s = eval_supervised(subject_features, 
            verb_features, 
            object_features, 
            subject_novelty_scores, 
            verb_novelty_scores, 
            object_novelty_scores, 
            self.models)

        assert len(subject_nov_scores_s) == len(verb_nov_scores_s) == len(object_nov_scores_s)

        subject_nov_scores_s = [torch.tensor(s, device='cuda:0') if s is not None else None for s in subject_nov_scores_s]
        verb_nov_scores_s = [torch.tensor(s, device='cuda:0') if s is not None else None for s in verb_nov_scores_s]
        object_nov_scores_s = [torch.tensor(s, device='cuda:0') if s is not None else None for s in object_nov_scores_s]

        return subject_nov_scores_s, verb_nov_scores_s, object_nov_scores_s

    def feedback_callback(self, subject_features, verb_features, object_features, 
        subject_novelty_scores, verb_novelty_scores, object_novelty_scores,
        subject_labels, verb_labels, object_labels):
        
        assert subject_features.shape[0] == subject_labels.shape[0], "Number of subject instances must be equal to number of subject labels"
        assert verb_features.shape[0] == verb_labels.shape[0], "Number of verb instances must be equal to number of verb labels"
        assert object_features.shape[0] == object_labels.shape[0], "Number of object instances must be equal to number of object labels"

        assert subject_features.numel() == 0 or subject_features.shape[1] == self.num_appearance_features, "Number of appearance features is invalid."
        assert verb_features.numel() == 0 or  verb_features.shape[1] == self.num_verb_features, "Number of verb features is invalid."
        assert object_features.numel() == 0 or object_features.shape[1] == self.num_appearance_features, "Number of object features is invalid."

        if subject_features.numel() > 0:
            self.subject_features = torch.vstack([self.subject_features, subject_features]) if self.subject_features is not None else subject_features
        
        if verb_features.numel() > 0:
            self.verb_features = torch.vstack([self.verb_features, verb_features]) if self.verb_features is not None else verb_features
        
        if object_features.numel() > 0:
            self.object_features = torch.vstack([self.object_features, object_features]) if self.object_features is not None else object_features

        subject_labels = subject_labels.view(-1)
        verb_labels = verb_labels.view(-1)
        object_labels = object_labels.view(-1)

        self.subject_labels = torch.cat([self.subject_labels, subject_labels]) if self.subject_labels is not None else subject_labels.view(-1)
        self.verb_labels = torch.cat([self.verb_labels, verb_labels]) if self.verb_labels is not None else verb_labels.view(-1)
        self.object_labels = torch.cat([self.object_labels, object_labels]) if self.object_labels is not None else object_labels.view(-1)

        self.subject_novelty_scores = torch.cat([self.subject_novelty_scores, subject_novelty_scores.view(-1)]) if \
            self.subject_novelty_scores is not None else subject_novelty_scores.view(-1)

        self.verb_novelty_scores = torch.cat([self.verb_novelty_scores, verb_novelty_scores.view(-1)]) if \
            self.verb_novelty_scores is not None else verb_novelty_scores.view(-1)

        self.object_novelty_scores = torch.cat([self.object_novelty_scores, object_novelty_scores.view(-1)]) if \
            self.object_novelty_scores is not None else object_novelty_scores.view(-1)
