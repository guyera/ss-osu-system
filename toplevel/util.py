import torch
from unsupervisednoveltydetection import UnsupervisedNoveltyDetector, ThresholdTrialLevelPType
from noveltydetection.utils import Case1LogisticRegression, Case2LogisticRegression, Case3LogisticRegression
from unsupervisednoveltydetection.common import ClassifierV2, ActivationStatisticalModel


class BatchContext:
    def __init__(self):
        self.query_mask = None
        self.feedback_mask = None
        self.p_ni = None
        self.subject_novelty_scores_u = None
        self.verb_novelty_scores_u = None
        self.object_novelty_scores_u = None
        self.image_paths = None
        self.novelty_dataset = None
        self.query_indices = None
        self.top_3 = None
        self.s_unknown = None
        self.v_unknown = None
        self.o_unknown = None
        self.p_type = None
        self.df = None
        self.round_id = None

    def reset(self):
        self.query_mask = None
        self.feedback_mask = None
        self.p_ni = None
        self.subject_novelty_scores_u = None
        self.verb_novelty_scores_u = None
        self.object_novelty_scores_u = None
        self.image_paths = None
        self.novelty_dataset = None
        self.query_indices = None
        self.top_3 = None
        self.s_unknown = None
        self.v_unknown = None
        self.o_unknown = None
        self.p_type = None
        self.df = None
        self.round_id = None

    def is_set(self):
        return self.p_ni is not None and self.subject_novelty_scores_u is not None and \
            self.verb_novelty_scores_u is not None and self.object_novelty_scores_u is not None and \
            self.image_paths is not None and self.novelty_dataset is not None and self.df is not None


class UnsupervisedNoveltyDetectionManager:
    def __init__(self, 
        pretrained_path, 
        num_subject_classes, 
        num_verb_classes,
        num_object_classes, 
        num_spatial_features,
        p_type_alpha):
       
        self.p_type_alpha = p_type_alpha

        classifier = ClassifierV2(256, num_subject_classes, num_object_classes, num_verb_classes, num_spatial_features)
        self.detector = UnsupervisedNoveltyDetector(classifier, num_subject_classes, num_object_classes, num_verb_classes)
        

        state_dict = torch.load(pretrained_path)
        self.detector.load_state_dict(state_dict['module'])
        self.detector = self.detector.to('cuda:0')
        if 'resnet' in pretrained_path:
            model_ = 'resnet'
        else:
            model_ = 'swin_t'
        self.activation_statistical_model = ActivationStatisticalModel(model_).to('cuda:0')
        self.activation_statistical_model.load_state_dict(state_dict['activation_statistical_model'])
        
        self.case_1_logistic_regression = Case1LogisticRegression()
        self.case_1_logistic_regression.load_state_dict(state_dict['case_1_logistic_regression'])
        self.case_1_logistic_regression = self.case_1_logistic_regression.to('cuda:0')
        
        self.case_2_logistic_regression = Case2LogisticRegression()
        self.case_2_logistic_regression.load_state_dict(state_dict['case_2_logistic_regression'])
        self.case_2_logistic_regression = self.case_2_logistic_regression.to('cuda:0')
        
        self.case_3_logistic_regression = Case3LogisticRegression()
        self.case_3_logistic_regression.load_state_dict(state_dict['case_3_logistic_regression'])
        self.case_3_logistic_regression = self.case_3_logistic_regression.to('cuda:0')          
            
    def get_calibrators(self):
        return self.case_1_logistic_regression, self.case_2_logistic_regression, self.case_3_logistic_regression

    def top3(self, backbone, dataset, batch_p_type, trial_p_type, scg_predictions, p_ni):
        spatial_features = []
        subject_box_features = []
        object_box_features = []
        verb_box_features = []
        
        with torch.no_grad():
            for example_spatial_features, _, _, _, _, _, _, example_subject_images, example_object_images, example_verb_images, whole_images in dataset:
                spatial_feature = example_spatial_features if example_spatial_features is not None else None
                subject_feature = backbone(example_subject_images.unsqueeze(0)).squeeze(0) if example_subject_images is not None else None
                object_feature = backbone(example_object_images.unsqueeze(0)).squeeze(0) if example_object_images is not None else None
                verb_feature = backbone(example_verb_images.unsqueeze(0)).squeeze(0) if example_verb_images is not None else None
                
                spatial_features.append(spatial_feature)
                subject_box_features.append(subject_feature)
                object_box_features.append(object_feature)
                verb_box_features.append(verb_feature)

            # op = ThresholdTrialLevelPType(trial_p_type, self.p_type_alpha)
            op = None
        
            top3 = self.detector.top3(spatial_features, subject_box_features, verb_box_features, 
                object_box_features, batch_p_type, op)
                        
        assert not any([any([torch.isnan(p[1]) for p in preds]) for preds in top3['top3']]), "NaNs in unsupervied detector's top-3"
                
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
                
        top3_merged = self.detector.merge_predictions(scg_predictions, top3['t67'], top3['cases'], top3['top3'], p_ni)
            
        return top3_merged   

    def score(self, backbone, dataset):
        spatial_features = []
        subject_box_features = []
        object_box_features = []
        verb_box_features = []
        # whole_image_features = []
        incident_activation_statistical_scores = []
        
        with torch.no_grad():
            for example_spatial_features, _, _, _, _, _, _, example_subject_images, example_object_images, example_verb_images, whole_images  in dataset:
                spatial_feature = example_spatial_features if example_spatial_features is not None else None
                subject_feature = backbone(example_subject_images.unsqueeze(0)).squeeze(0) if example_subject_images is not None else None
                object_feature = backbone(example_object_images.unsqueeze(0)).squeeze(0) if example_object_images is not None else None
                verb_feature = backbone(example_verb_images.unsqueeze(0)).squeeze(0) if example_verb_images is not None else None
                
                spatial_features.append(spatial_feature)
                subject_box_features.append(subject_feature)
                object_box_features.append(object_feature)
                verb_box_features.append(verb_feature)    

                whole_image_features = self.activation_statistical_model.compute_features(backbone, whole_images.unsqueeze(0))
                incident_activation_statistical_scores.append(self.activation_statistical_model.score(whole_image_features).squeeze(0))
        
                            
            results = self.detector.scores_and_p_t4(spatial_features, subject_box_features, verb_box_features, 
                object_box_features)
            
        return results, incident_activation_statistical_scores
