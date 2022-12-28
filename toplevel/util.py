import torch
from tupleprediction import\
    Case1LogisticRegression,\
    Case2LogisticRegression,\
    Case3LogisticRegression,\
    TuplePredictor
from scoring import ActivationStatisticalModel
from boxclassifier import ClassifierV2

import os

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
        model_dir,
        backbone_architecture,
        num_subject_classes, 
        num_verb_classes,
        num_object_classes, 
        num_spatial_features,
        p_type_alpha):
       
        self.p_type_alpha = p_type_alpha

        self.classifier = ClassifierV2(256, num_subject_classes, num_object_classes, num_verb_classes, num_spatial_features)
        self.tuple_predictor = TuplePredictor(num_subject_classes, num_object_classes, num_verb_classes)
        
        pretrained_path = os.path.join(
            model_dir,
            backbone_architecture.value['name']
        )
        classifier_state_dict = torch.load(os.path.join(
            pretrained_path,
            'classifier.pth'
        ))
        tuple_prediction_state_dict = torch.load(os.path.join(
            pretrained_path,
            'tuple-prediction.pth'
        ))
        self.classifier.load_state_dict(classifier_state_dict)
        self.classifier = self.classifier.to('cuda:0')
        self.tuple_predictor.load_state_dict(tuple_prediction_state_dict)
        self.tuple_predictor = self.tuple_predictor.to('cuda:0')
        self.activation_statistical_model = ActivationStatisticalModel(backbone_architecture).to('cuda:0')
        self.activation_statistical_model.load_state_dict(tuple_prediction_state_dict['activation_statistical_model'])
        
        self.case_1_logistic_regression = Case1LogisticRegression()
        self.case_1_logistic_regression.load_state_dict(tuple_prediction_state_dict['case_1_logistic_regression'])
        self.case_1_logistic_regression = self.case_1_logistic_regression.to('cuda:0')
        
        self.case_2_logistic_regression = Case2LogisticRegression()
        self.case_2_logistic_regression.load_state_dict(tuple_prediction_state_dict['case_2_logistic_regression'])
        self.case_2_logistic_regression = self.case_2_logistic_regression.to('cuda:0')
        
        self.case_3_logistic_regression = Case3LogisticRegression()
        self.case_3_logistic_regression.load_state_dict(tuple_prediction_state_dict['case_3_logistic_regression'])
        self.case_3_logistic_regression = self.case_3_logistic_regression.to('cuda:0')          
            
    def get_calibrators(self):
        return self.case_1_logistic_regression, self.case_2_logistic_regression, self.case_3_logistic_regression

    def top3(self, subject_probs, verb_probs, object_probs, batch_p_type, scg_predictions, p_ni):
        with torch.no_grad():
            top3 = self.tuple_predictor.top3(subject_probs, verb_probs, object_probs, batch_p_type)

        assert not any([any([torch.isnan(p[1]) for p in preds]) for preds in top3['top3']]), "NaNs in tuple predictor's top-3"

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

        top3_merged = self.tuple_predictor.merge_predictions(scg_predictions, top3['t67'], top3['cases'], top3['top3'], p_ni)

        return top3_merged   

    def score(self, backbone, dataset):
        spatial_features = []
        subject_box_features = []
        object_box_features = []
        verb_box_features = []
        # whole_image_features = []
        incident_activation_statistical_scores = []
        
        # TODO tensorize by case
        with torch.no_grad():
            for example_spatial_features, _, _, _, example_subject_images, example_object_images, example_verb_images, whole_images  in dataset:
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

            results = {}
            subject_probs, subject_scores, verb_probs, verb_scores,\
                object_probs, object_scores = self.classifier.predict_score(
                        spatial_features,
                        subject_box_features,
                        verb_box_features,
                        object_box_features
                    )
            results['subject_novelty_score'] = subject_scores
            results['verb_novelty_score'] = verb_scores
            results['object_novelty_score'] = object_scores
            results['subject_probs'] = subject_probs
            results['verb_probs'] = verb_probs
            results['object_probs'] = object_probs
            p_t4 = self.tuple_predictor.p_t4(subject_probs, verb_probs, object_probs)
            results['p_t4'] = p_t4

        return results, incident_activation_statistical_scores
