import torch
from tupleprediction import NoveltyTypeClassifier, TuplePredictor
from scoring import\
    ActivationStatisticalModel,\
    make_logit_scorer,\
    CompositeScorer
from boxclassifier import ClassifierV2, ConfidenceCalibrator

import os

class BatchContext:
    def __init__(self):
        self.reset()

    def reset(self):
        self.query_mask = None
        self.p_ni = None
        self.image_paths = None
        self.bboxes = None
        self.query_indices = None
        self.predictions = None
        self.p_type = None

    def is_set(self):
        set_attrs = [
            self.p_ni,
            self.image_paths,
            self.bboxes,
            self.predictions,
            self.p_type
        ]
        return all([x is not None for x in set_attrs])


class UnsupervisedNoveltyDetectionManager:
    def __init__(self, 
        model_dir,
        backbone_architecture,
        n_species_cls,
        n_activity_cls,
        n_known_species_cls,
        n_known_activity_cls,
        device):
       
        self.tuple_predictor = TuplePredictor(
            n_species_cls,
            n_activity_cls,
            n_known_species_cls,
            n_known_activity_cls
        )

        pretrained_path = os.path.join(
            model_dir,
            backbone_architecture.value['name']
        )

        classifier_state_dict = torch.load(
            os.path.join(
                pretrained_path,
                'classifier.pth'
            ),
            map_location=device
        )
        self.classifier = ClassifierV2(
            256,
            n_species_cls,
            n_activity_cls
        ).to(device)
        self.classifier.load_state_dict(classifier_state_dict)

        confidence_calibrator_state_dict = torch.load(
            os.path.join(
                pretrained_path,
                'confidence-calibrator.pth'
            ),
            map_location=device
        )
        self.confidence_calibrator = ConfidenceCalibrator().to(device)
        self.confidence_calibrator.load_state_dict(
            confidence_calibrator_state_dict
        )

        tuple_prediction_state_dict = torch.load(
            os.path.join(
                pretrained_path,
                'tuple-prediction.pth'
            ),
            map_location=device
        )

        self.activation_statistical_model = ActivationStatisticalModel(
            backbone_architecture
        ).to(device)
        self.activation_statistical_model.load_state_dict(
            tuple_prediction_state_dict['activation_statistical_model']
        )
        self.scorer =\
            make_logit_scorer(n_known_species_cls, n_known_activity_cls)

        self.novelty_type_classifier = NoveltyTypeClassifier(
            self.scorer.n_scores() + 1 # + 1 for activation statistics
        ).to(device)
        self.novelty_type_classifier.load_state_dict(
            tuple_prediction_state_dict['novelty_type_classifier']
        )
        self.novelty_type_classifier.eval()

    def predict(self, species_probs, activity_probs, p_type):
        return self.tuple_predictor.predict(
            species_probs,
            activity_probs,
            p_type
        )

    def score(self, backbone, dataset):
        scores = []
        species_probs = []
        activity_probs = []
        with torch.no_grad():
            for _, _, _, box_images, whole_images in dataset:
                if len(box_images) == 0:
                    scores.append(None)
                    species_probs.append(None)
                    activity_probs.append(None)
                    continue

                box_images = box_images.to(backbone.device)
                whole_images = whole_images.to(backbone.device)
                box_features = backbone(box_images)
                whole_image_features =\
                    self.activation_statistical_model.compute_features(
                        backbone,
                        whole_images[None]
                    )
                env_scores =\
                    self.activation_statistical_model.score(
                        whole_image_features
                    ).squeeze(0)

                cur_species_logits, cur_activity_logits =\
                    self.classifier.predict(box_features)
                cur_species_probs, cur_activity_probs =\
                    self.confidence_calibrator.calibrate(
                        cur_species_logits,
                        cur_activity_logits
                    )
                species_probs.append(cur_species_probs)
                activity_probs.append(cur_activity_probs)

                logit_scores = self.scorer(
                    cur_species_logits,
                    cur_activity_logits,
                    [cur_species_logits.shape[0]]
                ).squeeze(0)
                cur_scores = torch.cat((logit_scores, env_scores), dim=0)
                scores.append(cur_scores)

            results = {}
            results['scores'] = scores
            results['species_probs'] = species_probs
            results['activity_probs'] = activity_probs

        return results
