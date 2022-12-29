import torch

from boxclassifier._utils import _state_dict, _load_state_dict

class ClassifierV2:
    def __init__(
            self,
            bottleneck_dim,
            n_species_cls,
            n_activity_cls,
            spatial_encoding_dim):
        self.device = 'cpu'
        self.bottleneck_dim = bottleneck_dim
        self.n_species_cls = n_species_cls
        self.n_activity_cls = n_activity_cls
        self.spatial_encoding_dim = spatial_encoding_dim
        self.reset()

    def reset(self):
        self.species_classifier = torch.nn.Linear(self.bottleneck_dim, self.n_species_cls - 1).to(self.device)
        self.activity_classifier = torch.nn.Linear(self.bottleneck_dim + self.spatial_encoding_dim, self.n_activity_cls - 1).to(self.device)
    
    def predict(self, box_features):
        self.species_classifier.eval()
        self.activity_classifier.eval()

        features = torch.flatten(
            box_features,
            start_dim=1
        )

        species_logits = self.species_classifier(box_features)
        activity_logits = self.activity_classifier(box_features)
        return species_logits, activity_logits

    def predict_species(self, box_features):
        self.species_classifier.eval()
        logits = self.species_classifier(torch.flatten(features, start_dim=1))
        return logits

    def predict_activity(self, features):
        self.activity_classifier.eval()
        logits = self.activity_classifier(torch.flatten(features, start_dim=1))
        return logits

    def to(self, device):
        self.device = device
        self.species_classifier = self.species_classifier.to(device)
        self.activity_classifier = self.activity_classifier.to(device)
        return self
    
    def state_dict(self):
        state_dict = {}
        state_dict['species_classifier'] = _state_dict(self.species_classifier)
        state_dict['activity_classifier'] = _state_dict(self.activity_classifier)
        return state_dict

    def load_state_dict(self, state_dict):
        _load_state_dict(
            self.species_classifier,
            state_dict['species_classifier']
        )
        _load_state_dict(
            self.activity_classifier,
            state_dict['activity_classifier']
        )
