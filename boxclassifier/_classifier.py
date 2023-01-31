import re

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from boxclassifier._utils import _state_dict, _load_state_dict

class ClassifierV2:
    def __init__(
            self,
            bottleneck_dim,
            n_species_cls,
            n_activity_cls):
        self.device = 'cpu'
        self.bottleneck_dim = bottleneck_dim
        self.n_species_cls = n_species_cls
        self.n_activity_cls = n_activity_cls
        self.reset()

    def reset(self, bottleneck_dim=None):
        if bottleneck_dim is not None:
            self.bottleneck_dim = bottleneck_dim
        self.species_classifier = torch.nn.Linear(self.bottleneck_dim, self.n_species_cls).to(self.device)
        self.activity_classifier = torch.nn.Linear(self.bottleneck_dim, self.n_activity_cls).to(self.device)

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
    
    def state_dict(self, *args, **kwargs):
        state_dict = {}
        state_dict['species_classifier'] =\
            self.species_classifier.state_dict(*args, **kwargs)
        state_dict['activity_classifier'] =\
            self.activity_classifier.state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(self, state_dict):
        if isinstance(self.species_classifier, DDP):
            self.species_classifier.load_state_dict(
                state_dict['species_classifier']
            )
        else:
            species_classifier_state_dict = {
                re.sub('^module\.', '', k): v for\
                    k, v in state_dict['species_classifier'].items()
            }
            self.species_classifier.load_state_dict(
                species_classifier_state_dict
            )

        if isinstance(self.activity_classifier, DDP):
            self.activity_classifier.load_state_dict(
                state_dict['activity_classifier']
            )
        else:
            activity_classifier_state_dict = {
                re.sub('^module\.', '', k): v for\
                    k, v in state_dict['activity_classifier'].items()
            }
            self.activity_classifier.load_state_dict(
                activity_classifier_state_dict
            )

    def ddp(self, device_ids=None):
        self.species_classifier = DDP(
            self.species_classifier,
            device_ids=device_ids
        )
        self.activity_classifier = DDP(
            self.activity_classifier,
            device_ids=device_ids
        )
