import torch

from boxclassifier._utils import _state_dict, _load_state_dict

class TemperatureScaler(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, logits):
        return logits / self.temperature


class ConfidenceCalibrator:
    def __init__(self):
        self.device = 'cpu'
        self.reset()

    def reset(self):
        self.species_calibrator = TemperatureScaler().to(self.device)
        self.activity_calibrator = TemperatureScaler().to(self.device)

    def calibrate_species(self, logits):
        calibrated_logits = self.species_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate_activity(self, logits):
        calibrated_logits = self.activity_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate(self, species_logits, activity_logits):
        calibrated_species_logits = self.species_calibrator(species_logits)
        calibrated_activity_logits = self.activity_calibrator(activity_logits)
        
        species_predictions = torch.nn.functional.softmax(calibrated_species_logits, dim = 1)
        activity_predictions = torch.nn.functional.softmax(calibrated_activity_logits, dim = 1)
        
        return species_predictions, activity_predictions
    
    def to(self, device):
        self.device = device
        self.species_calibrator = self.species_calibrator.to(device)
        self.activity_calibrator = self.activity_calibrator.to(device)
        return self
    
    def state_dict(self, *args, **kwargs):
        state_dict = {}
        state_dict['species_calibrator'] =\
            self.species_calibrator.state_dict(*args, **kwargs)
        state_dict['activity_calibrator'] =\
            self.activity_calibrator.state_dict(*args, **kwargs)
        return state_dict

    def load_state_dict(self, state_dict):
        self.species_calibrator.load_state_dict(
            state_dict['species_calibrator']
        )
        self.activity_calibrator.load_state_dict(
            state_dict['activity_calibrator']
        )
