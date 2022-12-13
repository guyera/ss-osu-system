import torch

from unsupervisednoveltydetection._utils import _state_dict, _load_state_dict

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
        self.subject_calibrator = TemperatureScaler().to(self.device)
        self.object_calibrator = TemperatureScaler().to(self.device)
        self.verb_calibrator = TemperatureScaler().to(self.device)

    def calibrate_subject(self, logits):
        calibrated_logits = self.subject_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate_object(self, logits):
        calibrated_logits = self.object_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate_verb(self, logits):
        calibrated_logits = self.verb_calibrator(logits)
        predictions = torch.nn.functional.softmax(calibrated_logits, dim = 1)
        return predictions

    def calibrate(self, subject_logits, object_logits, verb_logits):
        calibrated_subject_logits = self.subject_calibrator(subject_logits)
        calibrated_object_logits = self.object_calibrator(object_logits)
        calibrated_verb_logits = self.verb_calibrator(verb_logits)
        
        subject_predictions = torch.nn.functional.softmax(calibrated_subject_logits, dim = 1)
        object_predictions = torch.nn.functional.softmax(calibrated_object_logits, dim = 1)
        verb_predictions = torch.nn.functional.softmax(calibrated_verb_logits, dim = 1)
        
        return subject_predictions, object_predictions, verb_predictions
    
    def to(self, device):
        self.device = device
        self.subject_calibrator = self.subject_calibrator.to(device)
        self.object_calibrator = self.object_calibrator.to(device)
        self.verb_calibrator = self.verb_calibrator.to(device)
        return self
    
    def state_dict(self):
        state_dict = {}
        state_dict['subject_calibrator'] =\
            _state_dict(self.subject_calibrator)
        state_dict['object_calibrator'] =\
            _state_dict(self.object_calibrator)
        state_dict['verb_calibrator'] =\
            _state_dict(self.verb_calibrator)
        return state_dict

    def load_state_dict(self, state_dict):
        _load_state_dict(
            self.subject_calibrator,
            state_dict['subject_calibrator']
        )
        _load_state_dict(
            self.object_calibrator,
            state_dict['object_calibrator']
        )
        _load_state_dict(
            self.verb_calibrator,
            state_dict['verb_calibrator']
        )
