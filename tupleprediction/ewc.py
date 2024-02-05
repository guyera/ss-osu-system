from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data
from enum import Enum
from tqdm import tqdm
import os

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)

class LossFnEnum(Enum):
    cross_entropy = 'cross-entropy'
    focal = 'focal'

    def __str__(self):
        return self.value


class EWC(object):
    def __init__(self, backbone, species_classifier, activity_classifier, train_loader, _class_frequencies, _loss_fn, _label_smoothing, device,  precision_matrices_path=None):

        self.backbone = backbone
        self.species_classifier = species_classifier
        self.activity_classifier = activity_classifier
        self.train_loader = train_loader

        self._class_frequencies = _class_frequencies
        self._loss_fn = _loss_fn
        self._label_smoothing = _label_smoothing
        self.device = device

        self.params_backbone = {n: p for n, p in self.backbone.named_parameters() if p.requires_grad}
        self.params_species_classifier = {n: p for n, p in self.species_classifier.named_parameters() if p.requires_grad}
        self.params_activity_classifier = {n: p for n, p in self.activity_classifier.named_parameters() if p.requires_grad}
        self._means = {}

        self.precision_matrices_path = precision_matrices_path
        if self.precision_matrices_path and os.path.isfile(self.precision_matrices_path):
            self._precision_matrices = torch.load(self.precision_matrices_path, map_location=self.device)
        else:
            self._precision_matrices = self._diag_fisher()
            if self.precision_matrices_path:
                self._save_precision_matrices()

        for n, p in deepcopy(self.params_backbone).items():
            self._means[n] = variable(p.data)

        for n, p in deepcopy(self.params_species_classifier).items():
            self._means[n+'species'] = variable(p.data)

        for n, p in deepcopy(self.params_activity_classifier).items():
            self._means[n+'activity'] = variable(p.data)

    def _save_precision_matrices(self):
        torch.save(self._precision_matrices, self.precision_matrices_path)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params_backbone).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)
        
        for n, p in deepcopy(self.params_species_classifier).items():
            p.data.zero_()
            precision_matrices[n+'species'] = variable(p.data)
        
        for n, p in deepcopy(self.params_activity_classifier).items():
            p.data.zero_()
            precision_matrices[n+'activity'] = variable(p.data)

        self.backbone.eval()
        self.activity_classifier.eval()
        self.species_classifier.eval()
        for species_labels, activity_labels, box_images in tqdm(self.train_loader, desc="Calculating Fisher Information on Old Task"):
            species_labels = species_labels.to(self.device)
            activity_labels = activity_labels.to(self.device)

            box_images = box_images.to(self.device)
            box_features = self.backbone(box_images)

            species_preds = self.species_classifier(box_features)
            activity_preds = self.activity_classifier(box_features)

            # Compute losses
            species_weights = None
            activity_weights = None
            if self._class_frequencies is not None:
                species_frequencies, activity_frequencies = self._class_frequencies
                species_frequencies = species_frequencies.to(self.device)
                activity_frequencies = activity_frequencies.to(self.device)

                species_proportions = species_frequencies /\
                    species_frequencies.sum()
                unnormalized_species_weights =\
                    torch.pow(1.0 / species_proportions, 1.0 / 3.0)
                unnormalized_species_weights[species_proportions == 0.0] = 0.0
                proportional_species_sum =\
                    (species_proportions * unnormalized_species_weights).sum()
                species_weights =\
                    unnormalized_species_weights / proportional_species_sum

                activity_proportions = activity_frequencies /\
                    activity_frequencies.sum()
                unnormalized_activity_weights =\
                    torch.pow(1.0 / activity_proportions, 1.0 / 3.0)
                unnormalized_activity_weights[activity_proportions == 0.0] = 0.0
                proportional_activity_sum =\
                    (activity_proportions * unnormalized_activity_weights).sum()
                activity_weights =\
                    unnormalized_activity_weights / proportional_activity_sum
            
            species_loss = torch.nn.functional.cross_entropy(
                species_preds,
                species_labels,
                weight=species_weights,
                label_smoothing=self._label_smoothing
            )
            activity_loss = torch.nn.functional.cross_entropy(
                activity_preds,
                activity_labels,
                weight=activity_weights,
                label_smoothing=self._label_smoothing
            )
            

            non_feedback_loss = species_loss + activity_loss

            self.backbone.zero_grad()
            self.species_classifier.zero_grad()
            self.activity_classifier.zero_grad()

            non_feedback_loss.backward()

            for n, p in self.backbone.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.train_loader)
            
            for n, p in self.species_classifier.named_parameters():
                precision_matrices[n+'species'].data += p.grad.data ** 2 / len(self.train_loader)
            
            for n, p in self.activity_classifier.named_parameters():
                precision_matrices[n+'activity'].data += p.grad.data ** 2 / len(self.train_loader)

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, backbone: nn.Module, species_classifier: nn.Module, activity_classifier: nn.Module,):
        loss = 0
        for n, p in backbone.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
            
        for n, p in species_classifier.named_parameters():
            _loss = self._precision_matrices[n+'species'] * (p - self._means[n+'species']) ** 2

            loss += _loss.sum()
        for n, p in activity_classifier.named_parameters():
            _loss = self._precision_matrices[n+'activity'] * (p - self._means[n+'activity']) ** 2
            loss += _loss.sum()
        
        return loss

