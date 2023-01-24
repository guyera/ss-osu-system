from enum import Enum

from torch.nn import Module, Linear
from torchvision.models import\
    swin_t as swin_t_ctor,\
    swin_b as swin_b_ctor,\
    resnet50 as resnet50_ctor

def _swin_t_ctor(weights=None, num_classes=1000):
    if weights is None:
        return swin_t_ctor(num_classes=num_classes)
    elif num_classes is None:
        return swin_t_ctor(weights=weights)
    else:
        model = swin_t_ctor(weights=weights)
        model.head = Linear(
            model.head.weight.shape[1],
            256
        )
        return model

def _resnet50_ctor(weights=None, num_classes=1000):
    if weights is None:
        return resnet50_ctor(weights=weights, num_classes=num_classes)
    elif num_classes is None:
        return resnet50_ctor(weights=weights)
    else:
        model = resnet50_ctor(weights=weights)
        model.fc = Linear(
            model.fc.weight.shape[1],
            256
        )
        return model

class Backbone(Module):
    class Architecture(Enum):
        swin_t = {
            'name': 'swin_t',
            'ctor': _swin_t_ctor,
            'register_feature_hook' :\
                (lambda model, hook : model.features[2]\
                    .register_forward_hook(hook))
        }
        resnet50 = {
            'name': 'resnet50',
            'ctor': _resnet50_ctor,
            'register_feature_hook' :\
                (lambda model, hook : model.layer4.register_forward_hook(hook))
        }

        def __str__(self):
            return self.value['name']

    def __init__(self, architecture, pretrained=True):
        super().__init__()
        self.device = 'cpu'
        self.architecture = architecture
        self._pretrained = pretrained
        self.reset()

    def reset(self):
        weights = 'IMAGENET1K_V1' if self._pretrained else None
        self.model = self.architecture.value['ctor'](
            weights=weights,
            num_classes=256
        ).to(self.device)

    def forward(self, x):
        return self.model(x)

    def to(self, device):
        super().to(device)
        self.device = device
        return self

    def register_feature_hook(self, hook):
        return self.architecture.value['register_feature_hook'](
            self.model,
            hook
        )

    def retrainable_parameters(self):
        return self.parameters()
