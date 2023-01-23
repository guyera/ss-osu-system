import torch

from sklearn.neighbors import KernelDensity

from backbone import Backbone
from scoring._scorer import Scorer

class ActivationStatisticalModel(Scorer):
    _bandwidths = {
        Backbone.Architecture.swin_t: 50,
        Backbone.Architecture.resnet50: 5
    }
    def __init__(self, backbone_architecture):
        self._bandwidth = self._bandwidths[backbone_architecture]
        self._kde = KernelDensity(kernel='gaussian', bandwidth=self._bandwidth)
        self._device = None
        self._v = None

    def forward_hook(self, module, inputs, outputs):
        self._features = outputs

    def compute_features(self, backbone, batch):
        handle = backbone.register_feature_hook(self.forward_hook)
        backbone(batch)
        handle.remove()
        return self._features.view(self._features.shape[0], -1)

    def pca_reduce(self, v, features, k):
        return torch.matmul(features, v[:, :k])

    def fit(self, features):
        # Fit PCA
        _, _, v = torch.svd(features)
        self._v = v

        # PCA-project features
        projected_features = self.pca_reduce(v, features, 64)

        # Fit self._kde to pca-projected features
        self._kde.fit(projected_features.cpu().numpy())

    def score(
            self,
            species_logits,
            activity_logits,
            whole_image_features,
            box_counts):
        # Compute and return negative log likelihood under self._kde
        projected_features = self.pca_reduce(self._v, whole_image_features, 64)
        np_scores = -self._kde.score_samples(projected_features.cpu().numpy())
        scores = torch.from_numpy(np_scores)\
            .to(self._device)[:, None]\
            .to(torch.float)
        return scores

    def n_scores(self):
        return 1

    def reset(self):
        self._v = None
        self._kde = KernelDensity(kernel='gaussian', bandwidth=self._bandwidth)

    def to(self, device):
        self._device = device
        if self._v is not None:
            self._v = self._v.to(device)
        return self

    def state_dict(self):
        sd = {}
        if self._v is not None:
            sd['v'] = self._v.to('cpu')
        else:
            sd['v'] = None
        sd['kde'] = self._kde
        return sd

    def load_state_dict(self, sd):
        self._v = sd['v']
        self._kde = sd['kde']
        if self._device is not None and self._v is not None:
            self._v = self._v.to(self._device)
