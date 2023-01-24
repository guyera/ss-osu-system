from swin_et import swin_et

class SideTuningBackbone(Module):
    def __init__(self, backbone):
        super().__init__()
        self._backbone = backbone
        self.device = backbone.device
        self.reset()

    def reset(self):
        # This does NOT reset the underlying backbone, since the backbone should
        # remain frozen during retraining / accommodation finetuning.
        # It only resets the swin_et side network
        self._model = swin_et(num_classes=256).to(self.device)

    def forward(self, x):
        y1 = self._backbone(x)
        y2 = self._model(x)
        x = torch.cat((y1, y2), dim=-1)
        return x

    def to(self, device):
        super().to(device)
        self._backbone = self._backbone.to(device)
        self._model = self._model.to(device)
        self.device = device
        return self

    def register_feature_hook(self, hook):
        return self._backbone.register_feature_hook(hook)

    def retrainable_parameters(self):
        # Backbone parameters are not retrainable; only return swin_et's
        # parameters
        return self._model.parameters()
