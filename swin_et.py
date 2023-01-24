from typing import Any

from torchvision.models import SwinTransformer

# "swin extra-tiny"
def swin_et(**kwargs: Any) -> SwinTransformer:
    """
    Constructs an even smaller version of the swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.
    """
    return SwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2], # Same depth as swin-t
        num_heads=[1, 2, 4, 8], # But narrower than swin-t
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        **kwargs
    )

# swin_t definition from torchvision, for reference:
'''
def swin_t(*, weights: Optional[Swin_T_Weights] = None, progress: bool = True, **kwargs: Any) -> SwinTransformer:
    """
    Constructs a swin_tiny architecture from
    `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows <https://arxiv.org/pdf/2103.14030>`_.

    Args:
        weights (:class:`~torchvision.models.Swin_T_Weights`, optional): The
            pretrained weights to use. See
            :class:`~torchvision.models.Swin_T_Weights` below for
            more details, and possible values. By default, no pre-trained
            weights are used.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.swin_transformer.SwinTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/swin_transformer.py>`_
            for more details about this class.

    .. autoclass:: torchvision.models.Swin_T_Weights
        :members:
    """
    weights = Swin_T_Weights.verify(weights)

    return _swin_transformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        weights=weights,
        progress=progress,
        **kwargs,
    )
'''
