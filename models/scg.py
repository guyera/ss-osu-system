"""
Models

Fred Zhang <frederic.zhang@anu.edu.au>

The Australian National University
Australian Centre for Robotic Vision
"""

from typing import Optional, List, Tuple

import pocket.models as models
from scg_interaction_head import InteractionHead, GraphHead
from torch import nn, Tensor
from torchvision.models.detection import transform
from torchvision.ops import MultiScaleRoIAlign


class HOINetworkTransform(transform.GeneralizedRCNNTransform):
    """
    Transformations for input image and target (box pairs)

    Arguments(Positional):
        min_size(int)
        max_size(int)
        image_mean(list[float] or tuple[float])
        image_std(list[float] or tuple[float])

    Refer to torchvision.models.detection for more details
    """
    def __init__(self, *args):
        super().__init__(*args)

    def resize(self, image, target):
        """
        Override method to resize box pairs
        """
        h, w = image.shape[-2:]
        min_size = float(min(image.shape[-2:]))
        max_size = float(max(image.shape[-2:]))
        scale_factor = min(
            self.min_size[0] / min_size,
            self.max_size / max_size
        )

        image = nn.functional.interpolate(
            image[None], scale_factor=scale_factor,
            mode='bilinear', align_corners=False,
            recompute_scale_factor=True
        )[0]
        if target is None:
            return image, target

        target['boxes_h'] = transform.resize_boxes(target['boxes_h'],
            (h, w), image.shape[-2:])
        target['boxes_o'] = transform.resize_boxes(target['boxes_o'],
            (h, w), image.shape[-2:])

        return image, target

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            loss = results.pop()

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_h, boxes_o = pred['boxes_h'], pred['boxes_o']
            boxes_h = transform.resize_boxes(boxes_h, im_s, o_im_s)
            boxes_o = transform.resize_boxes(boxes_o, im_s, o_im_s)
            pred['boxes_h'], pred['boxes_o'] = boxes_h, boxes_o

        if self.training:
            results.append(loss)

        return results


class GenericHOINetwork(nn.Module):
    """A generic architecture for HOI classification

    Parameters:
    -----------
        backbone: nn.Module
        interaction_head: nn.Module
        transform: nn.Module
        postprocess: bool
            If True, rescale bounding boxes to original image size
    """

    def __init__(self,
                 backbone: nn.Module, interaction_head: nn.Module,
                 transform: nn.Module, postprocess: bool = True
                 ) -> None:
        super().__init__()
        self.backbone = backbone
        self.interaction_head = interaction_head
        self.transform = transform

        self.postprocess = postprocess

    def preprocess(self,
                   images: List[Tensor],
                   detections: List[dict],
                   targets: Optional[List[dict]] = None
                   ) -> Tuple[
        List[Tensor], List[dict],
        List[dict], List[Tuple[int, int]]
    ]:
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)

        for det, o_im_s, im_s in zip(
                detections, original_image_sizes, images.image_sizes
        ):
            boxes = det['boxes']
            boxes = transform.resize_boxes(boxes, o_im_s, im_s)
            det['boxes'] = boxes

        return images, detections, targets, original_image_sizes

    def forward(self,
                images: List[Tensor],
                detections: List[dict],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:
        """
        Parameters:
        -----------
            images: List[Tensor]
            detections: List[dict]
            targets: List[dict]

        Returns:
        --------
            results: List[dict]
        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        images, detections, targets, original_image_sizes = self.preprocess(
            images, detections, targets)

        features = self.backbone(images.tensors)
        results = self.interaction_head(features, detections,
                                        images.image_sizes, targets)

        if self.postprocess and results is not None:
            return self.transform.postprocess(
                results,
                images.image_sizes,
                original_image_sizes
            )
        else:
            return results


class SpatiallyConditionedGraph(GenericHOINetwork):
    def __init__(self,
                 object_to_action: List[list],
                 human_idx: int,
                 # Backbone parameters
                 backbone_name: str = "resnet50",
                 pretrained: bool = True,
                 # Pooler parameters
                 output_size: int = 7,
                 sampling_ratio: int = 2,
                 # Box pair head parameters
                 node_encoding_size: int = 1024,
                 representation_size: int = 1024,
                 num_classes: int = 117,
                 box_score_thresh: float = 0.2,
                 fg_iou_thresh: float = 0.5,
                 num_iterations: int = 2,
                 distributed: bool = False,
                 # Transformation parameters
                 min_size: int = 800, max_size: int = 1333,
                 image_mean: Optional[List[float]] = None,
                 image_std: Optional[List[float]] = None,
                 postprocess: bool = True,
                 # Preprocessing parameters
                 box_nms_thresh: float = 0.5,
                 max_human: int = 15,
                 max_object: int = 15
                 ) -> None:

        detector = models.fasterrcnn_resnet_fpn(backbone_name,
                                                pretrained=pretrained)
        backbone = detector.backbone

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=output_size,
            sampling_ratio=sampling_ratio
        )

        box_pair_head = GraphHead(
            out_channels=backbone.out_channels,
            roi_pool_size=output_size,
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            human_idx=human_idx,
            object_class_to_target_class=object_to_action,
            fg_iou_thresh=fg_iou_thresh,
            num_iter=num_iterations
        )

        box_pair_predictor = nn.Linear(representation_size * 2, num_classes)
        box_pair_suppressor = nn.Linear(representation_size * 2, 1)

        interaction_head = InteractionHead(
            box_roi_pool=box_roi_pool,
            box_pair_head=box_pair_head,
            box_pair_suppressor=box_pair_suppressor,
            box_pair_predictor=box_pair_predictor,
            num_classes=num_classes,
            human_idx=human_idx,
            box_nms_thresh=box_nms_thresh,
            box_score_thresh=box_score_thresh,
            max_human=max_human,
            max_object=max_object,
            distributed=distributed
        )

        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]
        transform = HOINetworkTransform(min_size, max_size,
                                        image_mean, image_std)

        super().__init__(backbone, interaction_head, transform, postprocess)
