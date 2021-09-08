from typing import Optional, List, Tuple
import numpy as np
import pocket
import torch
import pocket.models as models
from pocket.core import DistributedLearningEngine
from pocket.utils import DetectionAPMeter, HandyTimer, BoxPairAssociation, all_gather
from .scg_interaction_head import InteractionHead, GraphHead
from torch import nn, Tensor
from torchvision.models.detection import transform
from torchvision.models.detection.faster_rcnn import TwoMLPHead
from torchvision.ops import MultiScaleRoIAlign


class CustomFastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.

    Arguments:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes):
        super(CustomFastRCNNPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)

        return scores


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

        target['boxes_s'] = transform.resize_boxes(target['boxes_s'],
                                                   (h, w), image.shape[-2:])
        target['boxes_o'] = transform.resize_boxes(target['boxes_o'],
                                                   (h, w), image.shape[-2:])

        return image, target

    def postprocess(self, results, image_shapes, original_image_sizes):
        if self.training:
            loss = results.pop()

        for pred, im_s, o_im_s in zip(results, image_shapes, original_image_sizes):
            boxes_s, boxes_o = pred['boxes_s'], pred['boxes_o']
            boxes_s = transform.resize_boxes(boxes_s, im_s, o_im_s)
            boxes_o = transform.resize_boxes(boxes_o, im_s, o_im_s)
            pred['boxes_s'], pred['boxes_o'] = boxes_s, boxes_o

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
        # print(f'images: {images}')
        for det, o_im_s, im_s in zip(
                detections, original_image_sizes, images.image_sizes
        ):
            # Now separating object and subject boxes
            # boxes = det['boxes']
            # boxes = transform.resize_boxes(boxes, o_im_s, im_s)
            # det['boxes'] = boxes
            sub_boxes = det['subject_boxes']
            sub_boxes = transform.resize_boxes(sub_boxes, o_im_s, im_s)
            det['subject_boxes'] = sub_boxes
            obj_boxes = det['object_boxes']
            obj_boxes = transform.resize_boxes(obj_boxes, o_im_s, im_s)
            det['object_boxes'] = obj_boxes

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

    def get_box_features(self,
                         images: List[Tensor],
                         detections: List[dict],
                         features=None,
                         ) -> (List[Tensor], List[Tensor]):
        """Returns box features used in the network

        This will return the features on all the boxes but SCG might only use some of these boxes, for verb
        prediction, based on nms and threshold on number of boxes per image. Relevant code and variables to look at:
                - models.scg.scg_interaction_head.InteractionHead.preprocess
                - models.scg.scg_interaction_head.InteractionHead.max_subject
                - models.scg.scg_interaction_head.InteractionHead.max_object

        # TODO: Add format
        """
        images, detections, _, _ = self.preprocess(
            images, detections)
        if features is not None:
            features = self.backbone(images.tensors)
        subject_box_coords = [detection['subject_boxes'] for detection in detections]
        object_box_coords = [detection['object_boxes'] for detection in detections]
        subject_box_features = self.interaction_head.box_roi_pool(features, subject_box_coords, images.image_sizes)
        object_box_features = self.interaction_head.box_roi_pool(features, object_box_coords, images.image_sizes)
        # box_features = self.interaction_head.box_head(box_features)
        return subject_box_features, object_box_features

    def get_verb_features(self,
                          images: List[Tensor],
                          detections: List[dict],
                          ) -> List[Tensor]:
        """"""
        features = self.backbone(images.tensors)
        subject_box_features, object_box_features = self.get_box_features(images, detections, features)
        subject_box_coords = [detection['subject_boxes'] for detection in detections]
        object_box_coords = [detection['object_boxes'] for detection in detections]
        subject_pred_detections = self.interaction_head.classify_boxes(subject_box_features, subject_box_coords)
        object_pred_detections = self.interaction_head.classify_boxes(object_box_features, object_box_coords)
        detections = self.interaction_head.preprocess(subject_pred_detections, object_pred_detections)
        subject_box_coords = list()
        subject_box_all_scores = list()
        object_box_coords = list()
        object_box_all_scores = list()
        for detection in detections:
            subject_box_coords.append(detection['boxes'][:detection['num_subjects']])
            subject_box_all_scores.append(detection['box_all_scores'][:detection['num_subjects']])
            object_box_coords.append(detection['boxes'][detection['num_subjects']:])
            object_box_all_scores.append(detection['box_all_scores'][detection['num_subjects']:])
        subject_box_features = self.interaction_head.box_roi_pool(features, subject_box_coords, images.image_sizes)
        object_box_features = self.interaction_head.box_roi_pool(features, object_box_coords, images.image_sizes)
        result = self.interaction_head.box_pair_head(features, images.image_sizes,
                                                     subject_box_features, subject_box_coords,
                                                     subject_box_all_scores, object_box_features, object_box_coords,
                                                     object_box_all_scores)
        return result[0]


class SpatiallyConditionedGraph(GenericHOINetwork):
    def __init__(self,
                 object_to_action: List[list],
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
                 num_obj_classes: int = 80,
                 num_subject_classes: int = 80,
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
                 max_subject: int = 15,
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
        representation_size = 1024
        box_head = TwoMLPHead(
            backbone.out_channels * 7 ** 2,
            representation_size)

        box_pair_head = GraphHead(
            out_channels=backbone.out_channels,
            roi_pool_size=output_size,
            node_encoding_size=node_encoding_size,
            representation_size=representation_size,
            num_cls=num_classes,
            object_class_to_target_class=object_to_action,
            fg_iou_thresh=fg_iou_thresh,
            num_iter=num_iterations
        )

        box_verb_predictor = nn.Linear(representation_size * 2, num_classes)
        box_verb_suppressor = nn.Linear(representation_size * 2, 1)

        custom_box_classifier_dict = nn.ModuleDict({"object": CustomFastRCNNPredictor(1024, num_obj_classes),
                                                    "subject": CustomFastRCNNPredictor(1024, num_subject_classes)})

        interaction_head = InteractionHead(
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_pair_head=box_pair_head,
            box_verb_suppressor=box_verb_suppressor,
            box_verb_predictor=box_verb_predictor,
            custom_box_classifier=custom_box_classifier_dict,
            num_classes=num_classes,
            box_nms_thresh=box_nms_thresh,
            box_score_thresh=box_score_thresh,
            max_subject=max_subject,
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


class CustomisedDLE(DistributedLearningEngine):
    def __init__(self, net, train_loader, val_loader, num_classes=117, **kwargs):
        super().__init__(net, None, train_loader, **kwargs)
        self.val_loader = val_loader
        self.num_classes = num_classes

    def _on_start(self):
        self.meter = DetectionAPMeter(self.num_classes, algorithm='11P')
        self.hoi_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)
        self.intr_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)
        self.obj_cls_loss = pocket.utils.SyncedNumericalMeter(maxlen=self._print_interval)

    def _on_each_iteration(self):
        self._state.optimizer.zero_grad()
        output = self._state.net(
            *self._state.inputs, targets=self._state.targets)
        loss_dict = output.pop()
        if loss_dict['hoi_loss'].isnan():
            raise ValueError(f"The HOI loss is NaN for rank {self._rank}")

        self._state.loss = sum(loss for loss in loss_dict.values())
        self._state.loss.backward()
        self._state.optimizer.step()

        self.hoi_loss.append(loss_dict['hoi_loss'])
        self.intr_loss.append(loss_dict['interactiveness_loss'])
        self.obj_cls_loss.append(loss_dict['box_classification_loss'])

        self._synchronise_and_log_results(output, self.meter)

    def _on_end_epoch(self):
        timer = HandyTimer(maxlen=2)
        # Compute training mAP
        if self._rank == 0:
            with timer:
                ap_train = self.meter.eval()
        # Run validation and compute mAP
        with timer:
            ap_val = self.validate()
        # Print performance and time
        if self._rank == 0:
            print("Epoch: {} | training mAP: {:.4f}, evaluation time: {:.2f}s |"
                  "validation mAP: {:.4f}, total time: {:.2f}s\n".format(
                self._state.epoch, ap_train.mean().item(), timer[0],
                ap_val.mean().item(), timer[1]
            ))
            self.meter.reset()
        super()._on_end_epoch()

    def _print_statistics(self):
        super()._print_statistics()
        hoi_loss = self.hoi_loss.mean()
        intr_loss = self.intr_loss.mean()
        obj_cls_loss = self.obj_cls_loss.mean()
        if self._rank == 0:
            print(f"=> HOI classification loss: {hoi_loss:.4f},",
                  f"interactiveness loss: {intr_loss:.4f},"
                  f"object classification loss: {obj_cls_loss:.4f}")
        self.hoi_loss.reset()
        self.intr_loss.reset()
        self.obj_cls_loss.reset()

    def _synchronise_and_log_results(self, output, meter):
        scores = []
        pred = []
        labels = []
        # Collate results within the batch
        for result in output:
            scores.append(result['scores'].detach().cpu().numpy())
            pred.append(result['prediction'].cpu().float().numpy())
            labels.append(result["labels"].cpu().numpy())
        # Sync across subprocesses
        all_results = np.stack([
            np.concatenate(scores),
            np.concatenate(pred),
            np.concatenate(labels)
        ])
        all_results_sync = all_gather(all_results)
        # Collate and log results in master process
        if self._rank == 0:
            scores, pred, labels = torch.from_numpy(
                np.concatenate(all_results_sync, axis=1)
            ).unbind(0)
            meter.append(scores, pred, labels)

    @torch.no_grad()
    def validate(self):
        meter = DetectionAPMeter(self.num_classes, algorithm='11P')

        self._state.net.eval()
        for batch in self.val_loader:
            inputs = pocket.ops.relocate_to_cuda(batch)
            results = self._state.net(*inputs)

            self._synchronise_and_log_results(results, meter)

        # Evaluate mAP in master process
        if self._rank == 0:
            return meter.eval()
        else:
            return None
