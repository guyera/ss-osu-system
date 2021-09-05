from collections import OrderedDict
from typing import Optional, List, Tuple
import pocket
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.ops.boxes as box_ops
from utils import compute_spatial_encodings, binary_focal_loss
from pocket.ops import Flatten
from torch import nn, Tensor
from torch.nn import Module
from typing import Dict, Union, Any


class InteractionHead(Module):
    """Interaction head that constructs and classifies box pairs

    Parameters:
    -----------
    box_roi_pool: Module
        Module that performs RoI pooling or its variants
    box_pair_head: Module
        Module that constructs and computes box pair features
    box_pair_suppressor: Module
        Module that computes unary weights for each box pair
    box_pair_predictor: Module
        Module that classifies box pairs
    num_classes: int
        Number of target classes
    box_nms_thresh: float, default: 0.5
        Threshold used for non-maximum suppression
    box_score_thresh: float, default: 0.2
        Threshold used to filter out low-quality boxes
    max_subject: int, default: 15
        Number of subject detections to keep in each image
    max_object: int, default: 15
        Number of object (excluding subjects) detections to keep in each image
    distributed: bool, default: False
        Whether the model is trained under distributed data parallel. If True,
        the number of positive logits will be averaged across all subprocesses
    """

    def __init__(self,
                 # Network components
                 box_roi_pool: Module,
                 box_head: Module,
                 box_pair_head: Module,
                 box_pair_suppressor: Module,
                 box_pair_predictor: Module,
                 custom_box_classifier: Module,
                 # Dataset properties
                 num_classes: int,
                 # Hyperparameters
                 box_nms_thresh: float = 0.5,
                 box_score_thresh: float = 0.2,
                 max_subject: int = 15,
                 max_object: int = 15,
                 # Misc
                 distributed: bool = False
                 ) -> None:
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_head = box_head
        self.box_pair_head = box_pair_head
        self.box_pair_suppressor = box_pair_suppressor
        self.box_pair_predictor = box_pair_predictor
        self.custom_box_classifier = custom_box_classifier

        self.num_classes = num_classes

        self.box_nms_thresh = box_nms_thresh
        self.box_score_thresh = box_score_thresh
        self.max_subject = max_subject
        self.max_object = max_object

        self.distributed = distributed

    def preprocess(self,
                   subject_detections: List[dict],
                   object_detections: List[dict],
                   targets: List[dict],
                   append_gt: Optional[bool] = None
                   ) -> List[Dict[str, Union[int, Any]]]:

        results = []
        for b_idx, _ in enumerate(subject_detections):
            # Append ground truth during training
            if append_gt is None:
                append_gt = self.training
            if append_gt:
                target = targets[b_idx]
                n_s = target["boxes_s"].shape[0]
                n_o = target["boxes_o"].shape[0]
                boxes = torch.cat([target["boxes_s"], target["boxes_o"], subject_detections[b_idx]['boxes'],
                                   object_detections[b_idx]['boxes']])
                scores = torch.cat([torch.ones(n_s, device=subject_detections[b_idx]['scores'].device),
                                    torch.ones(n_o, device=subject_detections[b_idx]['scores'].device),
                                    subject_detections[b_idx]['scores'],
                                    object_detections[b_idx]['scores']])
                labels = torch.cat([
                    target["subject"],
                    target["object"],
                    subject_detections[b_idx]['labels'],
                    object_detections[b_idx]['labels']
                ])
                sub_idx = list(range(n_s)) + list(range(n_s + n_o, n_s + n_o + len(subject_detections[b_idx])))
                obj_idx = list(range(n_s, n_s + n_o)) + list(range(n_s + n_o + len(subject_detections[b_idx]),
                                                                   len(scores)))
            else:
                boxes = torch.cat([subject_detections[b_idx]['boxes'],
                                   object_detections[b_idx]['boxes']])
                scores = torch.cat([subject_detections[b_idx]['scores'],
                                   object_detections[b_idx]['scores']])
                labels = torch.cat([subject_detections[b_idx]['labels'],
                                   object_detections[b_idx]['labels']])
                sub_idx = list(range(len(subject_detections[b_idx])))
                obj_idx = list(range(len(subject_detections[b_idx]), len(scores)))

            # Remove low scoring examples
            sub_idx = pocket.ops.relocate_to_cuda(torch.IntTensor(sub_idx))
            obj_idx = pocket.ops.relocate_to_cuda(torch.IntTensor(obj_idx))
            active_idx = torch.nonzero(
                scores >= self.box_score_thresh
            ).squeeze(1)
            # Class-wise non-maximum suppression
            keep_idx = box_ops.batched_nms(
                boxes[active_idx],
                scores[active_idx],
                labels[active_idx],
                self.box_nms_thresh
            )
            active_idx = active_idx[keep_idx]
            # Sort detections by scores
            sorted_idx = torch.argsort(scores[active_idx], descending=True)
            active_idx = active_idx[sorted_idx]
            # Keep a fixed number of detections
            s_idx = torch.nonzero((active_idx[..., None] == sub_idx).any(-1)).squeeze(1)
            o_idx = torch.nonzero((active_idx[..., None] == obj_idx).any(-1)).squeeze(1)
            if len(s_idx) > self.max_subject:
                s_idx = s_idx[:self.max_subject]
            if len(o_idx) > self.max_object:
                o_idx = o_idx[:self.max_object]
            # Permute subjects to the top
            keep_idx = torch.cat([s_idx, o_idx])
            active_idx = active_idx[keep_idx]

            results.append(dict(
                boxes=boxes[active_idx].view(-1, 4),
                labels=labels[active_idx].view(-1),
                scores=scores[active_idx].view(-1),
                num_subjects=len(s_idx),
            ))

        return results

    def compute_interaction_classification_loss(self, results: List[dict]) -> Tensor:
        scores = []
        labels = []
        for result in results:
            scores.append(result['scores'])
            labels.append(result['labels'])

        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            torch.cat(scores), labels, reduction='sum', gamma=0.2
        )
        return loss / n_p

    def compute_interactiveness_loss(self, results: List[dict]) -> Tensor:
        weights = []
        labels = []
        for result in results:
            weights.append(result['weights'])
            labels.append(result['unary_labels'])

        weights = torch.cat(weights)
        labels = torch.cat(labels)
        n_p = len(torch.nonzero(labels))
        if self.distributed:
            world_size = dist.get_world_size()
            n_p = torch.as_tensor([n_p], device='cuda')
            dist.barrier()
            dist.all_reduce(n_p)
            n_p = (n_p / world_size).item()
        loss = binary_focal_loss(
            weights, labels, reduction='sum', gamma=2.0
        )
        return loss / n_p

    def compute_box_classification_loss(self, labels: List[Tensor], box_logits: List[Tensor]) -> Tensor:
        labels = torch.cat(labels, dim=0)
        classification_loss = F.cross_entropy(torch.cat(box_logits), labels)
        return classification_loss

    def postprocess(self,
                    logits_p: Tensor,
                    logits_s: Tensor,
                    prior: List[Tensor],
                    boxes_s: List[Tensor],
                    boxes_o: List[Tensor],
                    object_class: List[Tensor],
                    labels: List[Tensor],
                    ) -> List[dict]:
        """
        Parameters:
        -----------
        logits_p: Tensor
            (N, K) Classification logits on each action for all box pairs
        logits_s: Tensor
            (N, 1) Logits for unary weights
        prior: List[Tensor]
            Prior scores organised by images. Each tensor has shape (2, M, K).
            M could be different for different images
        boxes_s: List[Tensor]
            Subject bounding box coordinates organised by images (M, 4)
        boxes_o: List[Tensor]
            Object bounding box coordinates organised by images (M, 4)
        object_classes: List[Tensor]
            Object indices for each pair organised by images (M,)
        labels: List[Tensor]
            Binary labels on each action organised by images (M, K)

        Returns:
        --------
        results: List[dict]
            Results organised by images, with keys as below
            `boxes_s`: Tensor[M, 4]
            `boxes_o`: Tensor[M, 4]
            `index`: Tensor[L]
                Expanded indices of box pairs for each predicted action
            `prediction`: Tensor[L]
                Expanded indices of predicted actions
            `scores`: Tensor[L]
                Scores for each predicted action
            `object`: Tensor[M]
                Object indices for each pair
            `prior`: Tensor[2, L]
                Prior scores for expanded pairs
            `weights`: Tensor[M]
                Unary weights for each box pair
            `labels`: Tensor[L], optional
                Binary labels on each action
            `unary_labels`: Tensor[M], optional
                Labels for the unary weights
        """
        num_boxes = [len(b) for b in boxes_s]

        weights = torch.sigmoid(logits_s).squeeze(1)
        scores = torch.sigmoid(logits_p)
        weights = weights.split(num_boxes)
        scores = scores.split(num_boxes)
        if len(labels) == 0:
            labels = [None for _ in range(len(num_boxes))]

        results = []
        for w, s, p, b_s, b_o, o, l in zip(
                weights, scores, prior, boxes_s, boxes_o, object_class, labels
        ):
            # Keep valid classes
            x, y = torch.nonzero(p[0]).unbind(1)
            result_dict = dict(
                boxes_s=b_s, boxes_o=b_o,
                index=x, prediction=y,
                scores=s[x, y] * p[:, x, y].prod(dim=0) * w[x].detach(),
                object=o, prior=p[:, x, y], weights=w,
            )
            # If binary labels are provided
            if l is not None:
                result_dict['labels'] = l[x, y]
                result_dict['unary_labels'] = l.sum(dim=1).clamp(max=1)

            results.append(result_dict)

        return results
    
    def classify_boxes(self, box_features, box_coords):
        box_features = self.box_head(box_features)
        box_logits = self.custom_box_classifier(box_features)
        box_scores = F.softmax(box_logits, -1).split([len(elem) for elem in box_coords], 0)
        box_logits = box_logits.split([len(elem) for elem in box_coords], 0)
        pred_detections = list()
        for i, boxes in enumerate(box_coords):
            scores, labels = torch.max(box_scores[i], dim=1)
            detection = {
                'boxes': boxes,
                'labels': labels,
                # This will be used as GT labels for training classification head
                'scores': scores,
                'box_logits': box_logits[i],
            }
            pred_detections.append(detection)
        return pred_detections

    def forward(self,
                features: OrderedDict,
                detections: List[dict],
                image_shapes: List[Tuple[int, int]],
                targets: Optional[List[dict]] = None
                ) -> List[dict]:
        """
        Parameters:
        -----------
        features: OrderedDict
            Feature maps returned by FPN
        detections: List[dict]
            Object detections with the following keys
            `boxes`: Tensor[N, 4]
            `labels`: Tensor[N]
            `scores`: Tensor[N]
        image_shapes: List[Tuple[int, int]]
            Image shapes, heights followed by widths
        targets: List[dict], optional
            Interaction targets with the following keys
            `boxes_s`: Tensor[G, 4]
            `boxes_o`: Tensor[G, 4]
            `object`: Tensor[G]
                Object class indices for each pair
            `labels`: Tensor[G]
                Target class indices for each pair

        Returns:
        --------
        results: List[dict]
            Results organised by images. During training the loss dict is appended to the
            end of the list, resulting in the length being larger than the number of images
            by one. For the result dict of each image, refer to `postprocess` for documentation.
            The loss dict has two keys
            `hoi_loss`: Tensor
                Loss for HOI classification
            `interactiveness_loss`: Tensor
                Loss incurred on learned unary weights
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        # Custom box classification head
        subject_box_coords = list()
        subject_orig_labels = list()
        object_box_coords = list()
        object_orig_labels = list()
        for detection in detections:
            subject_box_coords.append(detection['subject_boxes'])
            subject_orig_labels.append(detection['subject_labels'])
            object_box_coords.append(detection['object_boxes'])
            object_orig_labels.append(detection['object_labels'])

        subject_box_features = self.box_roi_pool(features, subject_box_coords, image_shapes)
        object_box_features = self.box_roi_pool(features, object_box_coords, image_shapes)
        # print(f'subject box length: {}')
        subject_pred_detections = self.classify_boxes(subject_box_features, subject_box_coords)
        object_pred_detections = self.classify_boxes(object_box_features, object_box_coords)

        # Computing classification head loss
        if self.training:
            subject_box_logits = [detection['box_logits'] for detection in subject_pred_detections]
            object_box_logits = [detection['orig_labels'] for detection in object_pred_detections]
            subject_cls_loss = self.compute_box_classification_loss(subject_orig_labels, subject_box_logits)
            object_cls_loss = self.compute_box_classification_loss(object_orig_labels, object_box_logits)
            box_cls_loss = subject_cls_loss + object_cls_loss
        # Original code resumes
        detections = self.preprocess(subject_pred_detections, object_pred_detections, targets)

        subject_box_coords = list()
        subject_box_labels = list()
        subject_box_scores = list()
        object_box_coords = list()
        object_box_labels = list()
        object_box_scores = list()
        for detection in detections:
            subject_box_coords.append(detection['boxes'][:detection['num_subjects']]) 
            subject_box_labels.append(detection['labels'][:detection['num_subjects']])
            subject_box_scores.append(detection['scores'][:detection['num_subjects']])
            object_box_coords.append(detection['boxes'][detection['num_subjects']:])
            object_box_labels.append(detection['labels'][detection['num_subjects']:])
            object_box_scores.append(detection['scores'][detection['num_subjects']:])

        subject_box_features = self.box_roi_pool(features, subject_box_coords, image_shapes)
        object_box_features = self.box_roi_pool(features, object_box_coords, image_shapes)

        box_pair_features, boxes_s, boxes_o, object_class, \
        box_pair_labels, box_pair_prior = self.box_pair_head(
            features, image_shapes, subject_box_features, subject_box_coords, subject_box_labels, subject_box_scores,
            object_box_features, object_box_coords, object_box_labels, object_box_scores, targets
        )

        box_pair_features = torch.cat(box_pair_features)
        logits_p = self.box_pair_predictor(box_pair_features)
        logits_s = self.box_pair_suppressor(box_pair_features)

        results = self.postprocess(
            logits_p, logits_s, box_pair_prior,
            boxes_s, boxes_o,
            object_class, box_pair_labels
        )

        if self.training:
            loss_dict = dict(
                hoi_loss=self.compute_interaction_classification_loss(results),
                interactiveness_loss=self.compute_interactiveness_loss(results),
                box_classification_loss=box_cls_loss,
            )
            results.append(loss_dict)

        return results


class MultiBranchFusion(Module):
    """
    Multi-branch fusion module

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    cardinality: int
        The number of homogeneous branches
    """

    def __init__(self,
                 appearance_size: int, spatial_size: int,
                 representation_size: int, cardinality: int
                 ) -> None:
        super().__init__()
        self.cardinality = cardinality

        sub_repr_size = int(representation_size / cardinality)
        assert sub_repr_size * cardinality == representation_size, \
            "The given representation size should be divisible by cardinality"

        self.fc_1 = nn.ModuleList([
            nn.Linear(appearance_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_2 = nn.ModuleList([
            nn.Linear(spatial_size, sub_repr_size)
            for _ in range(cardinality)
        ])
        self.fc_3 = nn.ModuleList([
            nn.Linear(sub_repr_size, representation_size)
            for _ in range(cardinality)
        ])

    def forward(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        return F.relu(torch.stack([
            fc_3(F.relu(fc_1(appearance) * fc_2(spatial)))
            for fc_1, fc_2, fc_3
            in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0))


class MessageMBF(MultiBranchFusion):
    """
    MBF for the computation of anisotropic messages

    Parameters:
    -----------
    appearance_size: int
        Size of the appearance features
    spatial_size: int
        Size of the spatial features
    representation_size: int
        Size of the intermediate representations
    node_type: str
        Nature of the sending node. Choose between `subject` amd `object`
    cardinality: int
        The number of homogeneous branches
    """

    def __init__(self,
                 appearance_size: int,
                 spatial_size: int,
                 representation_size: int,
                 node_type: str,
                 cardinality: int
                 ) -> None:
        super().__init__(appearance_size, spatial_size, representation_size, cardinality)

        if node_type == 'subject':
            self._forward_method = self._forward_subject_nodes
        elif node_type == 'object':
            self._forward_method = self._forward_object_nodes
        else:
            raise ValueError("Unknown node type \"{}\"".format(node_type))

    def _forward_subject_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_s, n = spatial.shape[:2]
        assert len(appearance) == n_s, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n, 1, 1)
                * fc_2(spatial).permute([1, 0, 2])
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def _forward_object_nodes(self, appearance: Tensor, spatial: Tensor) -> Tensor:
        n_s, n = spatial.shape[:2]
        assert len(appearance) == n, "Incorrect size of dim0 for appearance features"
        return torch.stack([
            fc_3(F.relu(
                fc_1(appearance).repeat(n_s, 1, 1)
                * fc_2(spatial)
            )) for fc_1, fc_2, fc_3 in zip(self.fc_1, self.fc_2, self.fc_3)
        ]).sum(dim=0)

    def forward(self, *args) -> Tensor:
        return self._forward_method(*args)


class GraphHead(Module):
    """
    Graphical model head

    Parameters:
    -----------
    output_channels: int
        Number of output channels of the backbone
    roi_pool_size: int
        Spatial resolution of the pooled output
    node_encoding_size: int
        Size of the node embeddings
    num_cls: int
        Number of target classes
    object_class_to_target_class: List[list]
        The mapping (potentially one-to-many) from objects to target classes
    fg_iou_thresh: float, default: 0.5
        The IoU threshold to identify a positive example
    num_iter: int, default 2
        Number of iterations of the message passing process
    """

    def __init__(self,
                 out_channels: int,
                 roi_pool_size: int,
                 node_encoding_size: int,
                 representation_size: int,
                 num_cls: int,
                 object_class_to_target_class: List[list],
                 fg_iou_thresh: float = 0.5,
                 num_iter: int = 2
                 ) -> None:

        super().__init__()
        self.out_channels = out_channels
        self.roi_pool_size = roi_pool_size
        self.node_encoding_size = node_encoding_size
        self.representation_size = representation_size

        self.num_cls = num_cls
        self.object_class_to_target_class = object_class_to_target_class

        self.fg_iou_thresh = fg_iou_thresh
        self.num_iter = num_iter

        # Box head to map RoI features to low dimensional
        self.box_head = nn.Sequential(
            Flatten(start_dim=1),
            nn.Linear(out_channels * roi_pool_size ** 2, node_encoding_size),
            nn.ReLU(),
            nn.Linear(node_encoding_size, node_encoding_size),
            nn.ReLU()
        )

        # Compute adjacency matrix
        self.adjacency = nn.Linear(representation_size, 1)

        # Compute messages
        self.sub_to_obj = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='subject',
            cardinality=16
        )
        self.obj_to_sub = MessageMBF(
            node_encoding_size, 1024,
            representation_size, node_type='object',
            cardinality=16
        )

        self.norm_s = nn.LayerNorm(node_encoding_size)
        self.norm_o = nn.LayerNorm(node_encoding_size)

        # Map spatial encodings to the same dimension as appearance features
        self.spatial_head = nn.Sequential(
            nn.Linear(36, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        # Spatial attention head
        self.attention_head = MultiBranchFusion(
            node_encoding_size * 2,
            1024, representation_size,
            cardinality=16
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # Attention head for global features
        self.attention_head_g = MultiBranchFusion(
            256, 1024,
            representation_size, cardinality=16
        )

    def associate_with_ground_truth(self,
                                    boxes_s: Tensor,
                                    boxes_o: Tensor,
                                    targets: List[dict]
                                    ) -> Tensor:
        n = boxes_s.shape[0]
        labels = torch.zeros(n, self.num_cls, device=boxes_s.device)

        x, y = torch.nonzero(torch.min(
            box_ops.box_iou(boxes_s, targets["boxes_s"]),
            box_ops.box_iou(boxes_o, targets["boxes_o"])
        ) >= self.fg_iou_thresh).unbind(1)

        labels[x, targets["labels"][y]] = 1

        return labels

    def compute_prior_scores(self,
                             x: Tensor, y: Tensor,
                             scores: Tensor,
                             object_class: Tensor
                             ) -> Tensor:
        """
        Parameters:
        -----------
            x: Tensor[M]
                Indices of subject boxes (paired)
            y: Tensor[M]
                Indices of object boxes (paired)
            scores: Tensor[N]
                Object detection scores (before pairing)
            object_class: Tensor[N]
                Object class indices (before pairing)
        """
        prior_s = torch.zeros(len(x), self.num_cls, device=scores.device)
        prior_o = torch.zeros_like(prior_s)

        # Raise the power of object detection scores during inference
        p = 1.0 if self.training else 2.8
        s_s = scores[x].pow(p)
        s_o = scores[y].pow(p)

        # Map object class index to target class index
        # Object class index to target class index is a one-to-many mapping
        target_cls_idx = [self.object_class_to_target_class[obj.item()]
                          for obj in object_class[y]]
        # Duplicate box pair indices for each target class
        pair_idx = [i for i, tar in enumerate(target_cls_idx) for _ in tar]
        # Flatten mapped target indices
        flat_target_idx = [t for tar in target_cls_idx for t in tar]

        prior_s[pair_idx, flat_target_idx] = s_s[pair_idx]
        prior_o[pair_idx, flat_target_idx] = s_o[pair_idx]

        return torch.stack([prior_s, prior_o])

    def forward(self,
                features: OrderedDict, image_shapes: List[Tuple[int, int]],
                subject_box_features: Tensor, subject_box_coords: List[Tensor],
                subject_box_labels: List[Tensor], subject_box_scores: List[Tensor],
                object_box_features: Tensor, object_box_coords: List[Tensor],
                object_box_labels: List[Tensor], object_box_scores: List[Tensor],
                targets: Optional[List[dict]] = None
                ) -> Tuple[
        List[Tensor], List[Tensor], List[Tensor],
        List[Tensor], List[Tensor], List[Tensor]
    ]:
        """
        Parameters:
        -----------
            features: OrderedDict
                Feature maps returned by FPN
            image_shapes: List[Tuple[int, int]]
                Image shapes, heights followed by widths
            subject_box_features: Tensor
                (N, C, P, P) Pooled box features for subjects
            subject_box_coords: List[Tensor]
                Bounding box coordinates organised by images for subjects
            subject_box_labels: List[Tensor]
                Bounding box object types organised by images for subjects
            subject_box_scores: List[Tensor]
                Bounding box scores organised by images for subjects
            object_box_features: Tensor
                (N, C, P, P) Pooled box features for objects
            object_box_coords: List[Tensor]
                Bounding box coordinates organised by images for objects
            object_box_labels: List[Tensor]
                Bounding box object types organised by images for objects
            object_box_scores: List[Tensor]
                Bounding box scores organised by images for objects
            targets: List[dict]
                Interaction targets with the following keys
                `boxes_s`: Tensor[G, 4]
                `boxes_o`: Tensor[G, 4]
                `labels`: Tensor[G]

        Returns:
        --------
            all_box_pair_features: List[Tensor]
            all_boxes_s: List[Tensor]
            all_boxes_o: List[Tensor]
            all_object_class: List[Tensor]
            all_labels: List[Tensor]
            all_prior: List[Tensor]
        """
        if self.training:
            assert targets is not None, "Targets should be passed during training"

        global_features = self.avg_pool(features['3']).flatten(start_dim=1)
        subject_box_features = self.box_head(subject_box_features)
        object_box_features = self.box_head(object_box_features)

        subject_counter = 0
        object_counter = 0
        all_boxes_s = []
        all_boxes_o = []
        all_object_class = []
        all_labels = []
        all_prior = []
        all_box_pair_features = []
        for b_idx, _ in enumerate(subject_box_coords):
            n_s = len(subject_box_coords[b_idx])
            n_o = len(object_box_coords[b_idx])
            device = subject_box_features.device

            # Skip image when there are no detected subject or object instances
            # and when there is only one detected instance
            if n_s == 0 or (n_o + n_s) <= 1:
                all_box_pair_features.append(torch.zeros(
                    0, 2 * self.representation_size,
                    device=device)
                )
                all_boxes_s.append(torch.zeros(0, 4, device=device))
                all_boxes_o.append(torch.zeros(0, 4, device=device))
                all_object_class.append(torch.zeros(0, device=device, dtype=torch.int64))
                all_prior.append(torch.zeros(2, 0, self.num_cls, device=device))
                all_labels.append(torch.zeros(0, self.num_cls, device=device))
                continue
            # TODO: Reimplement this check
            # if not torch.all(labels[:num_subjects] == self.human_idx):
            #     raise ValueError("Subject detections are not permuted to the top")

            print(f'n_S: {n_s}')
            print(f'n_o: {n_o}')
            print(f'subject_box_features: {subject_box_features.size()}')
            print(f'object_box_features: {object_box_features.size()}')
            node_encodings = subject_box_features[subject_counter: subject_counter + n_s] + \
                             object_box_features[object_counter: object_counter + n_o]
            # Duplicate subject nodes
            s_node_encodings = subject_box_features[subject_counter: subject_counter + n_s]
            # Get the pairwise index between every subject and object instance
            x, y = torch.meshgrid(
                torch.arange(n_s, device=device),
                torch.arange(n_s + n_o, device=device)
            )
            print(f'x: {x}')
            print(f'y: {y}')
            # Remove pairs consisting of the same subject instance
            x_keep, y_keep = torch.nonzero(x != y).unbind(1)
            if len(x_keep) == 0:
                # Should never happen, just to be safe
                raise ValueError("There are no valid subject-object pairs")
            # subject nodes have been duplicated and will be treated independently
            # of the subjects included amongst object nodes
            x = x.flatten()
            y = y.flatten()

            # Compute spatial features
            box_pair_spatial = compute_spatial_encodings(
                [coords[x]], [coords[y]], [image_shapes[b_idx]]
            )
            box_pair_spatial = self.spatial_head(box_pair_spatial)
            # Reshape the spatial features
            box_pair_spatial_reshaped = box_pair_spatial.reshape(n_s, n_s + n_o, -1)

            adjacency_matrix = torch.ones(n_s, n_s + n_o, device=device)
            for _ in range(self.num_iter):
                # Compute weights of each edge
                weights = self.attention_head(
                    torch.cat([
                        s_node_encodings[x],
                        node_encodings[y]
                    ], 1),
                    box_pair_spatial
                )
                adjacency_matrix = self.adjacency(weights).reshape(n_s, n_s + n_o)

                # Update subject nodes
                messages_to_s = F.relu(torch.sum(
                    adjacency_matrix.softmax(dim=1)[..., None] *
                    self.obj_to_sub(
                        node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                s_node_encodings = self.norm_s(
                    s_node_encodings + messages_to_s
                )

                # Update object nodes (including subject nodes)
                messages_to_o = F.relu(torch.sum(
                    adjacency_matrix.t().softmax(dim=1)[..., None] *
                    self.sub_to_obj(
                        s_node_encodings,
                        box_pair_spatial_reshaped
                    ), dim=1)
                )
                node_encodings = self.norm_o(
                    node_encodings + messages_to_o
                )

            if targets is not None:
                all_labels.append(self.associate_with_ground_truth(
                    coords[x_keep], coords[y_keep], targets[b_idx])
                )

            all_box_pair_features.append(torch.cat([
                self.attention_head(
                    torch.cat([
                        s_node_encodings[x_keep],
                        node_encodings[y_keep]
                    ], 1),
                    box_pair_spatial_reshaped[x_keep, y_keep]
                ), self.attention_head_g(
                    global_features[b_idx, None],
                    box_pair_spatial_reshaped[x_keep, y_keep])
            ], dim=1))
            all_boxes_s.append(coords[x_keep])
            all_boxes_o.append(coords[y_keep])
            all_object_class.append(labels[y_keep])
            # The prior score is the product of the object detection scores
            all_prior.append(self.compute_prior_scores(
                x_keep, y_keep, scores, labels)
            )

            subject_counter += n_s
            object_counter += n_o

        return all_box_pair_features, all_boxes_s, all_boxes_o, \
               all_object_class, all_labels, all_prior
