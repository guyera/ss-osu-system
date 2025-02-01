# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import time
import itertools
from copy import deepcopy
import numpy as np
import os
from datetime import datetime

from enum import Enum
import pickle as pkl
from abc import ABC, abstractmethod
import sys
from tupleprediction.ewc import EWC_All_Models, EWC_Logit_Layers

from tqdm import tqdm
from torch.utils.data import\
    Dataset,\
    DataLoader,\
    Subset as TorchSubset,\
    ConcatDataset as TorchConcatDataset
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F

from boximagedataset import BoxImageDataset, BoxImageMemoryDataset
from utils import gen_custom_collate, gen_tqdm_description, LengthedIter
from transforms import\
    Compose,\
    Normalize,\
    ResizePad,\
    RandAugment,\
    RandomHorizontalFlip,\
    NoOpTransform
import sys, traceback, gc
def print_nan(t, name):
    if not torch.is_tensor(t):
        return
    if torch.any(torch.isnan(t)):
        print("NaNs present in", name)
        sys.exit(-1)
    elif torch.any(torch.isinf(t)):
        print("Infs present in", name)
        sys.exit(-1)

def contains_nan(tensor):
    return torch.isnan(tensor).any() 


class NonFeedbackDataset(Dataset):
    def __init__(self, features, species_labels, activity_labels):
        self.features = features
        self.species_labels = species_labels
        self.activity_labels = activity_labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.species_labels[idx], self.activity_labels[idx]

class FeedbackDatasetBatched(Dataset):
    def __init__(self, features, species_labels, activity_labels, batch_size):
        self.batch_size = batch_size
        self.batches = self.create_mini_batches(features, species_labels, activity_labels, batch_size)

    def create_mini_batches(self, features, species_labels, activity_labels, batch_size):
        batches = []
        for start_idx in range(0, len(features), batch_size):
            end_idx = min(start_idx + batch_size, len(features))
            batch = (features[start_idx:end_idx], species_labels[start_idx:end_idx], activity_labels[start_idx:end_idx])
            batches.append(batch)
        return batches

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class BoxPredictionGradBalancer(torch.autograd.Function):
    @staticmethod
    def forward(input, weights):
        return input

    @staticmethod
    def setup_context(ctx, inputs, output):
        input, weights = inputs
        ctx.weights = weights

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None

        if grad_output is not None:
            grad_input = grad_output * ctx.weights

        return grad_input, None

_balance_box_prediction_grads = BoxPredictionGradBalancer.apply

'''
Dynamic program solution to the multiple instance count problem

Notation:
    N: Number of images in the batch
    M_i: Number of box images in the ith image
    K: Number of classes
Parameters:
    predictions: List of N tensors. predictions[i] is a tensor of shape [M_i, K]
        predictions[i][j, k] is the predicted probability that the jth box of
        the ith image belongs to class k.
    targets: Tensor of shape [N, K]
        targets[i, k] is the ground truth number of boxes in image i that
        belong to class k.
'''
def multiple_instance_count_cross_entropy_dyn(predictions, targets, allow_print=False):
    losses = []
    # predictions = tuple(pred.double() for pred in predictions)
    # targets = targets.double()
    for img_predictions, img_targets in zip(predictions, targets):
        present_classes = img_targets != 0
        present_counts = img_targets[present_classes]
        present_predictions = img_predictions[:, present_classes]

        # Construct the dynamic program buffer. Its size is
        # [N+1, C_1+1, ... C_K+1], where N is the number of boxes, K is the
        # number of present classes, and C_k is the number of boxes belonging
        # to present class k. Note that the absent classes are implied to
        # have counts of 0, hence there are invisible [1, 1, ..., 1] dimensions
        # as well, but we can ignore those. Initialize the values to zero
        # since we compute them by summing over directions. However, set the
        # very first corner to 1, since we'll be using it as a starting
        # multiplier for the remaining cells.
        dyn_prog = torch.zeros(
            len(img_predictions) + 1,
            *((present_counts + 1).tolist()),
            device=targets.device
        )
        dyn_prog[tuple([0] * len(dyn_prog.shape))] = 1
        dyn_prog_shape_tensor = torch.tensor(
            list(dyn_prog.shape),
            dtype=torch.long,
            device=targets.device
        )

        # At any given point in our dynamic program, we will be maintaining
        # a tuple of per-dimension index tensors that we will use to index
        # the most-recently-computed diagonal plane of the dynamic program
        # buffer. For ease of tensorization, we'll store it as a 2D tensor
        # and convert it to a tuple for indexing on the fly. Initially,
        # we start on the [0, 0, ..., 0] location of our buffer (0 instances
        # observed for every class), and ONLY that location (so we have 1
        # index, equal to [0, 0, ..., 0])
        cur_dyn_indices = torch.zeros(
            len(present_counts) + 1,
            1,
            dtype=torch.long,
            device=targets.device
        )

        # At each iteration of our dynamic program, we can expand outward
        # from our current indices in K different directions, where each
        # direction k represents "count one more box, observing an instance of
        # class k". To compute these shifted indices, we'll add 1 to the
        # corresponding class dimension of cur_dyn_indices, and 1 to the
        # box index. We can tensorize this by instead adding the kth row of the
        # identity matrix augmented with a prepended "1's" column
        eye = torch.eye(len(present_counts), dtype=torch.long)
        aug_eye = torch.cat(
            (torch.ones(len(present_counts), 1, dtype=torch.long), eye),
            dim=1
        )
        aug_eye = aug_eye.to(targets.device)

        # For each iteration in our dynamic program
        for box_idx in range(len(present_predictions)):
            # Keep a running list of indices for the next iteration
            next_indices = []

            # For each direction of expansion within the dynamic program buffer
            for cls_idx in range(len(present_counts)):
                # Perform the tensorized index shift to expand +1 in the
                # box dimension and +1 in the current class dimension within
                # the dynamic program buffer
                shift = aug_eye[cls_idx]
                shifted_dyn_indices = cur_dyn_indices + shift[:, None]

                # Mask to remove indices which overflow past the bounds
                # of the dynamic program buffer
                overflow_mask = shifted_dyn_indices <\
                    dyn_prog_shape_tensor[:, None]
                overflow_mask = torch.all(overflow_mask, dim=0)
                shifted_dyn_indices = shifted_dyn_indices[:, overflow_mask]
                masked_cur_dyn_indices = cur_dyn_indices[:, overflow_mask]

                # Use the shifted dyn indices to index the expansion locations
                # of the dynamic buffer. Update those locations in the buffer;
                # the expansion represents observing one more instance of this
                # class relative to the buffer values indexed at
                # cur_dyn_indices, so (with a conditional independence
                # assumption across boxes) we multiply the dynamic program
                # buffer values indexed at cur_dyn_indices by the probability
                # that this box belongs to this class to perform the expansion
                # update. The update itself is done by adding this joint
                # probability to the existing shifted indices' probabilities,
                # since these joint probabilities are mutually exclusive and
                # we're trying to compute the logical OR across them. All
                # indexing is done by converting the indexing tensors to
                # indexing tuples, each component of which is an indexing tensor
                # for the corresponding dimension of the dynamic program buffer
                dyn_prog[tuple(shifted_dyn_indices)] +=\
                    dyn_prog[tuple(masked_cur_dyn_indices)] *\
                        present_predictions[box_idx, cls_idx]

                # Append shifted_dyn_indices to next_indices
                next_indices.append(shifted_dyn_indices)

            # Concatenate next_indices along dim 1 and assign it to
            # cur_dyn_indices
            cur_dyn_indices = torch.cat(next_indices, dim=1)

            # There will be some duplicates since we've expanded into the same
            # cells from multiple directions. Remove these duplicates, leaving
            # only a single copy of each unique index.
            cur_dyn_indices = torch.unique(cur_dyn_indices, dim=1)
        
        # Get P(predictions match count targets) = <the highest-index vertex
        # of the dynamic program>
        prob_match = dyn_prog[tuple(dyn_prog_shape_tensor - 1)]

        cur_loss = -torch.log(prob_match)
        if not torch.isinf(cur_loss):
            losses.append(cur_loss)

    # Compute aggregate loss across images
    if len(losses) > 0:
        losses = torch.stack(losses, dim=0)
        loss = losses.mean()
    else:
        loss = 0
    return loss


'''
Dynamic program solution to the multiple instance presence problem

Notation:
    N: Number of images in the batch
    M_i: Number of box images in the ith image
    K: Number of classes
Parameters:
    predictions: List of N tensors. predictions[i] is a tensor of shape [M_i, K]
        predictions[i][j, k] is the predicted probability that the jth box of
        the ith image belongs to class k.
    targets: Tensor of shape [N, K]
        targets[i, k] is the ground truth number of boxes in image i that
        belong to class k.
'''
def multiple_instance_presence_cross_entropy_dyn(predictions, targets, allow_print=False):
    losses = []
    # predictions = tuple(pred.double() for pred in predictions)
    # targets = targets.double()
    for img_predictions, img_targets in zip(predictions, targets):
        present_classes = img_targets != 0
        present_predictions = img_predictions[:, present_classes]
        if allow_print:
            print_nan(present_predictions, 'present_predictions')

        # Construct the dynamic program buffer. Its size is
        # [N+1, 2, 2, ... 2], where N is the number of boxes, and there are
        # K 2's, with K being the number of present classes. Initialize the
        # values to zero since we compute them by summing over directions.
        # However, set the very first corner to 1, since we'll be using it as a
        # starting multiplier for the remaining cells.
        dyn_prog = torch.zeros(
            len(present_predictions) + 1,
            *([2] * present_predictions.shape[1]),
            device=targets.device
        )
        dyn_prog[tuple([0] * len(dyn_prog.shape))] = 1
        dyn_prog_shape_tensor = torch.tensor(
            list(dyn_prog.shape),
            dtype=torch.long,
            device=targets.device
        )
        if allow_print:
            print_nan(dyn_prog, 'dyn_prog 1')

        # At any given point in our dynamic program, we will be maintaining
        # a tuple of per-dimension index tensors that we will use to index
        # the most-recently-computed diagonal plane of the dynamic program
        # buffer. For ease of tensorization, we'll store it as a 2D tensor
        # and convert it to a tuple for indexing on the fly. Initially,
        # we start on the [0, 0, ..., 0] location of our buffer (0 instances
        # observed for every class), and ONLY that location (so we have 1
        # index, equal to [0, 0, ..., 0])
        cur_dyn_indices = torch.zeros(
            present_predictions.shape[1] + 1,
            1,
            dtype=torch.long,
            device=targets.device
        )

        # At each iteration of our dynamic program, we can expand outward
        # from our current indices in K+1 different directions, where the
        # first direction represents "count one more box, observing no new
        # classes", and each remaining direction k represents "count one more
        # box, observing an instance of class k for the first time". To compute
        # these shifted indices, we'll add 1 to the corresponding class
        # dimension of cur_dyn_indices, and 1 to the box index. We can
        # tensorize this by instead adding the kth row of a carefully
        # constructed matrix where the first column is all 1's, and the
        # remaining columns are drawn from the K+1 identity matrix.
        aug_eye = torch.eye(
            present_predictions.shape[1] + 1,
            dtype=torch.long,
            device=targets.device
        )
        aug_eye[:, 0] = 1

        # For each iteration in our dynamic program
        for box_idx in range(len(present_predictions)):
            # Keep a running list of indices for the next iteration
            next_indices = []

            # For each direction of expansion within the dynamic program buffer
            for col_idx in range(present_predictions.shape[1] + 1):
                # Perform the tensorized index shift to expand +1 in the
                # box dimension and, if relevant, +1 in the current class
                # dimension within the dynamic program buffer
                shift = aug_eye[col_idx]
                shifted_dyn_indices = cur_dyn_indices + shift[:, None]

                # Mask to remove indices which overflow past the bounds
                # of the dynamic program buffer
                overflow_mask = shifted_dyn_indices <\
                    dyn_prog_shape_tensor[:, None]
                overflow_mask = torch.all(overflow_mask, dim=0)
                shifted_dyn_indices = shifted_dyn_indices[:, overflow_mask]
                masked_cur_dyn_indices = cur_dyn_indices[:, overflow_mask]

                if col_idx == 0:
                    # If col_idx is zero, then this iteration represents the
                    # expansion direction wherein a new box is counted, but
                    # no new classes are observed (all of the 0/1 indices in
                    # the dyn prog indices remain as-is, meaning we only
                    # observed a class which was already marked by a 1 index).
                    # To compute the probability associated with such a
                    # direction in our dynamic program, we need to sum over
                    # the probabilities corresponding to the previously
                    # observed classes for each relevant dyn prog index.

                    # Get current (== shifted) class indices w.r.t. the dynamic
                    # program
                    cur_cls_indices = masked_cur_dyn_indices[1:]
                    # equivalent to shifted_dyn_indices[1:]
                    
                    # Determine which ones have already been observed
                    # and sum over their prediction probabilities for this
                    # box
                    observed_masks = cur_cls_indices == 1
                    masked_cls_preds =\
                        present_predictions[box_idx, :, None] * observed_masks
                    observed_preds_sums = masked_cls_preds.sum(dim=0)

                    # These sums represent P(counted new box, observed a class
                    # that has already been observed before) for each current
                    # dynamic program value. Multiply these probabilities by
                    # the corresponding current dynamic program values to get
                    # the next ones
                    if contains_nan(dyn_prog):
                        print("Skipping batch due to NaN values in dyn_prog 2")
                        continue
                    dyn_prog[tuple(shifted_dyn_indices)] +=\
                        dyn_prog[tuple(masked_cur_dyn_indices)] *\
                            observed_preds_sums
                    

                    if allow_print:
                        print_nan(dyn_prog, 'dyn_prog 2')
                else:
                    # if col_idx is NOT zero, then col_idx - 1 represents
                    # the class which is now being observed, having previously
                    # not been observed. If it HAS previously been observed,
                    # then the shifted class index for it would be 2, which
                    # lies outside the dyn prog bounds and would have already
                    # been masked out as a valid shifted index.

                    # Compute class index
                    cls_idx = col_idx - 1

                    # The probability associated with this direction is
                    # simply P(this box == this class). Compute this probability
                    # and multiply it by the current dyn prog values to get
                    # the contribution to the next dyn prog values.
                    if contains_nan(dyn_prog):
                        print("Skipping batch due to NaN values in dyn_prog 2")
                        continue  
                    dyn_prog[tuple(shifted_dyn_indices)] +=\
                        dyn_prog[tuple(masked_cur_dyn_indices)] *\
                            present_predictions[box_idx, cls_idx]
                    
                    if allow_print:
                        print_nan(dyn_prog, 'dyn_prog 3')

                # Append shifted_dyn_indices to next_indices
                next_indices.append(shifted_dyn_indices)

            # Concatenate next_indices along dim 1 and assign it to
            # cur_dyn_indices
            cur_dyn_indices = torch.cat(next_indices, dim=1)

            # There will be some duplicates since we've expanded into the same
            # cells from multiple directions. Remove these duplicates, leaving
            # only a single copy of each unique index.
            cur_dyn_indices = torch.unique(cur_dyn_indices, dim=1)
        
        # Get P(predictions match count targets) = <the highest-index vertex
        # of the dynamic program>
        prob_match = dyn_prog[tuple(dyn_prog_shape_tensor - 1)]
        if allow_print:
            print_nan(dyn_prog, 'dyn_prog 4')
            print_nan(prob_match, 'prob_match')

        # NLL is loss
        cur_loss = -torch.log(prob_match)
        if not torch.isinf(cur_loss):
            losses.append(cur_loss)

    # Compute aggregate loss across images
    if len(losses) > 0:
        losses = torch.stack(losses, dim=0)
        loss = losses.mean()
    else:
        loss = 0
    return loss

'''
Notation:
    N: Number of images in the batch
    M_i: Number of box images in the ith image
    K: Number of classes
Parameters:
    predictions: List of N tensors. predictions[i] is a tensor of shape [M_i, K]
        predictions[i][j, k] is the predicted probability that the jth box of
        the ith image belongs to class k.
    targets: Tensor of shape [N, K]
        targets[i, k] is the ground truth number of boxes in image i that
        belong to class k.
'''
def multiple_instance_count_cross_entropy(predictions, targets):
    losses = []
    # predictions = tuple(pred.double() for pred in predictions)
    # targets = targets.double()
    # For each image
    for img_predictions, img_targets in zip(predictions, targets):
        # We're working with img_predictions of shape [M_i, K] and img_targets
        # of shape [K]. Index the predictions whose classes are present in
        # the ground truth labels
        present_classes = img_targets != 0
        remaining_predictions = img_predictions[:, present_classes]
        remaining_targets = img_targets[present_classes]
        combination_pred_running_product = None

        # The general strategy is as follows: remaining_predictions will be
        # iteratively updated, and in general its shape will be
        # [B, K, *c], where *c = (C_{N-1}, C_{N-2}, ..., C_1). C_i denotes
        # the number of combinations selected when considering present class
        # i. The combination dimensions are prepended rather than appended to
        # simplify broadcasting to keep track of the running cross product
        # predictions.

        # At each iteration, we'll consider combinations for the class whose
        # predictions are at remaining_predictions[:, 0] and whose counts are
        # at target[0]. Those index-0 dimensions are sliced off at each
        # iteration until all present classes have been considered.

        # Note that the last class is trivial since there's only one
        # possible combination: all remaining boxes must be assigned the one
        # remaining label.

        # For each present class
        for _ in range(present_predictions.shape[1]):
            ## Consider all of the ways in which we can assign this class label
            ## to the correct number of the remaining boxes
            cur_combinations = torch.tensor(
                list(
                    itertools.combinations(
                        list(range(len(remaining_predictions))),
                        remaining_targets[0]
                    )
                ),
                device=targets.device
            )

            ## Get the predictions for the current present class associated with
            ## each combination (which is definitionally the index-0 remaining
            ## class).
            combination_predictions = remaining_predictions[
                cur_combinations,
                0
            ]

            ## Final softmax probabilities for a given combination are computed
            ## by multiplying across selected boxes.
            combination_pred_product = torch.prod(
                combination_predictions,
                dim=1
            )

            ## The running product is stored as a Tensor whose dimensions, in
            ## reverse order (right-to-left), correspond to cross products of 
            ## combinations from past iterations of this for loop. If
            ## the running product is of shape [*shape], then
            ## combination_prediction_product is of shape [C, *shape], where
            ## C is len(cur_combinations). Broadcasting rules allow direct
            ## multiplication to continue expanding this in a cross-product
            ## fashion.
            if combination_pred_running_product is None:
                combination_pred_running_product = combination_pred_product
            else:
                combination_pred_running_product = \
                    combination_pred_running_product * combination_pred_product

            ## Update the remaining predictions by, for each combination,
            ## removing the selected boxes and filtering out the current present
            ## class

            # First, construct a compliment index for cur_combinations.
            # (Computing the complement of a multidimensional index is messy)
            all_boxes = torch.ones(
                *(cur_combinations.shape),
                dtype=torch.bool,
                device=targets.device
            )
            offset = torch.arange(len(all_boxes), device=targets.device) *\
                all_boxes.shape[1]
            flattened_compliment = all_boxes.flatten()
            flattened_index = (cur_combinations + offset[:, None]).flatten()
            flattened_compliment[flattened_index] = False # Compliment step
            compliment_mask = flattened_compliment.reshape(*(all_boxes.shape))
            index_pool = torch.arange(
                all_boxes.shape[1],
                device=targets.device
            )
            cur_combinations_compliment = torch.stack(
                [index_pool[mask_row] for mask_row in compliment_mask],
                dim=0
            )

            # For each combination compliment, get the predictions for the
            # remaining present classes other than the current one. This
            # filters out the boxes selected for each combination as well
            # as the current present class.
            combination_compliment_predictions = remaining_predictions[
                cur_combinations_compliment,
                1:
            ]

            # remaining_predictions is shaped [B, K, *c], B is the number
            # of remaining boxes after selecting combinations denoted by
            # the dimensions contained in *c --- that is, *c contains
            # all of the numbers of combinations selected at each iteration
            # in reverse order, and B contains the number of boxes remaining
            # after all stages of combination selection.
            # combination_compliment_predictions, however, is shaped
            # [C, B, K, *prev_c], where *(C, *prev_c) = *c. To update
            # remaining_predictions, permute combination_compliment_predictions
            # to match the desired shape
            remaining_predictions = combination_compliment_predictions.permute((
                1,
                2,
                0,
                *(list(range(3, len(combination_compliment_predictions.shape))))
            ))

            ## Update remaining targets
            remaining_targets = remaining_targets[1:]

        ## Sum over combination_pred_running_products to compute the
        ## probability for the exhaustive logical OR satisfying the targets
        sum_prob = combination_pred_running_product.sum()

        ## Compute per-image loss
        cur_loss = -torch.log(sum_prob)
        if not torch.isinf(cur_loss):
            losses.append(cur_loss)

    ## Aggregate per-image losses and return
    if len(losses) > 0:
        losses = torch.stack(losses, dim=0)
        return losses.mean()
    else:
        return 0


'''
Notation:
    N: Number of images in the batch
    M_i: Number of box images in the ith image
    K: Number of classes
Parameters:
    predictions: List of N tensors. predictions[i] is a tensor of shape [M_i, K]
        predictions[i][j, k] is the predicted probability that the jth box of
        the ith image belongs to class k.
    targets: Boolean tensor of shape [N, K]
        targets[i, k] is the ground truth presence boolean for class k in image
        i.
'''
def multiple_instance_presence_cross_entropy(predictions, targets):
    losses = []
    # predictions = tuple(pred.double() for pred in predictions)
    # targets = targets.double()
    # For each image
    for img_predictions, img_targets in zip(predictions, targets):
        # We're working with img_predictions of shape [M_i, K] and img_targets
        # of shape [K].

        # The general strategy is as follows:
        # P(A, B, C present, others absent) = P(A, B, C present | others absent)
        #       * P(others absent).
        #
        # The second term, P(others absent), can be computed as
        # P(each box belongs to either A, B, or C)
        #       = \prod_{j}(P(y_j = A, B, or C))
        #
        # To compute the first term, we first compute P(y_j = K | others absent)
        # for K \in {A, B, C} by filtering out the absent classes and
        # renormalizing the remaining probabilities.
        # 
        # From here on out, everything is conditioned on <others absent>,
        # but it's left out of the notation for brevity.
        # 
        # Next, we can compute:
        # P(A, B, C present) = 1 - P(A, B, or C is absent)
        # 
        # We can actually compute this complement directly. For two events,
        # P(A or B) = P(A) + P(B) - P(A and B). But to generalize the formula
        # to N events, we start with the single-event conjunctions, then
        # subtract the two-event conjunctions, then add the three-event
        # conjunctions, then subtract the four-event conjunctions, and so on.
        # e.g., for three classes:
        # P (A or B or C) = P(A) + P(B) + P(C) - P(A and B) - P(A and C)
        #       - P(B and C) + P(A and B and C).
        # In our case, P(A, B, or C absent)
        #       = P(A absent) + P(B absent) + P(C absent)
        #           - P(A, B absent) - P(A, C absent) - P(B, C absent)
        #           + P(A, B, C absent)
        # We can compute each conjunction easily. For instance:
        # P(A, B, C absent) = \prod_j(1 - P(y_j \in {A, B, C})).
        # In total, we have to compute \sum_{i=1}^{N}(N choose i)
        # conjunctions, where N is the number of present classes, each of which
        # involves summing over i columns and then computing a product across
        # the rows (and there's a row per box).

        # Step 1: Compute P(others absent)
        present_predictions = img_predictions[:, img_targets]
        prob_others_absent = torch.prod(present_predictions.sum(dim=1))

        # Step 2: Filter out present-class predictions and condition on
        # the event <others absent>
        # TODO handle case where denominator == 0. This might require moving
        # computations to the log space
        cond_present_predictions = present_predictions / \
            present_predictions.sum(dim=1, keepdim=True)

        # Step 3: For each i in 1, ..., N, alternate adding and subtracting
        # the relevant N choose i conjunction probabilities to compute
        # P(A, B, or C absent)
        prob_any_absent = 0
        add_iteration = True
        for i in range(1, cond_present_predictions.shape[1] + 1):
            # Compute the class indices for the N choose i conjunctions
            cur_combinations = torch.tensor(
                list(
                    itertools.combinations(
                        list(range(cond_present_predictions.shape[1])),
                        i
                    )
                ),
                device=targets.device
            )

            # Index the classes using cur_combinations
            combination_preds = cond_present_predictions[:, cur_combinations]
            # combination_preds is of shape [B, C, i], where C is N choose i
            # and represents a conjunction.

            # Compute absence conjunction probabilities for each combination:
            # P(conjunction of absences) = 1 - P(disjunction present)
            combination_box_absence = 1 - combination_preds.sum(dim=2)
            combination_absence = torch.prod(combination_box_absence, dim=0)

            # If add_iteration is True, add these absence conjunction
            # probabilities. Else, subtract them. i.e.,
            # P(A, B, or C)
            #   = P(A) + P(B) + P(C)
            #   - P(A, B) - P(A, C) - P(B, C)
            #   + P(A, B, C)
            absence_conjunction_sum = combination_absence.sum()
            if add_iteration:
                prob_any_absent =\
                    prob_any_absent + absence_conjunction_sum
            else:
                prob_any_absent =\
                    prob_any_absent - absence_conjunction_sum

            # Toggle add_iteration
            add_iteration = not add_iteration

        # Step 4: We have
        # P(any of the ground-truth present classes are absent | others absent).
        # Compute the compliment to get
        # P(present classes are predicted present | others absent).
        prob_all_present = 1 - prob_any_absent

        # Step 5: Compute P(A, B, ... C present, others Absent)
        #       = P(A, B, ..., C present | others absent) * P(others absent)
        prob_conform = prob_all_present * prob_others_absent

        ## Compute per-image loss
        cur_loss = -torch.log(prob_conform)
        if not torch.isinf(cur_loss):
            losses.append(cur_loss)

    ## Aggregate per-image losses and return
    if len(losses) > 0:
        losses = torch.stack(losses, dim=0)
        return losses.mean()
    else:
        return 0


class Augmentation(Enum):
    rand_augment = {
        'ctor': RandAugment
    }
    horizontal_flip = {
        'ctor': RandomHorizontalFlip
    }
    none = {
        'ctor': NoOpTransform
    }

    def __str__(self):
        return self.name

    def ctor(self):
        return self.value['ctor']


class SchedulerType(Enum):
    cosine = {
        'ctor': CosineAnnealingLR
    }
    none = {
        'ctor': None
    }

    def __str__(self):
        return self.name

    def ctor(self):
        return self.value['ctor']


class LossFnEnum(Enum):
    cross_entropy = 'cross-entropy'
    focal = 'focal'

    def __str__(self):
        return self.value


class BackboneTrainingType(Enum):
    end_to_end = 'end-to-end'
    classifiers = 'classifiers'
    side_tuning = 'side-tuning'

    def __str__(self):
        return self.value


class ClassifierTrainer(ABC):
    @abstractmethod
    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        return NotImplemented

    @abstractmethod
    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        return NotImplemented

    @abstractmethod
    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        return NotImplemented


'''
Custom Subset dataset class that works with BoxImageDatasets and derivatives,
forwarding label_dataset() and box_count() to the underlying dataset
appropriately.
'''
class Subset(Dataset):
    def __init__(self, dataset, indices):
        self._dataset = dataset
        self._indices = indices

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        return self._dataset[self._indices[idx]]

    def label_dataset(self):
        return TorchSubset(self._dataset.label_dataset(), self._indices)

    def box_count(self, idx):
        return self._dataset.box_count(self._indices[idx])


'''
Custom ConcatDataset class that works with BoxImageDatasets and derivatives,
forwarding label_dataset() and box_count() to the underlying dataset
appropriately.
'''
class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self._datasets = datasets
        self._lens = [len(x) for x in datasets]
        self._len = sum(self._lens)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        len_idx = 0
        while idx >= self._lens[len_idx]:
            idx -= self._lens[len_idx]
            len_idx += 1
        return self._datasets[len_idx][idx]

    def label_dataset(self):
        return TorchConcatDataset([x.label_dataset() for x in self._datasets])

    def box_count(self, idx):
        len_idx = 0
        while idx >= self._lens[len_idx]:
            idx -= self._lens[len_idx]
            len_idx += 1
        return self._datasets[len_idx].box_count(idx)


class FlattenedBoxImageDataset(Dataset):
    def __init__(self, box_image_dataset):
        self._dataset = box_image_dataset
        box_counts = [
            box_image_dataset.box_count(x)\
                for x in range(len(box_image_dataset))
        ]
        box_to_img_mapping = []
        box_to_local_box_mapping = []
        for img_idx, box_count in enumerate(box_counts):
            for i in range(box_count):
                box_to_img_mapping.append(img_idx)
                box_to_local_box_mapping.append(i)
        self._box_to_img_mapping = box_to_img_mapping
        self._box_to_local_box_mapping = box_to_local_box_mapping

    def __len__(self):
        return len(self._box_to_img_mapping)

    def __getitem__(self, idx):
        
        img_idx = self._box_to_img_mapping[idx]
        local_box_idx = self._box_to_local_box_mapping[idx]
        species_labels, activity_labels, _, box_images, _ =\
            self._dataset[img_idx]
        # Flatten / concatenate the box images and repeat the
        # labels per-box
        one_hot_species_label = torch.argmax(species_labels, dim=0)
        one_hot_activity_label = torch.argmax(activity_labels.float(), dim=0)

        # print(f"img_idx {img_idx}, idx {idx}, local_box_idx {local_box_idx}, lenght of box images {len(box_images)}, self._box_to_local_box_mapping[idx {self._box_to_local_box_mapping[idx]}")
        box_image = box_images[local_box_idx]


        return one_hot_species_label, one_hot_activity_label, box_image


'''
Wraps a dataset around an existing dataset and a precomputed feature
tensor for the whole dataset. Indexing at i returns the underlying data
from the dataset as well as a sub-tensor of the feature tensor
indexed at i along dim 0 (most likely the ith datapoint's feature vector).
'''
class FeatureConcatDataset(Dataset):
    def __init__(self, dataset, box_features):
        self._dataset = dataset
        self._box_features = box_features

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        data = self._dataset[idx]
        box_features = self._box_features[idx]
        return data + (box_features,)


class TransformingBoxImageDataset(Dataset):
    def __init__(self, dataset, transform):
        self._dataset = dataset
        self._transform = transform

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        species_labels,\
            activity_labels,\
            novelty_type_labels,\
            box_images,\
            whole_image =\
                self._dataset[idx]
        # box_images = self._transform(box_images)
        if len(box_images) > 0:
            box_images = self._transform(box_images)
        whole_image = self._transform(whole_image)

        return species_labels,\
            activity_labels,\
            novelty_type_labels,\
            box_images,\
            whole_image

    def label_dataset(self):
        return self._dataset.label_dataset()

    def box_count(self, idx):
        return self._dataset.box_count(idx)


class FeedbackSamplingConfiguration(ABC):
    @abstractmethod
    def configure_datasets(self, train_dataset, feedback_dataset):
        return NotImplemented

    @abstractmethod
    def configure_data(
            self,
            train_box_features,
            train_species_labels,
            train_activity_labels,
            feedback_box_features,
            feedback_species_labels,
            feedback_activity_labels):
        return NotImplemented


class CombinedFeedbackSamplingConfiguration(FeedbackSamplingConfiguration):
    def configure_datasets(self, train_dataset, feedback_dataset):
        return None, ConcatDataset((train_dataset, feedback_dataset))

    def configure_data(
            self,
            train_box_features,
            train_species_labels,
            train_activity_labels,
            feedback_box_features,
            feedback_species_labels,
            feedback_activity_labels):
        # Convert train_species_labels to count vectors and
        # train_activity_labels to presence vectors
        train_species_labels = torch.nn.functional.one_hot(
            train_species_labels,
            feedback_species_labels.shape[1]
        )
        train_activity_labels = torch.nn.functional.one_hot(
            train_activity_labels,
            feedback_activity_labels.shape[1]
        )
        return None,\
            None,\
            None,\
            torch.split(train_box_features, 1) + feedback_box_features,\
            torch.cat((train_species_labels, feedback_species_labels), dim=0),\
            torch.cat((train_activity_labels, feedback_activity_labels), dim=0)

def none_ctor():
    return None

class FeedbackSamplingConfigurationOption(Enum):
    combined = {
        'ctor': CombinedFeedbackSamplingConfiguration
    }
    none = {
        'ctor': none_ctor,
    }

    def __str__(self):
        return self.name

    def ctor(self):
        return self.value['ctor']


class DistributedRandomBoxImageBatchSampler:
    def __init__(
            self,
            box_counts,
            boxes_per_batch,
            num_replicas,
            rank,
            seed=0):
        box_counts = torch.tensor(
            box_counts,
            dtype=torch.long
        )
        self._box_counts = box_counts
        self._boxes_per_batch = boxes_per_batch
        self._num_replicas = num_replicas
        self._rank = rank
        self._generator = np.random.default_rng(seed=seed)

    def __iter__(self):
        # Randomly permute indices and corresponding box counts
        unpermuted_indices = list(range(len(self._box_counts)))
        indices = self._generator.permuted(unpermuted_indices)
        box_counts = self._box_counts[indices]

        # Determine the batch organization for each replica
        all_batches = []
        cur_batches = [[] for _ in range(self._num_replicas)]
        cur_batch_sizes = [0] * self._num_replicas
        for img_idx in range(len(box_counts)):
            # Determine which rank's batch to add this image to---the one with
            # the smallest current batch size
            cur_batch_size = min(cur_batch_sizes)
            cur_rank = cur_batch_sizes.index(cur_batch_size)

            # Determine whether we can add the image to the selected batch.
            # If not, flush the current batches and resume.
            if cur_batch_size + box_counts[img_idx] > self._boxes_per_batch \
                    and cur_batch_size != 0:
                all_batches.append(cur_batches[self._rank])
                cur_batches = [[] for _ in range(self._num_replicas)]
                cur_batch_sizes = [0] * self._num_replicas

                cur_batch_size = 0
                cur_rank = 0

            # Append the image to the smallest batch
            # cur_batches[cur_rank].append(img_idx)
            cur_batches[cur_rank].append(indices[img_idx])
            cur_batch_sizes[cur_rank] += box_counts[img_idx]

        # Finished adding all images to batches. If all current batches are
        # non-empty, flush them one last time.
        min_batch_size = min(cur_batch_sizes)
        if min_batch_size != 0:
            all_batches.append(cur_batches[self._rank])

        # Return an iterator for the batches corresponding to this replica's
        # rank (resting assured that all replicas will receive the same
        # number of batches by construction)
        return LengthedIter(all_batches)

    def __len__(self):
        copy_sampler = deepcopy(self)
        next_iter = iter(copy_sampler)
        return len(next_iter)


class EWCLogitLayerClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_feature_file,
            val_feature_file,
            box_transform,
            post_cache_train_transform,
            feedback_batch_size=32,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            feedback_loss_weight=0.5,
            loss_fn=LossFnEnum.cross_entropy,
            class_frequencies=None,
            ewc_lambda= 1000):
        self.ewc_lambda = ewc_lambda
        self._lr = lr
        self._train_feature_file = train_feature_file
        self._val_feature_file = val_feature_file
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._feedback_batch_size = feedback_batch_size
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._feedback_loss_weight = feedback_loss_weight
        self._loss_fn = loss_fn
        self._class_frequencies = class_frequencies

    # def _train_epoch(
    #         self,
    #         box_features,
    #         species_labels,
    #         activity_labels,
    #         feedback_box_features,
    #         feedback_species_labels,
    #         feedback_activity_labels,
    #         species_classifier,
    #         activity_classifier,
    #         optimizer,
    #         device,
    #         feedback_class_frequencies):
    #     # Set everything to train mode
    #     species_classifier.train()
    #     activity_classifier.train()

    #     # Compute class weights
    #     species_weights = None
    #     activity_weights = None
    #     species_frequencies = None
    #     activity_frequencies = None
    #     if self._class_frequencies is not None:
    #         train_species_frequencies,\
    #             train_activity_frequencies = self._class_frequencies
    #         train_species_frequencies = train_species_frequencies.to(device)
    #         train_activity_frequencies = train_activity_frequencies.to(device)
            
    #         species_frequencies = train_species_frequencies
    #         activity_frequencies = train_activity_frequencies

    #     if feedback_class_frequencies is not None:
    #         feedback_species_frequencies,\
    #             feedback_activity_frequencies = feedback_class_frequencies
    #         feedback_species_frequencies =\
    #             feedback_species_frequencies.to(device)
    #         feedback_activity_frequencies =\
    #             feedback_activity_frequencies.to(device)

    #         if species_frequencies is None:
    #             species_frequencies = feedback_species_frequencies
    #             activity_frequencies = feedback_activity_frequencies
    #         else:
    #             species_frequencies =\
    #                 species_frequencies + feedback_species_frequencies
    #             activity_frequencies =\
    #                 activity_frequencies + feedback_activity_frequencies

    #     if species_frequencies is not None:
    #         species_proportions = species_frequencies /\
    #             species_frequencies.sum()
    #         unnormalized_species_weights =\
    #             torch.pow(1.0 / species_proportions, 1.0 / 3.0)
    #         unnormalized_species_weights[species_proportions == 0.0] = 0.0
    #         proportional_species_sum =\
    #             (species_proportions * unnormalized_species_weights).sum()
    #         species_weights =\
    #             unnormalized_species_weights / proportional_species_sum

    #         activity_proportions = activity_frequencies /\
    #             activity_frequencies.sum()
    #         unnormalized_activity_weights =\
    #             torch.pow(1.0 / activity_proportions, 1.0 / 3.0)
    #         unnormalized_activity_weights[activity_proportions == 0.0] = 0.0
    #         proportional_activity_sum =\
    #             (activity_proportions * unnormalized_activity_weights).sum()
    #         activity_weights =\
    #             unnormalized_activity_weights / proportional_activity_sum

        
    #     ## Feedback loss

    #     # Flatten feedback box features, compute predictions, and re-split
    #     # per-image
    #     feedback_loss = 0
    #     feedback_n_species_correct = 0
    #     feedback_n_activity_correct = 0
    #     feedback_n_examples = 0
    #     if feedback_box_features is not None and\
    #             feedback_species_labels is not None and\
    #             feedback_activity_labels is not None:
    #         flattened_feedback_box_features = torch.cat(
    #             feedback_box_features,
    #             dim=0
    #         )
    #         flattened_feedback_species_logits = species_classifier(
    #             flattened_feedback_box_features
    #         )
    #         flattened_feedback_activity_logits = activity_classifier(
    #             flattened_feedback_box_features
    #         )
    #         flattened_feedback_species_preds = torch.nn.functional.softmax(
    #             flattened_feedback_species_logits,
    #             dim=1
    #         )
    #         flattened_feedback_activity_preds = torch.nn.functional.softmax(
    #             flattened_feedback_activity_logits,
    #             dim=1
    #         )

    #         # If class balancing, scale the gradients according to class
    #         # weights
    #         if species_weights is not None:
    #             flattened_feedback_species_preds =\
    #                 _balance_box_prediction_grads(
    #                     flattened_feedback_species_preds,
    #                     species_weights
    #                 )
    #             flattened_feedback_activity_preds =\
    #                 _balance_box_prediction_grads(
    #                     flattened_feedback_activity_preds,
    #                     activity_weights
    #                 )

    #         feedback_box_counts = [len(x) for x in feedback_box_features]
    #         feedback_species_preds = torch.split(
    #             flattened_feedback_species_preds,
    #             feedback_box_counts,
    #             dim=0
    #         )
    #         feedback_activity_preds = torch.split(
    #             flattened_feedback_activity_preds,
    #             feedback_box_counts,
    #             dim=0
    #         )

    #         # Logging metrics
    #         feedback_box_counts_t = torch.tensor(
    #             feedback_box_counts,
    #             device=device,
    #             dtype=torch.long
    #         )
    #         single_box_mask = feedback_box_counts_t == 1
    #         single_box_indices = torch.arange(
    #             len(single_box_mask),
    #             dtype=torch.long,
    #             device=device
    #         )[single_box_mask]
    #         if len(single_box_indices) > 0:
    #             single_box_species_preds = torch.cat(
    #                 [
    #                     feedback_species_preds[i] for i in single_box_indices
    #                 ],
    #                 dim=0
    #             )
    #             single_box_activity_preds = torch.cat(
    #                 [
    #                     feedback_activity_preds[i] for i in single_box_indices
    #                 ],
    #                 dim=0
    #             )
    #             single_box_species_labels =\
    #                 feedback_species_labels[single_box_indices]
    #             single_box_activity_labels =\
    #                 feedback_activity_labels[single_box_indices]
    #             feedback_species_correct =\
    #                 torch.argmax(single_box_species_preds, dim=1) ==\
    #                     torch.argmax(single_box_species_labels, dim=1)
    #             feedback_activity_correct =\
    #                 torch.argmax(single_box_activity_preds, dim=1) ==\
    #                     torch.argmax(single_box_activity_labels.to(torch.long), dim=1)
    #             feedback_n_species_correct =\
    #                 feedback_species_correct.to(torch.int).sum()
    #             feedback_n_activity_correct =\
    #                 feedback_activity_correct.to(torch.int).sum()

    #             feedback_n_examples = len(single_box_species_labels)

    #         # Compute loss
    #         # We have image-level count feedback labels for species
    #         feedback_species_loss = multiple_instance_count_cross_entropy_dyn(
    #             feedback_species_preds,
    #             feedback_species_labels
    #         )
    #         # We have image-level presence feedback labels for activities
    #         feedback_activity_loss = multiple_instance_presence_cross_entropy_dyn(
    #             feedback_activity_preds,
    #             feedback_activity_labels
    #         )
    #         feedback_loss = feedback_species_loss + feedback_activity_loss

    #         # Compute loss as weighted average between feedback and non-feedback
    #         # losses
    #     ewc_penalty_ = self.ewc_calculation.penalty(species_classifier, activity_classifier)
    #     non_feedback_loss = 10000 * ewc_penalty_ 

    #     loss =  feedback_loss + non_feedback_loss
    #     n_species_correct =feedback_n_species_correct
    #     n_activity_correct = feedback_n_activity_correct
    #     n_examples = feedback_n_examples

    #     # Optimizer step
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()

    #     mean_species_accuracy = float(n_species_correct) / n_examples
    #     mean_activity_accuracy = float(n_activity_correct) / n_examples

    #     mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

    #     return loss.detach().cpu().item(), mean_accuracy
    def _train_epoch(
            self,
            box_features,
            species_labels,
            activity_labels,
            feedback_box_features,
            feedback_species_labels,
            feedback_activity_labels,
            species_classifier,
            activity_classifier,
            optimizer,
            device,
            feedback_class_frequencies):
        # Set everything to train mode
        species_classifier.train()
        activity_classifier.train()

        # Compute class weights
        species_weights = None
        activity_weights = None
        species_frequencies = None
        activity_frequencies = None
        if self._class_frequencies is not None:
            train_species_frequencies,\
                train_activity_frequencies = self._class_frequencies
            train_species_frequencies = train_species_frequencies.to(device)
            train_activity_frequencies = train_activity_frequencies.to(device)
            
            species_frequencies = train_species_frequencies
            activity_frequencies = train_activity_frequencies

        if feedback_class_frequencies is not None:
            feedback_species_frequencies,\
                feedback_activity_frequencies = feedback_class_frequencies
            feedback_species_frequencies =\
                feedback_species_frequencies.to(device)
            feedback_activity_frequencies =\
                feedback_activity_frequencies.to(device)

            if species_frequencies is None:
                species_frequencies = feedback_species_frequencies
                activity_frequencies = feedback_activity_frequencies
            else:
                species_frequencies =\
                    species_frequencies + feedback_species_frequencies
                activity_frequencies =\
                    activity_frequencies + feedback_activity_frequencies

        if species_frequencies is not None:
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

        
        ## Feedback loss

        # Flatten feedback box features, compute predictions, and re-split
        # per-image
        feedback_loss = 0
        feedback_n_species_correct = 0
        feedback_n_activity_correct = 0
        feedback_n_examples = 0
        all_loss =0
        n_examples =0 
        n_species_correct = 0
        n_activity_correct = 0
        feedback_dataset = FeedbackDatasetBatched(feedback_box_features, feedback_species_labels, feedback_activity_labels, batch_size = self._feedback_batch_size)
        feedback_loader = DataLoader(feedback_dataset, batch_size= 1, shuffle=True, num_workers=0)
        
        for feedback_data in tqdm(feedback_loader, desc='EWC Training Progress'):
            feedback_box_features_, feedback_species_labels_, feedback_activity_labels_ = feedback_data
            if feedback_box_features is not None and\
                    feedback_species_labels is not None and\
                    feedback_activity_labels is not None:

                feedback_box_features_ = [torch.squeeze(f, dim=0) for f in feedback_box_features_]
                feedback_species_labels_ =  torch.squeeze(feedback_species_labels_, dim=0) 
                feedback_activity_labels_ = torch.squeeze(feedback_activity_labels_, dim=0)
                flattened_feedback_box_features = torch.cat(
                    feedback_box_features_,
                    dim=0
                )
                flattened_feedback_species_logits = species_classifier(
                    flattened_feedback_box_features
                )
                flattened_feedback_activity_logits = activity_classifier(
                    flattened_feedback_box_features
                )
                flattened_feedback_species_preds = torch.nn.functional.softmax(
                    flattened_feedback_species_logits,
                    dim=1
                )
                flattened_feedback_activity_preds = torch.nn.functional.softmax(
                    flattened_feedback_activity_logits,
                    dim=1
                )

                # If class balancing, scale the gradients according to class
                # weights
                if species_weights is not None:
                    flattened_feedback_species_preds =\
                        _balance_box_prediction_grads(
                            flattened_feedback_species_preds,
                            species_weights
                        )
                    flattened_feedback_activity_preds =\
                        _balance_box_prediction_grads(
                            flattened_feedback_activity_preds,
                            activity_weights
                        )

                feedback_box_counts = [len(x) for x in feedback_box_features_]
                feedback_species_preds = torch.split(
                    flattened_feedback_species_preds,
                    feedback_box_counts,
                    dim=0
                )
                feedback_activity_preds = torch.split(
                    flattened_feedback_activity_preds,
                    feedback_box_counts,
                    dim=0
                )

                # Logging metrics
                feedback_box_counts_t = torch.tensor(
                    feedback_box_counts,
                    device=device,
                    dtype=torch.long
                )
                single_box_mask = feedback_box_counts_t == 1
                single_box_indices = torch.arange(
                    len(single_box_mask),
                    dtype=torch.long,
                    device=device
                )[single_box_mask]
                if len(single_box_indices) > 0:
                    single_box_species_preds = torch.cat(
                        [
                            feedback_species_preds[i] for i in single_box_indices
                        ],
                        dim=0
                    )
                    single_box_activity_preds = torch.cat(
                        [
                            feedback_activity_preds[i] for i in single_box_indices
                        ],
                        dim=0
                    )
                    single_box_species_labels =\
                        feedback_species_labels[single_box_indices]
                    single_box_activity_labels =\
                        feedback_activity_labels[single_box_indices]
                    feedback_species_correct =\
                        torch.argmax(single_box_species_preds, dim=1) ==\
                            torch.argmax(single_box_species_labels, dim=1)
                    feedback_activity_correct =\
                        torch.argmax(single_box_activity_preds, dim=1) ==\
                            torch.argmax(single_box_activity_labels.to(torch.long), dim=1)
                    feedback_n_species_correct =\
                        feedback_species_correct.to(torch.int).sum()
                    feedback_n_activity_correct =\
                        feedback_activity_correct.to(torch.int).sum()

                    feedback_n_examples = len(single_box_species_labels)

                # Compute loss
                # We have image-level count feedback labels for species
                feedback_species_loss = multiple_instance_count_cross_entropy_dyn(
                    feedback_species_preds,
                    feedback_species_labels_
                )
                # We have image-level presence feedback labels for activities
                feedback_activity_loss = multiple_instance_presence_cross_entropy_dyn(
                    feedback_activity_preds,
                    feedback_activity_labels_
                )
                feedback_loss = feedback_species_loss + feedback_activity_loss

                # Compute loss as weighted average between feedback and non-feedback
                # losses
            ewc_penalty_ = self.ewc_calculation.penalty(species_classifier, activity_classifier)
            non_feedback_loss = self.ewc_lambda * ewc_penalty_ 

            loss =  feedback_loss + non_feedback_loss

            # print( f'feedback_loss {feedback_loss} non_feedback_loss {non_feedback_loss} ewc_penalty_ {ewc_penalty_}')
            # Gradient and optimizer steps
            optimizer.zero_grad()
            loss.backward()
            for param in species_classifier.parameters():
                if param.grad is not None and \
                (torch.any(torch.isnan(param.grad.data)) or \
                    torch.any(torch.isinf(param.grad.data))):
                    # Found some NaNs in the gradients. "Skip" this batch by
                    # zeroing the gradients. This should work in distributed
                    # mode as well
                    print('Some NaNs in the gradients of species_classifier. Skiping this batch ...')
                    optimizer.zero_grad()
                    break
                    
            for param in activity_classifier.parameters():
                if param.grad is not None and \
                (torch.any(torch.isnan(param.grad.data)) or \
                    torch.any(torch.isinf(param.grad.data))):
                    # Found some NaNs in the gradients. "Skip" this batch by
                    # zeroing the gradients. This should work in distributed
                    # mode as well
                    print('Some NaNs in the gradients of activity_classifier. Skiping this batch ...')
                    optimizer.zero_grad()
                    break
            
            optimizer.step()

            n_species_correct += feedback_n_species_correct
            n_activity_correct += feedback_n_activity_correct
            n_examples += feedback_n_examples
            all_loss+=loss

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return all_loss.detach().cpu().item()/ n_examples, mean_accuracy

    def _val_epoch(
            self,
            box_features,
            species_labels,
            activity_labels,
            species_classifier,
            activity_classifier):
        with torch.no_grad():
            species_classifier.eval()
            activity_classifier.eval()

            species_preds = species_classifier(box_features)
            activity_preds = activity_classifier(box_features)

            species_correct = torch.argmax(species_preds, dim=1) == \
                species_labels
            n_species_correct = int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(activity_preds, dim=1) == \
                activity_labels
            n_activity_correct = int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )

            n_examples = species_labels.shape[0]
            mean_species_accuracy = float(n_species_correct) / n_examples
            mean_activity_accuracy = float(n_activity_correct) / n_examples

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    '''
    Params:
        species_classifier: ClassifierV2
            Species classifier to train
        activity_classifier: ClassifierV2
            Activity classifier to train
        root_log_dir: str
            Root directory for logging training. Should include named transform
            paths, if appropriate.
    '''
    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            self._lr,
            momentum=0.9,
            weight_decay=1e-5
        )

        # Define convergence parameters (early stopping + model selection)
        epochs_since_improvement = 0
        best_accuracy = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None
        mean_train_loss = None
        mean_train_accuracy = None
        mean_val_accuracy = None

        # If we didn't load an optimizer state dict, and so the scheduler
        # hasn't been constructed yet, then construct it
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}

        def get_log_dir():
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'logit-layer-classifier-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )
        


        train_box_features, train_species_labels, train_activity_labels =\
            torch.load(
                self._train_feature_file,
                map_location=device
            )
        val_box_features, val_species_labels, val_activity_labels =\
            torch.load(
                self._val_feature_file,
                map_location=device
            )

        if feedback_dataset is not None:
            feedback_box_counts = [
                feedback_dataset.box_count(i) for\
                    i in range(len(feedback_dataset))
            ]
            feedback_batch_sampler = DistributedRandomBoxImageBatchSampler(feedback_box_counts, self._feedback_batch_size, 1, 0)
            feedback_loader = DataLoader(
                feedback_dataset,
                num_workers=0,
                batch_sampler=feedback_batch_sampler,
                collate_fn=gen_custom_collate()
            )
            # Precompute feedback backbone features
            backbone.eval()
            feedback_box_features = []
            feedback_species_labels = []
            feedback_activity_labels = []
            with torch.no_grad():
                for batch_species_labels,\
                        batch_activity_labels,\
                        _,\
                        batch_box_images,\
                        _ in feedback_loader:
                    # Move batch to device
                    batch_species_labels =\
                        batch_species_labels.to(device)
                    batch_activity_labels =\
                        batch_activity_labels.to(device)
                    batch_box_images =\
                        [b.to(device) for b in batch_box_images]

                    # Get list of per-image box counts
                    batch_box_counts = [len(x) for x in batch_box_images]

                    # Flatten box images and compute features
                    flattened_box_images = torch.cat(batch_box_images, dim=0)
                    batch_box_features = backbone(flattened_box_images)

                    # Use batch_box_counts to split the computed features back
                    # to per-image feature tensors and concatenate to
                    # feedback_box_features
                    feedback_box_features += torch.split(
                        batch_box_features,
                        batch_box_counts,
                        dim=0
                    )

                    # Record labels
                    feedback_species_labels.append(batch_species_labels)
                    feedback_activity_labels.append(batch_activity_labels)
                    

                # Concatenate labels
                feedback_species_labels = torch.cat(
                    feedback_species_labels,
                    dim=0
                )
                feedback_activity_labels = torch.cat(
                    feedback_activity_labels,
                    dim=0
                )
        else:
            feedback_box_features = None
            feedback_species_labels = None
            feedback_activity_labels = None

        if feedback_sampling_configuration is not None:
            train_box_features,\
                train_species_labels,\
                train_activity_labels,\
                feedback_box_features,\
                feedback_species_labels,\
                feedback_activity_labels =\
                    feedback_sampling_configuration.configure_data(
                        train_box_features,
                        train_species_labels,
                        train_activity_labels,
                        feedback_box_features,
                        feedback_species_labels,
                        feedback_activity_labels
                    )

        # Train
        progress = tqdm(
            range(self._max_epochs),
            desc=gen_tqdm_description(
                'Training classifiers...',
                train_loss=mean_train_loss,
                train_accuracy=mean_train_accuracy,
                val_accuracy=mean_val_accuracy
            ),
            total=self._max_epochs
        )
        pre_ewc_path = '.temp/pre_EWC_Logit_Layers_path.pth'
        
        self.ewc_calculation = EWC_Logit_Layers(species_classifier, activity_classifier, train_box_features, train_species_labels,train_activity_labels, self._class_frequencies , self._loss_fn, self._label_smoothing, device, pre_ewc_path)

        for epoch in progress:
            if self._patience is not None and\
                    epochs_since_improvement >= self._patience:
                # We haven't improved in several epochs. Time to stop
                # training.
                break

            # Train for one full epoch
            mean_train_loss, mean_train_accuracy = self._train_epoch(
                train_box_features,
                train_species_labels,
                train_activity_labels,
                feedback_box_features,
                feedback_species_labels,
                feedback_activity_labels,
                species_classifier,
                activity_classifier,
                optimizer,
                device,
                feedback_class_frequencies
            )

            if root_log_dir is not None:
                training_loss_curve[epoch] = mean_train_loss
                training_accuracy_curve[epoch] = mean_train_accuracy
                log_dir = get_log_dir()
                os.makedirs(log_dir, exist_ok=True)
                training_log = os.path.join(log_dir, 'training.pkl')
                
                with open(training_log, 'wb') as f:
                    sd = {}
                    sd['training_loss_curve'] = training_loss_curve
                    sd['training_accuracy_curve'] = training_accuracy_curve
                    pkl.dump(sd, f)

            # Measure validation accuracy for early stopping / model selection.
            if epoch >= self._min_epochs - 1:
                mean_val_accuracy = self._val_epoch(
                    val_box_features,
                    val_species_labels,
                    val_activity_labels,
                    species_classifier,
                    activity_classifier
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if root_log_dir is not None:
                    validation_accuracy_curve[epoch] = mean_val_accuracy
                    log_dir = get_log_dir()
                    os.makedirs(log_dir, exist_ok=True)
                    validation_log = os.path.join(log_dir, 'validation.pkl')

                    with open(validation_log, 'wb') as f:
                        pkl.dump(validation_accuracy_curve, f)

            progress.set_description(
                gen_tqdm_description(
                    'Training classifiers...',
                    train_loss=mean_train_loss,
                    train_accuracy=mean_train_accuracy,
                    val_accuracy=mean_val_accuracy
                )
            )

        progress.close()

        # Load the best-accuracy state dicts
        # NOTE To save GPU memory, we could temporarily move the models to the
        # CPU before copying or loading their state dicts.
        species_classifier.load_state_dict(
            best_accuracy_species_classifier_state_dict
        )
        activity_classifier.load_state_dict(
            best_accuracy_activity_classifier_state_dict
        )

    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        # classifier.reset()
        pass


    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        pass


class LogitLayerClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_feature_file,
            val_feature_file,
            box_transform,
            post_cache_train_transform,
            feedback_batch_size=32,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            feedback_loss_weight=0.5,
            loss_fn=LossFnEnum.cross_entropy,
            class_frequencies=None):
        self._lr = lr
        self._train_feature_file = train_feature_file
        self._val_feature_file = val_feature_file
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._feedback_batch_size = feedback_batch_size
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._feedback_loss_weight = feedback_loss_weight
        self._loss_fn = loss_fn
        self._class_frequencies = class_frequencies

    def _train_epoch(
            self,
            box_features,
            species_labels,
            activity_labels,
            feedback_box_features,
            feedback_species_labels,
            feedback_activity_labels,
            species_classifier,
            activity_classifier,
            optimizer,
            device,
            feedback_class_frequencies):
        # Set everything to train mode
        species_classifier.train()
        activity_classifier.train()

        # Compute class weights
        species_weights = None
        activity_weights = None
        species_frequencies = None
        activity_frequencies = None
        if self._class_frequencies is not None:
            train_species_frequencies,\
                train_activity_frequencies = self._class_frequencies
            train_species_frequencies = train_species_frequencies.to(device)
            train_activity_frequencies = train_activity_frequencies.to(device)
            
            species_frequencies = train_species_frequencies
            activity_frequencies = train_activity_frequencies

        if feedback_class_frequencies is not None:
            feedback_species_frequencies,\
                feedback_activity_frequencies = feedback_class_frequencies
            feedback_species_frequencies =\
                feedback_species_frequencies.to(device)
            feedback_activity_frequencies =\
                feedback_activity_frequencies.to(device)

            if species_frequencies is None:
                species_frequencies = feedback_species_frequencies
                activity_frequencies = feedback_activity_frequencies
            else:
                species_frequencies =\
                    species_frequencies + feedback_species_frequencies
                activity_frequencies =\
                    activity_frequencies + feedback_activity_frequencies

        if species_frequencies is not None:
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

        ## Non-feedback loss
        non_feedback_loss = 0
        non_feedback_n_species_correct = 0
        non_feedback_n_activity_correct = 0
        non_feedback_n_examples = 0
        if box_features is not None and\
                species_labels is not None and\
                activity_labels is not None:
            # Compute logits by passing the features through the appropriate
            # classifiers
            species_preds = species_classifier(box_features)
            activity_preds = activity_classifier(box_features)

            # Logging metrics
            species_correct = torch.argmax(species_preds, dim=1) == \
                species_labels
            non_feedback_n_species_correct = int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(activity_preds, dim=1) == \
                activity_labels
            non_feedback_n_activity_correct = int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )

            non_feedback_n_examples = species_labels.shape[0]

            if self._loss_fn == LossFnEnum.cross_entropy:
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
            else:
                focal_loss = torch.hub.load(
                    'adeelh/pytorch-multi-class-focal-loss',
                    model='FocalLoss',
                    alpha=None,
                    gamma=2,
                    reduction='none',
                    force_reload=False
                )
                if species_weights is not None:
                    ex_species_weights = species_weights[species_labels]
                else:
                    ex_species_weights = 1
                species_loss_all = focal_loss(
                    species_preds,
                    species_labels
                )
                species_loss =\
                    (species_loss_all * ex_species_weights).mean()
                
                if activity_weights is not None:
                    ex_activity_weights = activity_weights[activity_labels]
                else:
                    ex_activity_weights = 1
                activity_loss_all = focal_loss(
                    activity_preds,
                    activity_labels
                )
                activity_loss =\
                    (activity_loss_all * ex_activity_weights).mean()

            non_feedback_loss = species_loss + activity_loss

        ## Feedback loss

        # Flatten feedback box features, compute predictions, and re-split
        # per-image
        feedback_loss = 0
        feedback_n_species_correct = 0
        feedback_n_activity_correct = 0
        feedback_n_examples = 0
        if feedback_box_features is not None and\
                feedback_species_labels is not None and\
                feedback_activity_labels is not None:
            flattened_feedback_box_features = torch.cat(
                feedback_box_features,
                dim=0
            )
            flattened_feedback_species_logits = species_classifier(
                flattened_feedback_box_features
            )
            flattened_feedback_activity_logits = activity_classifier(
                flattened_feedback_box_features
            )
            flattened_feedback_species_preds = torch.nn.functional.softmax(
                flattened_feedback_species_logits,
                dim=1
            )
            flattened_feedback_activity_preds = torch.nn.functional.softmax(
                flattened_feedback_activity_logits,
                dim=1
            )

            # If class balancing, scale the gradients according to class
            # weights
            if species_weights is not None:
                flattened_feedback_species_preds =\
                    _balance_box_prediction_grads(
                        flattened_feedback_species_preds,
                        species_weights
                    )
                flattened_feedback_activity_preds =\
                    _balance_box_prediction_grads(
                        flattened_feedback_activity_preds,
                        activity_weights
                    )

            feedback_box_counts = [len(x) for x in feedback_box_features]
            feedback_species_preds = torch.split(
                flattened_feedback_species_preds,
                feedback_box_counts,
                dim=0
            )
            feedback_activity_preds = torch.split(
                flattened_feedback_activity_preds,
                feedback_box_counts,
                dim=0
            )

            # Logging metrics
            feedback_box_counts_t = torch.tensor(
                feedback_box_counts,
                device=device,
                dtype=torch.long
            )
            single_box_mask = feedback_box_counts_t == 1
            single_box_indices = torch.arange(
                len(single_box_mask),
                dtype=torch.long,
                device=device
            )[single_box_mask]
            if len(single_box_indices) > 0:
                single_box_species_preds = torch.cat(
                    [
                        feedback_species_preds[i] for i in single_box_indices
                    ],
                    dim=0
                )
                single_box_activity_preds = torch.cat(
                    [
                        feedback_activity_preds[i] for i in single_box_indices
                    ],
                    dim=0
                )
                single_box_species_labels =\
                    feedback_species_labels[single_box_indices]
                single_box_activity_labels =\
                    feedback_activity_labels[single_box_indices]
                feedback_species_correct =\
                    torch.argmax(single_box_species_preds, dim=1) ==\
                        torch.argmax(single_box_species_labels, dim=1)
                feedback_activity_correct =\
                    torch.argmax(single_box_activity_preds, dim=1) ==\
                        torch.argmax(single_box_activity_labels.to(torch.long), dim=1)
                feedback_n_species_correct =\
                    feedback_species_correct.to(torch.int).sum()
                feedback_n_activity_correct =\
                    feedback_activity_correct.to(torch.int).sum()

                feedback_n_examples = len(single_box_species_labels)

            # Compute loss
            # We have image-level count feedback labels for species
            feedback_species_loss = multiple_instance_count_cross_entropy_dyn(
                feedback_species_preds,
                feedback_species_labels
            )
            # We have image-level presence feedback labels for activities
            feedback_activity_loss = multiple_instance_presence_cross_entropy_dyn(
                feedback_activity_preds,
                feedback_activity_labels
            )
            feedback_loss = feedback_species_loss + feedback_activity_loss

            # Compute loss as weighted average between feedback and non-feedback
            # losses

        loss = (1 - self._feedback_loss_weight) * non_feedback_loss +\
            self._feedback_loss_weight * feedback_loss
        n_species_correct =\
            non_feedback_n_species_correct + feedback_n_species_correct
        n_activity_correct =\
            non_feedback_n_activity_correct + feedback_n_activity_correct
        n_examples = non_feedback_n_examples + feedback_n_examples

        # Optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return loss.detach().cpu().item(), mean_accuracy
        
    # def _train_epoch(
    #         self,
    #         box_features,
    #         species_labels,
    #         activity_labels,
    #         feedback_box_features,
    #         feedback_species_labels,
    #         feedback_activity_labels,
    #         species_classifier,
    #         activity_classifier,
    #         optimizer,
    #         device,
    #         feedback_class_frequencies):
    #     # Set everything to train mode
    #     species_classifier.train()
    #     activity_classifier.train()

    #     # Compute class weights
    #     species_weights = None
    #     activity_weights = None
    #     species_frequencies = None
    #     activity_frequencies = None
    #     if self._class_frequencies is not None:
    #         train_species_frequencies,\
    #             train_activity_frequencies = self._class_frequencies
    #         train_species_frequencies = train_species_frequencies.to(device)
    #         train_activity_frequencies = train_activity_frequencies.to(device)
            
    #         species_frequencies = train_species_frequencies
    #         activity_frequencies = train_activity_frequencies

    #     if feedback_class_frequencies is not None:
    #         feedback_species_frequencies,\
    #             feedback_activity_frequencies = feedback_class_frequencies
    #         feedback_species_frequencies =\
    #             feedback_species_frequencies.to(device)
    #         feedback_activity_frequencies =\
    #             feedback_activity_frequencies.to(device)

    #         if species_frequencies is None:
    #             species_frequencies = feedback_species_frequencies
    #             activity_frequencies = feedback_activity_frequencies
    #         else:
    #             species_frequencies =\
    #                 species_frequencies + feedback_species_frequencies
    #             activity_frequencies =\
    #                 activity_frequencies + feedback_activity_frequencies

    #     if species_frequencies is not None:
    #         species_proportions = species_frequencies /\
    #             species_frequencies.sum()
    #         unnormalized_species_weights =\
    #             torch.pow(1.0 / species_proportions, 1.0 / 3.0)
    #         unnormalized_species_weights[species_proportions == 0.0] = 0.0
    #         proportional_species_sum =\
    #             (species_proportions * unnormalized_species_weights).sum()
    #         species_weights =\
    #             unnormalized_species_weights / proportional_species_sum

    #         activity_proportions = activity_frequencies /\
    #             activity_frequencies.sum()
    #         unnormalized_activity_weights =\
    #             torch.pow(1.0 / activity_proportions, 1.0 / 3.0)
    #         unnormalized_activity_weights[activity_proportions == 0.0] = 0.0
    #         proportional_activity_sum =\
    #             (activity_proportions * unnormalized_activity_weights).sum()
    #         activity_weights =\
    #             unnormalized_activity_weights / proportional_activity_sum

    #     ## Non-feedback loss
    #     non_feedback_loss = 0
    #     non_feedback_n_species_correct = 0
    #     non_feedback_n_activity_correct = 0
    #     non_feedback_n_examples = 0
    #     n_examples = 0
    #     all_loss =0
    #     n_species_correct = 0
    #     n_activity_correct = 0

    #     feedback_loss = 0
    #     feedback_n_species_correct = 0
    #     feedback_n_activity_correct = 0
    #     feedback_n_examples = 0

    #     if box_features is not None and\
    #     species_labels is not None and\
    #     activity_labels is not None:

    #         non_feedback_dataset = NonFeedbackDataset(box_features, species_labels, activity_labels)
    #         feedback_dataset = FeedbackDatasetBatched(feedback_box_features, feedback_species_labels, feedback_activity_labels, batch_size = self._feedback_batch_size)

    #         non_feedback_loader = DataLoader(non_feedback_dataset, batch_size=self._feedback_batch_size, shuffle=True, num_workers=0)
    #         feedback_loader = DataLoader(feedback_dataset, batch_size= 1, shuffle=True, num_workers=0)
    #         feedback_iterator = itertools.cycle(feedback_loader)
            
    #         # Training loop
    #         for non_feedback_data in tqdm(non_feedback_loader, desc='Training Progress'):
    #             non_feedback_features, non_feedback_species, non_feedback_activity = non_feedback_data

    #             species_preds = species_classifier(non_feedback_features)
    #             activity_preds = activity_classifier(non_feedback_features)

    #             # Logging metrics
    #             species_correct = torch.argmax(species_preds, dim=1) == \
    #                 non_feedback_species
    #             non_feedback_n_species_correct += int(
    #                 species_correct.to(torch.int).sum().detach().cpu().item()
    #             )

    #             activity_correct = torch.argmax(activity_preds, dim=1) == \
    #                 non_feedback_activity
    #             non_feedback_n_activity_correct += int(
    #                 activity_correct.to(torch.int).sum().detach().cpu().item()
    #             )

    #             non_feedback_n_examples = non_feedback_species.shape[0]

    #             if self._loss_fn == LossFnEnum.cross_entropy:
    #                 species_loss = torch.nn.functional.cross_entropy(
    #                     species_preds,
    #                     non_feedback_species,
    #                     weight=species_weights,
    #                     label_smoothing=self._label_smoothing
    #                 )
    #                 activity_loss = torch.nn.functional.cross_entropy(
    #                     activity_preds,
    #                     non_feedback_activity,
    #                     weight=activity_weights,
    #                     label_smoothing=self._label_smoothing
    #                 )
    #             else:
    #                 focal_loss = torch.hub.load(
    #                     'adeelh/pytorch-multi-class-focal-loss',
    #                     model='FocalLoss',
    #                     alpha=torch.tensor([.75, .25]),
    #                     gamma=2,
    #                     reduction='none',
    #                     force_reload=False
    #                 )

    #                 ex_species_weights = species_weights[non_feedback_species]
    #                 species_loss_all = focal_loss(
    #                     species_preds,
    #                     non_feedback_species
    #                 )
    #                 species_loss =\
    #                     (species_loss_all * ex_species_weights).mean()
                    
    #                 ex_activity_weights = activity_weights[non_feedback_activity]
    #                 activity_loss_all = focal_loss(
    #                     activity_preds,
    #                     non_feedback_activity
    #                 )
    #                 activity_loss =\
    #                     (activity_loss_all * ex_activity_weights).mean()

    #             non_feedback_loss = species_loss + activity_loss
    #             # Get next feedback batch, restart automatically if at the end
    #             try:
    #                 feedback_data = next(feedback_iterator)
    #                 feedback_features, feedback_species, feedback_activity = feedback_data
    #                 feedback_features = [torch.squeeze(f, dim=0) for f in feedback_features]
    #                 feedback_species =  torch.squeeze(feedback_species, dim=0) 
    #                 feedback_activity = torch.squeeze(feedback_activity, dim=0)
    #             except: 
    #                 print("Feedback batches iterator is empty. Iterator restarted")
    #                 feedback_data = next(feedback_iterator)
    #                 feedback_features, feedback_species, feedback_activity = feedback_data
    #                 feedback_features = [torch.squeeze(f, dim=0) for f in feedback_features]
    #                 feedback_species =  torch.squeeze(feedback_species, dim=0) 
    #                 feedback_activity = torch.squeeze(feedback_activity, dim=0)
             

    #             flattened_feedback_box_features = torch.cat(
    #                 feedback_features,
    #                 dim=0
    #             )
    #             flattened_feedback_species_logits = species_classifier(
    #                 flattened_feedback_box_features
    #             )
    #             flattened_feedback_activity_logits = activity_classifier(
    #                 flattened_feedback_box_features
    #             )
    #             flattened_feedback_species_preds = torch.nn.functional.softmax(
    #                 flattened_feedback_species_logits,
    #                 dim=1
    #             )
    #             flattened_feedback_activity_preds = torch.nn.functional.softmax(
    #                 flattened_feedback_activity_logits,
    #                 dim=1
    #             )

    #             # If class balancing, scale the gradients according to class
    #             # weights
    #             if species_weights is not None:
    #                 flattened_feedback_species_preds =\
    #                     _balance_box_prediction_grads(
    #                         flattened_feedback_species_preds,
    #                         species_weights
    #                     )
    #                 flattened_feedback_activity_preds =\
    #                     _balance_box_prediction_grads(
    #                         flattened_feedback_activity_preds,
    #                         activity_weights
    #                     )
    #             feedback_box_counts = [len(x) for x in feedback_features]
    #             feedback_species_preds = torch.split(
    #                 flattened_feedback_species_preds,
    #                 feedback_box_counts,
    #                 dim=0
    #             )
    #             feedback_activity_preds = torch.split(
    #                 flattened_feedback_activity_preds,
    #                 feedback_box_counts,
    #                 dim=0
    #             )

    #             # Logging metrics
    #             feedback_box_counts_t = torch.tensor(
    #                 feedback_box_counts,
    #                 device=device,
    #                 dtype=torch.long
    #             )
    #             single_box_mask = feedback_box_counts_t == 1
    #             single_box_indices = torch.arange(
    #                 len(single_box_mask),
    #                 dtype=torch.long,
    #                 device=device
    #             )[single_box_mask]
    #             if len(single_box_indices) > 0:
    #                 single_box_species_preds = torch.cat(
    #                     [
    #                         feedback_species_preds[i] for i in single_box_indices
    #                     ],
    #                     dim=0
    #                 )
    #                 single_box_activity_preds = torch.cat(
    #                     [
    #                         feedback_activity_preds[i] for i in single_box_indices
    #                     ],
    #                     dim=0
    #                 )
    #                 single_box_species_labels =\
    #                     feedback_species[single_box_indices]
    #                 single_box_activity_labels =\
    #                     feedback_activity[single_box_indices]
    #                 feedback_species_correct =\
    #                     torch.argmax(single_box_species_preds, dim=1) ==\
    #                         torch.argmax(single_box_species_labels, dim=1)
    #                 feedback_activity_correct =\
    #                     torch.argmax(single_box_activity_preds, dim=1) ==\
    #                         torch.argmax(single_box_activity_labels.to(torch.long), dim=1)
    #                 feedback_n_species_correct =\
    #                     feedback_species_correct.to(torch.int).sum()
    #                 feedback_n_activity_correct =\
    #                     feedback_activity_correct.to(torch.int).sum()

    #                 feedback_n_examples = len(single_box_species_labels)

    #             # Compute loss
    #             # We have image-level count feedback labels for species
    #             feedback_species_loss = multiple_instance_count_cross_entropy_dyn(
    #                 feedback_species_preds,
    #                 feedback_species
    #             )
    #             # We have image-level presence feedback labels for activities
    #             feedback_activity_loss = multiple_instance_presence_cross_entropy_dyn(
    #                 feedback_activity_preds,
    #                 feedback_activity
    #             )
    #             feedback_loss = feedback_species_loss + feedback_activity_loss
            

    #             loss = (1 - self._feedback_loss_weight) * non_feedback_loss +\
    #                 self._feedback_loss_weight * feedback_loss
               
    #             # Gradient and optimizer steps
    #             optimizer.zero_grad()
    #             loss.backward()
    #             for param in species_classifier.parameters():
    #                 if param.grad is not None and \
    #                 (torch.any(torch.isnan(param.grad.data)) or \
    #                     torch.any(torch.isinf(param.grad.data))):
    #                     # Found some NaNs in the gradients. "Skip" this batch by
    #                     # zeroing the gradients. This should work in distributed
    #                     # mode as well
    #                     print('Some NaNs in the gradients of species_classifier. Skiping this batch ...')
    #                     optimizer.zero_grad()
    #                     break
                        
    #             for param in activity_classifier.parameters():
    #                 if param.grad is not None and \
    #                 (torch.any(torch.isnan(param.grad.data)) or \
    #                     torch.any(torch.isinf(param.grad.data))):
    #                     # Found some NaNs in the gradients. "Skip" this batch by
    #                     # zeroing the gradients. This should work in distributed
    #                     # mode as well
    #                     print('Some NaNs in the gradients of activity_classifier. Skiping this batch ...')
    #                     optimizer.zero_grad()
    #                     break
                
    #             optimizer.step()

    #             n_species_correct +=\
    #                 non_feedback_n_species_correct + feedback_n_species_correct
    #             n_activity_correct +=\
    #                 non_feedback_n_activity_correct + feedback_n_activity_correct
    #             n_examples += non_feedback_n_examples + feedback_n_examples
    #             all_loss+=loss


    #     mean_species_accuracy = float(n_species_correct) / n_examples
    #     mean_activity_accuracy = float(n_activity_correct) / n_examples

    #     mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

    #     return all_loss.detach().cpu().item()/ n_examples, mean_accuracy

    def _val_epoch(
            self,
            box_features,
            species_labels,
            activity_labels,
            species_classifier,
            activity_classifier):
        with torch.no_grad():
            species_classifier.eval()
            activity_classifier.eval()

            species_preds = species_classifier(box_features)
            activity_preds = activity_classifier(box_features)

            species_correct = torch.argmax(species_preds, dim=1) == \
                species_labels
            n_species_correct = int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(activity_preds, dim=1) == \
                activity_labels
            n_activity_correct = int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )

            n_examples = species_labels.shape[0]
            mean_species_accuracy = float(n_species_correct) / n_examples
            mean_activity_accuracy = float(n_activity_correct) / n_examples

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    '''
    Params:
        species_classifier: ClassifierV2
            Species classifier to train
        activity_classifier: ClassifierV2
            Activity classifier to train
        root_log_dir: str
            Root directory for logging training. Should include named transform
            paths, if appropriate.
    '''
    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            self._lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Define convergence parameters (early stopping + model selection)
        epochs_since_improvement = 0
        best_accuracy = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None
        mean_train_loss = None
        mean_train_accuracy = None
        mean_val_accuracy = None

        # If we didn't load an optimizer state dict, and so the scheduler
        # hasn't been constructed yet, then construct it
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}

        def get_log_dir():
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'logit-layer-classifier-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )
        


        train_box_features, train_species_labels, train_activity_labels =\
            torch.load(
                self._train_feature_file,
                map_location=device
            )
        val_box_features, val_species_labels, val_activity_labels =\
            torch.load(
                self._val_feature_file,
                map_location=device
            )

        if feedback_dataset is not None:
            feedback_box_counts = [
                feedback_dataset.box_count(i) for\
                    i in range(len(feedback_dataset))
            ]
            feedback_batch_sampler = DistributedRandomBoxImageBatchSampler(feedback_box_counts, self._feedback_batch_size, 1, 0)
            feedback_loader = DataLoader(
                feedback_dataset,
                num_workers=0,
                batch_sampler=feedback_batch_sampler,
                collate_fn=gen_custom_collate()
            )
            # Precompute feedback backbone features
            backbone.eval()
            feedback_box_features = []
            feedback_species_labels = []
            feedback_activity_labels = []
            with torch.no_grad():
                for batch_species_labels,\
                        batch_activity_labels,\
                        _,\
                        batch_box_images,\
                        _ in feedback_loader:
                    # Move batch to device
                    batch_species_labels =\
                        batch_species_labels.to(device)
                    batch_activity_labels =\
                        batch_activity_labels.to(device)
                    batch_box_images =\
                        [b.to(device) for b in batch_box_images]

                    # Get list of per-image box counts
                    batch_box_counts = [len(x) for x in batch_box_images]

                    # Flatten box images and compute features
                    flattened_box_images = torch.cat(batch_box_images, dim=0)
                    batch_box_features = backbone(flattened_box_images)

                    # Use batch_box_counts to split the computed features back
                    # to per-image feature tensors and concatenate to
                    # feedback_box_features
                    feedback_box_features += torch.split(
                        batch_box_features,
                        batch_box_counts,
                        dim=0
                    )

                    # Record labels
                    feedback_species_labels.append(batch_species_labels)
                    feedback_activity_labels.append(batch_activity_labels)
                    

                # Concatenate labels
                feedback_species_labels = torch.cat(
                    feedback_species_labels,
                    dim=0
                )
                feedback_activity_labels = torch.cat(
                    feedback_activity_labels,
                    dim=0
                )
        else:
            feedback_box_features = None
            feedback_species_labels = None
            feedback_activity_labels = None

        if feedback_sampling_configuration is not None:
            train_box_features,\
                train_species_labels,\
                train_activity_labels,\
                feedback_box_features,\
                feedback_species_labels,\
                feedback_activity_labels =\
                    feedback_sampling_configuration.configure_data(
                        train_box_features,
                        train_species_labels,
                        train_activity_labels,
                        feedback_box_features,
                        feedback_species_labels,
                        feedback_activity_labels
                    )

        # Train
        progress = tqdm(
            range(self._max_epochs),
            desc=gen_tqdm_description(
                'Training classifiers...',
                train_loss=mean_train_loss,
                train_accuracy=mean_train_accuracy,
                val_accuracy=mean_val_accuracy
            ),
            total=self._max_epochs
        )
        mean_val_accuracy = 0
        for epoch in progress:
            if self._patience is not None and\
                    epochs_since_improvement >= self._patience:
                # We haven't improved in several epochs. Time to stop
                # training.
                break

            # Train for one full epoch
            mean_train_loss, mean_train_accuracy = self._train_epoch(
                train_box_features,
                train_species_labels,
                train_activity_labels,
                feedback_box_features,
                feedback_species_labels,
                feedback_activity_labels,
                species_classifier,
                activity_classifier,
                optimizer,
                device,
                feedback_class_frequencies
            )

            if root_log_dir is not None:
                training_loss_curve[epoch] = mean_train_loss
                training_accuracy_curve[epoch] = mean_train_accuracy
                log_dir = get_log_dir()
                os.makedirs(log_dir, exist_ok=True)
                training_log = os.path.join(log_dir, 'training.pkl')
                
                with open(training_log, 'wb') as f:
                    sd = {}
                    sd['training_loss_curve'] = training_loss_curve
                    sd['training_accuracy_curve'] = training_accuracy_curve
                    pkl.dump(sd, f)

            # Measure validation accuracy for early stopping / model selection.
            if epoch >= self._min_epochs - 1:
                mean_val_accuracy = self._val_epoch(
                    val_box_features,
                    val_species_labels,
                    val_activity_labels,
                    species_classifier,
                    activity_classifier
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if root_log_dir is not None:
                    validation_accuracy_curve[epoch] = mean_val_accuracy
                    log_dir = get_log_dir()
                    os.makedirs(log_dir, exist_ok=True)
                    validation_log = os.path.join(log_dir, 'validation.pkl')

                    with open(validation_log, 'wb') as f:
                        pkl.dump(validation_accuracy_curve, f)

            progress.set_description(
                gen_tqdm_description(
                    'Training classifiers...',
                    train_loss=mean_train_loss,
                    train_accuracy=mean_train_accuracy,
                    val_accuracy=mean_val_accuracy
                )
            )

        progress.close()

        # Load the best-accuracy state dicts
        # NOTE To save GPU memory, we could temporarily move the models to the
        # CPU before copying or loading their state dicts.
        species_classifier.load_state_dict(
            best_accuracy_species_classifier_state_dict
        )
        activity_classifier.load_state_dict(
            best_accuracy_activity_classifier_state_dict
        )

    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        classifier.reset()
        # pass


    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        pass
    
class MultiLoader:
    class MultiLoaderIter:
        def __init__(self, loaders, primary_loader):
            self._loaders = loaders
            self._primary_loader = primary_loader
            self._iters = [
                iter(loader) if loader is not None else None for\
                    loader in self._loaders
            ]

        def __next__(self):
            data = []
            for idx in range(len(self._iters)):
                itr = self._iters[idx]
                if itr is not None:
                    try:
                        data_tuple = next(itr)
                    except:
                        if idx == self._primary_loader:
                            raise StopIteration

                        loader = self._loaders[idx]
                        self._iters[idx] = iter(loader)
                        itr = self._iters[idx]
                        data_tuple = next(itr)
                else:
                    data_tuple = None

                data.append(data_tuple)

            return tuple(data)

    def __init__(self, loaders):
        self._loaders = loaders
        lengths = [
            len(loader) if loader is not None else 0 for\
                loader in self._loaders
        ]
        max_length = max(lengths)
        self._primary_loader = lengths.index(max_length)

    def __iter__(self):
        return MultiLoader.MultiLoaderIter(self._loaders, self._primary_loader)

    def __len__(self):
        return len(self._loaders[self._primary_loader])


class EndToEndClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_dataset,
            val_known_dataset,
            box_transform,
            post_cache_train_transform,
            retraining_batch_size=32,
            root_checkpoint_dir=None,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            feedback_loss_weight=0.5,
            scheduler_type=SchedulerType.none,
            loss_fn=LossFnEnum.cross_entropy,
            class_frequencies=None,
            memory_cache=True,
            load_best_after_training=True,
            val_reduce_fn=None):
        self._lr = lr
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._retraining_batch_size = retraining_batch_size
        self._root_checkpoint_dir = root_checkpoint_dir
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._feedback_loss_weight = feedback_loss_weight
        self._scheduler_type = scheduler_type
        self._loss_fn = loss_fn
        self._class_frequencies = class_frequencies
        self._memory_cache = memory_cache
        self._load_best_after_training = load_best_after_training
        self._val_reduce_fn = val_reduce_fn

    def _train_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer,
            species_labels,
            activity_labels,
            box_images,
            feedback_species_labels,
            feedback_activity_labels,
            feedback_box_images,
            device,
            allow_print,
            feedback_class_frequencies):
        # Compute class weights
        species_weights = None
        activity_weights = None
        species_frequencies = None
        activity_frequencies = None
        if self._class_frequencies is not None:
            train_species_frequencies,\
                train_activity_frequencies = self._class_frequencies
            train_species_frequencies = train_species_frequencies.to(device)
            train_activity_frequencies = train_activity_frequencies.to(device)
            
            species_frequencies = train_species_frequencies
            activity_frequencies = train_activity_frequencies

        if feedback_class_frequencies is not None:
            feedback_species_frequencies,\
                feedback_activity_frequencies = feedback_class_frequencies
            feedback_species_frequencies =\
                feedback_species_frequencies.to(device)
            feedback_activity_frequencies =\
                feedback_activity_frequencies.to(device)

            if species_frequencies is None:
                species_frequencies = feedback_species_frequencies
                activity_frequencies = feedback_activity_frequencies
            else:
                species_frequencies =\
                    species_frequencies + feedback_species_frequencies
                activity_frequencies =\
                    activity_frequencies + feedback_activity_frequencies

        if species_frequencies is not None:
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

        # Compute losses

        # Non-feedback loss
        non_feedback_loss = 0
        non_feedback_n_species_correct = 0
        non_feedback_n_activity_correct = 0
        non_feedback_n_examples = 0
        if box_images is not None and\
                species_labels is not None and\
                activity_labels is not None and\
                self._feedback_loss_weight != 1:
            # Move to device
            species_labels = species_labels.to(device)
            activity_labels = activity_labels.to(device)
            box_images = box_images.to(device)
            if allow_print:
                print_nan(box_images, 'box_images')

            # Extract box features
            box_features = backbone(box_images)
            if allow_print:
                print_nan(box_features, 'box_features')

            # Compute logits by passing the features through the appropriate
            # classifiers
            species_preds = species_classifier(box_features)
            activity_preds = activity_classifier(box_features)

            # Logging metrics
            species_correct = torch.argmax(species_preds, dim=1) == \
                species_labels
            non_feedback_n_species_correct = int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(activity_preds, dim=1) == \
                activity_labels
            non_feedback_n_activity_correct = int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )

            non_feedback_n_examples = len(species_labels)

            # Losses
            if self._loss_fn == LossFnEnum.cross_entropy:
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
            else:
                focal_loss = torch.hub.load(
                    'adeelh/pytorch-multi-class-focal-loss',
                    model='FocalLoss',
                    alpha=None,
                    gamma=2,
                    reduction='none',
                    force_reload=False
                )
                if species_weights is not None:
                    ex_species_weights = species_weights[species_labels]
                else:
                    ex_species_weights = 1
                species_loss_all = focal_loss(
                    species_preds,
                    species_labels
                )
                species_loss =\
                    (species_loss_all * ex_species_weights).mean()
                
                if activity_weights is not None:
                    ex_activity_weights = activity_weights[activity_labels]
                else:
                    ex_activity_weights = 1
                activity_loss_all = focal_loss(
                    activity_preds,
                    activity_labels
                )
                activity_loss =\
                    (activity_loss_all * ex_activity_weights).mean()

            non_feedback_loss = species_loss + activity_loss
            if allow_print:
                print_nan(non_feedback_loss, 'non_feedback_loss')

        feedback_loss = 0
        feedback_n_species_correct = 0
        feedback_n_activity_correct = 0
        feedback_n_examples = 0
        if feedback_box_images is not None:
            # Move to device
            feedback_species_labels = feedback_species_labels.to(device)
            feedback_activity_labels = feedback_activity_labels.to(device)
            feedback_box_images = [x.to(device) for x in feedback_box_images]

            # Get counts and flatten box images
            feedback_box_counts = [len(x) for x in feedback_box_images]
            feedback_box_images = torch.cat(feedback_box_images, dim=0)
            print_nan(feedback_box_images, 'feedback_box_images')

            # Compute features and logits
            feedback_box_features = backbone(feedback_box_images)
            if allow_print:
                print_nan(feedback_box_features, 'feedback_box_features')

            feedback_species_logits =\
                species_classifier(feedback_box_features)
            feedback_activity_logits =\
                activity_classifier(feedback_box_features)
            if allow_print:
                print_nan(feedback_activity_logits, 'feedback_activity_logits')

            feedback_species_preds =\
                torch.nn.functional.softmax(feedback_species_logits, dim=1)
            feedback_activity_preds =\
                torch.nn.functional.softmax(feedback_activity_logits, dim=1)

            # If class balancing, scale the gradients according
            # to class weights
            if species_weights is not None:
                feedback_species_preds = _balance_box_prediction_grads(
                    feedback_species_preds,
                    species_weights
                )
                feedback_activity_preds = _balance_box_prediction_grads(
                    feedback_activity_preds,
                    activity_weights
                )

            if allow_print:
                print_nan(feedback_activity_preds, 'feedback_activity_preds')
            
            # Re-split feedback predictions for loss computation
            feedback_species_preds = torch.split(
                feedback_species_preds,
                feedback_box_counts,
                dim=0
            )
            feedback_activity_preds = torch.split(
                feedback_activity_preds,
                feedback_box_counts,
                dim=0
            )

            # Logging metrics
            feedback_box_counts_t = torch.tensor(
                feedback_box_counts,
                device=device,
                dtype=torch.long
            )
            single_box_mask = feedback_box_counts_t == 1
            single_box_indices = torch.arange(
                len(single_box_mask),
                dtype=torch.long,
                device=device
            )[single_box_mask]
            if len(single_box_indices) > 0:
                single_box_species_preds = torch.cat(
                    [
                        feedback_species_preds[i] for i in single_box_indices
                    ],
                    dim=0
                ) 
                single_box_activity_preds = torch.cat(
                    [
                        feedback_activity_preds[i] for i in single_box_indices
                    ],
                    dim=0
                )
                single_box_species_labels =\
                    feedback_species_labels[single_box_indices]
                single_box_activity_labels =\
                    feedback_activity_labels[single_box_indices]
                feedback_species_correct =\
                    torch.argmax(single_box_species_preds, dim=1) ==\
                        torch.argmax(single_box_species_labels, dim=1)
                feedback_activity_correct =\
                    torch.argmax(single_box_activity_preds, dim=1) ==\
                        torch.argmax(single_box_activity_labels.to(torch.long), dim=1)
                feedback_n_species_correct =\
                    feedback_species_correct.to(torch.int).sum()
                feedback_n_activity_correct =\
                    feedback_activity_correct.to(torch.int).sum()

                feedback_n_examples = len(single_box_species_labels)

            # We have image-level count feedback labels for species
            feedback_species_loss =\
                multiple_instance_count_cross_entropy_dyn(
                    feedback_species_preds,
                    feedback_species_labels,
                    allow_print=allow_print
                )
            if allow_print:
                print_nan(feedback_species_loss, "feedback_species_loss")

            # We have image-level presence feedback labels for activities
            feedback_activity_loss =\
                multiple_instance_presence_cross_entropy_dyn(
                    feedback_activity_preds,
                    feedback_activity_labels
                )
            if allow_print:
                print_nan(feedback_activity_loss, "feedback_activity_loss")

            feedback_loss = feedback_species_loss + feedback_activity_loss
            if allow_print:
                print_nan(feedback_loss, "feedback_loss")

        loss = (1 - self._feedback_loss_weight) * non_feedback_loss +\
            self._feedback_loss_weight * feedback_loss
        n_species_correct =\
            non_feedback_n_species_correct + feedback_n_species_correct
        n_activity_correct =\
            non_feedback_n_activity_correct + feedback_n_activity_correct
        n_examples = non_feedback_n_examples + feedback_n_examples

        optimizer.zero_grad()
        if allow_print:
            print_nan(loss, 'loss 3')
            for param in backbone.parameters():
                print_nan(param, 'some parameter 1')

            for param in backbone.parameters():
                if param.grad is not None:
                    print_nan(param.grad.data, 'some parameter\'s gradient 1')

        loss.backward()

        for param in species_classifier.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of species_classifier. Skiping this batch ...')
                optimizer.zero_grad()
                break
                
        for param in activity_classifier.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of activity_classifier. Skiping this batch ...')
                optimizer.zero_grad()
                break
                
        for param in backbone.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of backbone. Skiping this batch ...')
                optimizer.zero_grad()
                break

        optimizer.step()

        if allow_print:
            for param in backbone.parameters():
                print_nan(param, 'some parameter 2')
            for param in backbone.parameters():
                if param.grad is not None:
                    print_nan(param.grad.data, 'some parameter\'s gradient 2')

        return loss.detach().cpu().item(),\
            n_species_correct,\
            n_activity_correct,\
            n_examples

    def _train_epoch(
            self,
            backbone,
            train_loader,
            feedback_loader,
            species_classifier,
            activity_classifier,
            optimizer,
            device,
            allow_print,
            feedback_class_frequencies):
        # Set everything to train mode
        backbone.train()
        species_classifier.train()
        activity_classifier.train()
        
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_examples = 0
        n_species_correct = 0
        n_activity_correct = 0

        feedback_iter =\
            iter(feedback_loader) if feedback_loader is not None else None
        #length = sum(1 for _ in feedback_iter)
        #print('feedback_iter 1', length)  # Outputs: 5
        #print(feedback_iter)
        feedback_species_labels = None
        feedback_activity_labels = None
        feedback_box_images = None
        batch_loss = None

        multi_loader = MultiLoader((train_loader, feedback_loader))
        if allow_print:
            train_loader_progress = tqdm(
                multi_loader,
                desc=gen_tqdm_description(
                    'Training batch...',
                    batch_loss=batch_loss
                )
            )
        else:
            train_loader_progress = multi_loader
        # for species_labels, activity_labels, box_images in train_loader_progress:
        for train_batch, feedback_batch in train_loader_progress:
            if train_batch is not None:
                species_labels, activity_labels, box_images = train_batch
            else:
                species_labels = None
                activity_labels = None
                box_images = None

            if feedback_batch is not None:
                feedback_species_labels,\
                    feedback_activity_labels,\
                    _,\
                    feedback_box_images,\
                    _ = feedback_batch
            else:
                feedback_species_labels = None
                feedback_activity_labels = None
                feedback_box_images = None

            gc.collect()

            batch_loss, batch_n_species_correct, batch_n_activity_correct, batch_n_examples =\
                self._train_batch(
                    backbone,
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    species_labels,
                    activity_labels,
                    box_images,
                    feedback_species_labels,
                    feedback_activity_labels,
                    feedback_box_images,
                    device,
                    allow_print,
                    feedback_class_frequencies
                )

            sum_loss += batch_loss
            n_iterations += 1
            n_examples += batch_n_examples
            n_species_correct += batch_n_species_correct
            n_activity_correct += batch_n_activity_correct
            
            if allow_print:
                train_loader_progress.set_description(
                    gen_tqdm_description(
                        'Training batch...',
                        batch_loss=batch_loss
                    )
                )
            


        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, mean_accuracy

    def _val_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images,
            device):
        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = backbone(box_images)

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_correct = torch.argmax(species_preds, dim=1) == \
            species_labels
        n_species_correct = species_correct.to(torch.int).sum()

        activity_correct = torch.argmax(activity_preds, dim=1) == \
            activity_labels
        n_activity_correct = activity_correct.to(torch.int).sum()

        return n_species_correct, n_activity_correct

    def _val_epoch(
            self,
            backbone,
            data_loader,
            species_classifier,
            activity_classifier,
            device):
        with torch.no_grad():
            backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = torch.zeros(1, device=device)
            n_species_correct = torch.zeros(1, device=device)
            n_activity_correct = torch.zeros(1, device=device)

            for species_labels, activity_labels, box_images in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        backbone,
                        species_classifier,
                        activity_classifier,
                        species_labels,
                        activity_labels,
                        box_images,
                        device
                    )
                n_examples += box_images.shape[0]
                n_species_correct += batch_n_species_correct
                n_activity_correct += batch_n_activity_correct

            if self._val_reduce_fn is not None:
                self._val_reduce_fn(n_examples)
                self._val_reduce_fn(n_species_correct)
                self._val_reduce_fn(n_activity_correct)
            mean_species_accuracy = float(n_species_correct.detach().cpu().item()) / float(n_examples.detach().cpu().item())
            mean_activity_accuracy = float(n_activity_correct.detach().cpu().item()) / float(n_examples.detach().cpu().item())

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        train_dataset = self._train_dataset

        if feedback_sampling_configuration is not None:
            train_dataset, feedback_dataset =\
                feedback_sampling_configuration.configure_datasets(
                    train_dataset,
                    feedback_dataset
                )

        if train_dataset is not None:
            if self._memory_cache:
                train_dataset = FlattenedBoxImageDataset(BoxImageMemoryDataset(train_dataset))
            else:
                train_dataset = FlattenedBoxImageDataset(train_dataset)

            if train_sampler_fn is not None:
                train_sampler = train_sampler_fn(train_dataset)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self._retraining_batch_size,
                    num_workers=0,
                    sampler=train_sampler
                )
            else:
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self._retraining_batch_size,
                    shuffle=True,
                    num_workers=0
                )
        else:
            train_loader = None

        if feedback_dataset is not None:
            feedback_box_counts = [
                feedback_dataset.box_count(i) for\
                    i in range(len(feedback_dataset))
            ]
            if feedback_batch_sampler_fn is not None:
                feedback_batch_sampler = feedback_batch_sampler_fn(
                    feedback_box_counts
                )
            else:
                feedback_batch_sampler = DistributedRandomBoxImageBatchSampler(
                    feedback_box_counts,
                    self._retraining_batch_size,
                    1,
                    0
                )
            feedback_loader = DataLoader(
                feedback_dataset,
                num_workers=0,
                batch_sampler=feedback_batch_sampler,
                collate_fn=gen_custom_collate()
            )
        else:
            feedback_loader = None
        
        # Construct validation loaders for early stopping / model selection.
        # I'm assuming our model selection strategy will be based solely on the
        # validation classification accuracy and not based on novelty detection
        # capabilities in any way. Otherwise, we can use the novel validation
        # data to measure novelty detection performance. These currently aren't
        # being stored (except in a special form for the logistic regressions),
        # so we'd have to modify __init__().
        if self._memory_cache:
            val_dataset = FlattenedBoxImageDataset(BoxImageMemoryDataset(self._val_known_dataset))
        else:
            val_dataset = FlattenedBoxImageDataset(self._val_known_dataset)

        if train_sampler_fn is not None:
            val_sampler = train_sampler_fn(val_dataset)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._retraining_batch_size,
                num_workers=0,
                sampler=val_sampler
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._retraining_batch_size,
                shuffle=False,
                num_workers=0
            )

        # Retrain the backbone and classifiers
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(backbone.parameters())\
                + list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            self._lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Init scheduler to None. It will be constructed after loading
        # the optimizer state dict, or after failing to do so
        scheduler = None
        scheduler_ctor = self._scheduler_type.ctor()

        # Define convergence parameters (early stopping + model selection)
        start_epoch = 0
        epochs_since_improvement = 0
        best_accuracy = None
        best_accuracy_backbone_state_dict = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None
        mean_train_loss = None
        mean_train_accuracy = None
        mean_val_accuracy = None

        def get_checkpoint_dir():
            return os.path.join(
                self._root_checkpoint_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )

        if self._root_checkpoint_dir is not None:
            checkpoint_dir = get_checkpoint_dir()
            training_checkpoint = os.path.join(checkpoint_dir, 'training.pth')
            validation_checkpoint =\
                os.path.join(checkpoint_dir, 'validation.pth')
            
            if os.path.exists(training_checkpoint):
                sd = torch.load(
                    training_checkpoint,
                    map_location=device
                )
                backbone.load_state_dict(sd['backbone'])
                species_classifier.load_state_dict(sd['species_classifier'])
                activity_classifier.load_state_dict(sd['activity_classifier'])
                optimizer.load_state_dict(sd['optimizer'])
                start_epoch = sd['start_epoch']
                mean_train_loss = sd['mean_train_loss']
                mean_train_accuracy = sd['mean_train_accuracy']

                if scheduler_ctor is not None:
                    scheduler = scheduler_ctor(
                        optimizer,
                        self._max_epochs
                    )
                    scheduler.load_state_dict(sd['scheduler'])
            if os.path.exists(validation_checkpoint):
                sd = torch.load(
                    validation_checkpoint,
                    map_location=device
                )
                epochs_since_improvement = sd['epochs_since_improvement']
                best_accuracy = sd['accuracy']
                best_accuracy_backbone_state_dict = sd['backbone_state_dict']
                best_accuracy_species_classifier_state_dict =\
                    sd['species_classifier_state_dict']
                best_accuracy_activity_classifier_state_dict =\
                    sd['activity_classifier_state_dict']
                mean_val_accuracy = sd['mean_val_accuracy']
        
        # If we didn't load an optimizer state dict, and so the scheduler
        # hasn't been constructed yet, then construct it
        if scheduler_ctor is not None and scheduler is None:
            scheduler = scheduler_ctor(
                optimizer,
                self._max_epochs
            )
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}

        def get_log_dir():
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )

        if root_log_dir is not None and allow_write:
            log_dir = get_log_dir()
            training_log = os.path.join(log_dir, 'training.pkl')
            validation_log =\
                os.path.join(log_dir, 'validation.pkl')

            if os.path.exists(training_log):
                with open(training_log, 'rb') as f:
                    sd = pkl.load(f)
                    training_loss_curve = sd['training_loss_curve']
                    training_accuracy_curve = sd['training_accuracy_curve']

            if os.path.exists(validation_log):
                with open(validation_log, 'rb') as f:
                    validation_accuracy_curve = pkl.load(f)

        # Train
        if allow_print:
            print('lr:', self._lr)
            progress = tqdm(
                range(start_epoch, self._max_epochs),
                desc=gen_tqdm_description(
                    'Training backbone and classifiers...',
                    train_loss=mean_train_loss,
                    train_accuracy=mean_train_accuracy,
                    val_accuracy=mean_val_accuracy
                ),
                total=self._max_epochs,
                initial=start_epoch
            )
        else:
            progress = range(start_epoch, self._max_epochs)
        for epoch in progress:
            if self._patience is not None and\
                    epochs_since_improvement >= self._patience:
                # We haven't improved in several epochs. Time to stop
                # training.
                break

            if train_loader is not None and train_loader.sampler is not None:
                # Set the sampler epoch for shuffling when running in
                # distributed mode
                train_loader.sampler.set_epoch(epoch)

            # Train for one full epoch
            mean_train_loss, mean_train_accuracy = self._train_epoch(
                backbone,
                train_loader,
                feedback_loader,
                species_classifier,
                activity_classifier,
                optimizer,
                device,
                allow_print,
                feedback_class_frequencies
            )

            if self._root_checkpoint_dir is not None and allow_write:
                checkpoint_dir = get_checkpoint_dir()
                training_checkpoint =\
                    os.path.join(checkpoint_dir, 'training.pth')
                os.makedirs(checkpoint_dir, exist_ok=True)

                sd = {}
                sd['backbone'] = backbone.state_dict()
                sd['species_classifier'] = species_classifier.state_dict()
                sd['activity_classifier'] = activity_classifier.state_dict()
                sd['optimizer'] = optimizer.state_dict()
                sd['start_epoch'] = epoch + 1
                sd['mean_train_loss'] = mean_train_loss
                sd['mean_train_accuracy'] = mean_train_accuracy
                if scheduler is not None:
                    sd['scheduler'] = scheduler.state_dict()
                torch.save(sd, training_checkpoint)

            if root_log_dir is not None and allow_write:
                training_loss_curve[epoch] = mean_train_loss
                training_accuracy_curve[epoch] = mean_train_accuracy
                log_dir = get_log_dir()
                os.makedirs(log_dir, exist_ok=True)
                training_log = os.path.join(log_dir, 'training.pkl')

                with open(training_log, 'wb') as f:
                    sd = {}
                    sd['training_loss_curve'] = training_loss_curve
                    sd['training_accuracy_curve'] = training_accuracy_curve
                    pkl.dump(sd, f)

            # Measure validation accuracy for early stopping / model selection.
            if epoch >= self._min_epochs - 1 and epoch % 10 == 0:
                mean_val_accuracy = self._val_epoch(
                    backbone,
                    val_loader,
                    species_classifier,
                    activity_classifier,
                    device
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict =\
                        deepcopy(backbone.state_dict())
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if self._root_checkpoint_dir is not None and allow_write:
                    checkpoint_dir = get_checkpoint_dir()
                    validation_checkpoint =\
                        os.path.join(checkpoint_dir, 'validation.pth')
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    sd = {}
                    sd['epochs_since_improvement'] = epochs_since_improvement
                    sd['accuracy'] = best_accuracy
                    sd['backbone_state_dict'] =\
                        best_accuracy_backbone_state_dict
                    sd['species_classifier_state_dict'] =\
                        best_accuracy_species_classifier_state_dict
                    sd['activity_classifier_state_dict'] =\
                        best_accuracy_activity_classifier_state_dict
                    sd['mean_val_accuracy'] = mean_val_accuracy
                    torch.save(sd, validation_checkpoint)

                if root_log_dir is not None and allow_write:
                    validation_accuracy_curve[epoch] = mean_val_accuracy
                    log_dir = get_log_dir()
                    os.makedirs(log_dir, exist_ok=True)
                    validation_log = os.path.join(log_dir, 'validation.pkl')
                    
                    with open(validation_log, 'wb') as f:
                        pkl.dump(validation_accuracy_curve, f)

            if allow_print:
                progress.set_description(
                    gen_tqdm_description(
                        'Training backbone and classifiers...',
                        train_loss=mean_train_loss,
                        train_accuracy=mean_train_accuracy,
                        val_accuracy=mean_val_accuracy
                    )
                )

        if allow_print:
            progress.close()

        # Load the best-accuracy state dict if configured to do so
        # NOTE To save GPU memory, we could temporarily move the models to the
        # CPU before copying or loading their state dicts.
        if self._load_best_after_training:
            backbone.load_state_dict(best_accuracy_backbone_state_dict)
            species_classifier.load_state_dict(
                best_accuracy_species_classifier_state_dict
            )
            activity_classifier.load_state_dict(
                best_accuracy_activity_classifier_state_dict
            )

    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        # backbone.zero_grad(set_to_none=True)
        # torch.cuda.empty_cache()
        # pass
        classifier.reset()
        backbone.reset()
        activation_statistical_model.reset()

    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        activation_stats_training_loader = DataLoader(
            val_known_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=gen_custom_collate(),
            num_workers=0
        )
        backbone.eval()

        all_features = []
        with torch.no_grad():
            for _, _, _, _, batch in activation_stats_training_loader:
                batch = batch.to(device)
                features = activation_statistical_model.compute_features(
                    backbone,
                    batch
                )
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        activation_statistical_model.fit(all_features)

class EWCClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_dataset,
            val_known_dataset,
            box_transform,
            post_cache_train_transform,
            retraining_batch_size=32,
            root_checkpoint_dir=None,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            feedback_loss_weight=0.5,
            scheduler_type=SchedulerType.none,
            loss_fn=LossFnEnum.cross_entropy,
            class_frequencies=None,
            memory_cache=True,
            load_best_after_training=True,
            val_reduce_fn=None):
        self._lr = lr
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._retraining_batch_size = retraining_batch_size
        self._root_checkpoint_dir = root_checkpoint_dir
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._feedback_loss_weight = feedback_loss_weight
        self._scheduler_type = scheduler_type
        self._loss_fn = loss_fn
        self._class_frequencies = class_frequencies
        self._memory_cache = memory_cache
        self._load_best_after_training = load_best_after_training
        self._val_reduce_fn = val_reduce_fn

        self.just_finetune = False 
        self.ewc = True

    def _train_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer,
            feedback_species_labels,
            feedback_activity_labels,
            feedback_box_images,
            device,
            allow_print,
            class_frequencies):
                
        if feedback_box_images is not None:
            feedback_species_labels = feedback_species_labels.to(device)
            feedback_activity_labels = feedback_activity_labels.to(device)
            feedback_box_images = [x.to(device) for x in feedback_box_images]

            feedback_box_counts = [len(x) for x in feedback_box_images]
            feedback_box_images = torch.cat(feedback_box_images, dim=0)
            print_nan(feedback_box_images, 'feedback_box_images')

            feedback_box_features = backbone(feedback_box_images)
            if allow_print:
                print_nan(feedback_box_features, 'feedback_box_features')

            feedback_species_logits =\
                species_classifier(feedback_box_features)
            feedback_activity_logits =\
                activity_classifier(feedback_box_features)
            if allow_print:
                print_nan(feedback_activity_logits, 'feedback_activity_logits')

            feedback_species_preds =\
                torch.nn.functional.softmax(feedback_species_logits, dim=1)
            feedback_activity_preds =\
                torch.nn.functional.softmax(feedback_activity_logits, dim=1)

            if allow_print:
                print_nan(feedback_activity_preds, 'feedback_activity_preds')

            if class_frequencies is not None:
                # Compute weights for class balancing
                species_frequencies, activity_frequencies = class_frequencies

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

                # Scale the gradients according to class weights
                feedback_species_preds = _balance_box_prediction_grads(
                    feedback_species_preds,
                    species_weights
                )
                feedback_activity_preds = _balance_box_prediction_grads(
                    feedback_activity_preds,
                    activity_weights
                )
            
            # Re-split feedback predictions for loss computation
            feedback_species_preds = torch.split(
                feedback_species_preds,
                feedback_box_counts,
                dim=0
            )
            feedback_activity_preds = torch.split(
                feedback_activity_preds,
                feedback_box_counts,
                dim=0
            )

            # We have image-level count feedback labels for species
            feedback_species_loss =\
                multiple_instance_count_cross_entropy_dyn(
                    feedback_species_preds,
                    feedback_species_labels,
                    allow_print=allow_print
                )
            if allow_print:
                print_nan(feedback_species_loss, "feedback_species_loss")

            # We have image-level presence feedback labels for activities
            feedback_activity_loss =\
                multiple_instance_presence_cross_entropy_dyn(
                    feedback_activity_preds,
                    feedback_activity_labels
                )
            if allow_print:
                print_nan(feedback_activity_loss, "feedback_activity_loss")

            feedback_loss = feedback_species_loss + feedback_activity_loss
            if allow_print:
                print_nan(feedback_loss, "feedback_loss")

            
            ewc_penalty_ = self.ewc_calculation.penalty(backbone, species_classifier, activity_classifier)
            non_feedback_loss = 10000 * ewc_penalty_ 
            loss =  feedback_loss + non_feedback_loss
     
            if allow_print:
                if torch.any(torch.isnan(loss)):
                    print('feedback_species_loss', feedback_species_loss)
                    print('feedback_activity_loss', feedback_activity_loss)
                    print('feedback_loss', feedback_loss)
                    print('non_feedback_loss', non_feedback_loss)
                    print('loss', loss)
                print_nan(feedback_loss, 'feedback_loss')
                print_nan(non_feedback_loss, 'non_feedback_loss')
                print_nan(loss, 'loss 1')
        else:
            loss = non_feedback_loss
            if allow_print:
                print_nan(loss, 'loss 2')

        optimizer.zero_grad()
        if allow_print:
            print_nan(loss, 'loss 3')
            for param in backbone.parameters():
                print_nan(param, 'some parameter 1')

            for param in backbone.parameters():
                if param.grad is not None:
                    print_nan(param.grad.data, 'some parameter\'s gradient 1')
        loss.backward()

        for param in species_classifier.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of species_classifier. Skiping this batch ...')
                optimizer.zero_grad()
                break
                
        for param in activity_classifier.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of activity_classifier. Skiping this batch ...')
                optimizer.zero_grad()
                break
                
        for param in backbone.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of backbone. Skiping this batch ...')
                optimizer.zero_grad()
                break


        optimizer.step()
        indices_with_len_1 = [i for i, tensor in enumerate(feedback_species_preds) if tensor.size(0) == 1]

        if len(indices_with_len_1) > 0:
            feedback_species_preds_len_1 = torch.stack([feedback_species_preds[i] for i in indices_with_len_1]).squeeze(1)
            feedback_activity_preds_len_1 = torch.stack([feedback_activity_preds[i] for i in indices_with_len_1]).squeeze(1)
            numerical_feedback_species_labels = feedback_species_labels[indices_with_len_1].to(dtype=torch.int)
            numerical_feedback_activity_labels = feedback_activity_labels[indices_with_len_1].to(dtype=torch.int)

            species_correct = torch.argmax(feedback_species_preds_len_1, dim=1) == \
                torch.argmax(numerical_feedback_species_labels, dim=1)
            n_species_correct = int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(feedback_activity_preds_len_1, dim=1) == \
                torch.argmax(numerical_feedback_activity_labels, dim=1)
            n_activity_correct = int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )
        else:
            n_species_correct = 0.
            n_activity_correct = 0.
        return loss.detach().cpu().item(),\
            n_species_correct,\
            n_activity_correct, \
            feedback_species_loss, \
            feedback_activity_loss, \
            non_feedback_loss


    def _train_epoch(
            self,
            backbone,
            feedback_loader,
            species_classifier,
            activity_classifier,
            optimizer,
            device,
            allow_print,
            class_frequencies):
        # Set everything to train mode
        backbone.train()
        species_classifier.train()
        activity_classifier.train()
        
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_examples = 0
        n_species_correct = 0
        n_activity_correct = 0
        activity_loss = 0
        species_loss = 0
        ewc_penalty = 0

        feedback_iter =\
            iter(feedback_loader) if feedback_loader is not None else None
        #length = sum(1 for _ in feedback_iter)
        #print('feedback_iter 1', length)  # Outputs: 5
        #print(feedback_iter)
        feedback_species_labels = None
        feedback_activity_labels = None
        feedback_box_images = None
        batch_loss = None
        
        if allow_print:
            train_feedback_loader_progress = tqdm(
                feedback_loader,
                desc=gen_tqdm_description(
                    'Training batch...',
                    batch_loss=batch_loss
                )
            )
        else:
            train_feedback_loader_progress = feedback_loader
        for i in train_feedback_loader_progress:
            if feedback_iter is not None:
                try:
                    feedback_species_labels,\
                        feedback_activity_labels,\
                        _,\
                        feedback_box_images,\
                        _ = next(feedback_iter)
                except:
                    feedback_iter = iter(feedback_loader)
                    feedback_species_labels,\
                        feedback_activity_labels,\
                        _,\
                        feedback_box_images,\
                        _ = next(feedback_iter)
                gc.collect()
            batch_loss, batch_n_species_correct, batch_n_activity_correct, \
            species_l, activity_l, ewc_p =\
                self._train_batch(
                    backbone,
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    feedback_species_labels,
                    feedback_activity_labels,
                    feedback_box_images,
                    device,
                    allow_print,
                    class_frequencies
                )

            sum_loss += batch_loss
            activity_loss += activity_l
            species_loss += species_l  
            ewc_penalty += ewc_p
            n_iterations += 1
            n_examples += len(feedback_box_images)
            n_species_correct += batch_n_species_correct
            n_activity_correct += batch_n_activity_correct
            
            if allow_print:
                train_feedback_loader_progress.set_description(
                    gen_tqdm_description(
                        'Training batch...',
                        batch_loss=batch_loss
                    )
                )
            
        activity_loss /= n_iterations
        species_loss /= n_iterations  
        ewc_penalty /= n_iterations

        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, species_loss, activity_loss, ewc_penalty, \
        mean_accuracy, mean_species_accuracy, mean_activity_accuracy

    def _val_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images,
            device):
        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = backbone(box_images)

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_correct = torch.argmax(species_preds, dim=1) == \
            species_labels
        n_species_correct = species_correct.to(torch.int).sum()

        activity_correct = torch.argmax(activity_preds, dim=1) == \
            activity_labels
        n_activity_correct = activity_correct.to(torch.int).sum()

        return n_species_correct, n_activity_correct

    def _val_epoch(
            self,
            backbone,
            data_loader,
            species_classifier,
            activity_classifier,
            device):
        with torch.no_grad():
            backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = torch.zeros(1, device=device)
            n_species_correct = torch.zeros(1, device=device)
            n_activity_correct = torch.zeros(1, device=device)

            for species_labels, activity_labels, box_images in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        backbone,
                        species_classifier,
                        activity_classifier,
                        species_labels,
                        activity_labels,
                        box_images,
                        device
                    )
                n_examples += box_images.shape[0]
                n_species_correct += batch_n_species_correct
                n_activity_correct += batch_n_activity_correct

            if self._val_reduce_fn is not None:
                self._val_reduce_fn(n_examples)
                self._val_reduce_fn(n_species_correct)
                self._val_reduce_fn(n_activity_correct)
            mean_species_accuracy = float(n_species_correct.detach().cpu().item()) / float(n_examples.detach().cpu().item())
            mean_activity_accuracy = float(n_activity_correct.detach().cpu().item()) / float(n_examples.detach().cpu().item())

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        if feedback_sampling_configuration is not None:
            raise ValueError(('EWC training currently only supports '
                              'feedback_sampling_configuration=None'))

        train_dataset = self._train_dataset

        if self._memory_cache:
            train_dataset = \
                FlattenedBoxImageDataset(BoxImageMemoryDataset(train_dataset))
        else:
            train_dataset = FlattenedBoxImageDataset(train_dataset)
        
        if train_sampler_fn is not None:
            train_sampler = train_sampler_fn(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self._retraining_batch_size,
                num_workers=0,
                sampler=train_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self._retraining_batch_size,
                shuffle=True,
                num_workers=0
            )

        if feedback_dataset is not None:
            feedback_box_counts = [
                feedback_dataset.box_count(i) for\
                    i in range(len(feedback_dataset))
            ]
            if feedback_batch_sampler_fn is not None:
                feedback_batch_sampler = feedback_batch_sampler_fn(
                    feedback_box_counts
                )
            else:
                feedback_batch_sampler = DistributedRandomBoxImageBatchSampler(
                    feedback_box_counts,
                    self._retraining_batch_size,
                    1,
                    0
                )
            feedback_loader = DataLoader(
                feedback_dataset,
                num_workers=0,
                batch_sampler=feedback_batch_sampler,
                collate_fn=gen_custom_collate()
            )
        else:
            feedback_loader = None
        
        # Construct validation loaders for early stopping / model selection.
        # I'm assuming our model selection strategy will be based solely on the
        # validation classification accuracy and not based on novelty detection
        # capabilities in any way. Otherwise, we can use the novel validation
        # data to measure novelty detection performance. These currently aren't
        # being stored (except in a special form for the logistic regressions),
        # so we'd have to modify __init__().
        if self._memory_cache:
            val_dataset = FlattenedBoxImageDataset(BoxImageMemoryDataset(self._val_known_dataset))
        else:
            val_dataset = FlattenedBoxImageDataset(self._val_known_dataset)

        if train_sampler_fn is not None:
            val_sampler = train_sampler_fn(val_dataset)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._retraining_batch_size,
                num_workers=0,
                sampler=val_sampler
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._retraining_batch_size,
                shuffle=False,
                num_workers=0
            )

        # Retrain the backbone and classifiers
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(backbone.parameters())\
                + list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            self._lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Init scheduler to None. It will be constructed after loading
        # the optimizer state dict, or after failing to do so
        scheduler = None
        scheduler_ctor = self._scheduler_type.ctor()

        # Define convergence parameters (early stopping + model selection)
        start_epoch = 0
        epochs_since_improvement = 0
        best_accuracy = None
        best_accuracy_backbone_state_dict = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None
        mean_train_loss = None
        mean_train_accuracy = None
        mean_val_accuracy = None

        def get_checkpoint_dir():
            return os.path.join(
                self._root_checkpoint_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )

        if self._root_checkpoint_dir is not None:
            checkpoint_dir = get_checkpoint_dir()
            training_checkpoint = os.path.join(checkpoint_dir, 'training.pth')
            validation_checkpoint =\
                os.path.join(checkpoint_dir, 'validation.pth')
            
            if os.path.exists(training_checkpoint):
                sd = torch.load(
                    training_checkpoint,
                    map_location=device
                )
                backbone.load_state_dict(sd['backbone'])
                species_classifier.load_state_dict(sd['species_classifier'])
                activity_classifier.load_state_dict(sd['activity_classifier'])
                optimizer.load_state_dict(sd['optimizer'])
                start_epoch = sd['start_epoch']
                mean_train_loss = sd['mean_train_loss']
                mean_train_accuracy = sd['mean_train_accuracy']

                if scheduler_ctor is not None:
                    scheduler = scheduler_ctor(
                        optimizer,
                        self._max_epochs
                    )
                    scheduler.load_state_dict(sd['scheduler'])
            if os.path.exists(validation_checkpoint):
                sd = torch.load(
                    validation_checkpoint,
                    map_location=device
                )
                epochs_since_improvement = sd['epochs_since_improvement']
                best_accuracy = sd['accuracy']
                best_accuracy_backbone_state_dict = sd['backbone_state_dict']
                best_accuracy_species_classifier_state_dict =\
                    sd['species_classifier_state_dict']
                best_accuracy_activity_classifier_state_dict =\
                    sd['activity_classifier_state_dict']
                mean_val_accuracy = sd['mean_val_accuracy']
        
        # If we didn't load an optimizer state dict, and so the scheduler
        # hasn't been constructed yet, then construct it
        if scheduler_ctor is not None and scheduler is None:
            scheduler = scheduler_ctor(
                optimizer,
                self._max_epochs
            )
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}
        training_species_loss_curve = {}
        training_activity_loss_curve = {}
        training_mean_species_accuracy_curve  = {}
        training_mean_activity_accuracy_curve  = {}
        training_ewc_penalty_curve = {}

        def get_log_dir():
            time_stamp = datetime.now().strftime('%H-%M')
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}',
                time_stamp  # Add just the time as the last component of the path
            )

        if root_log_dir is not None and allow_write:
            log_dir = get_log_dir()
            training_log = os.path.join(log_dir, 'training.pkl')
            validation_log =\
                os.path.join(log_dir, 'validation.pkl')

            if os.path.exists(training_log):
                with open(training_log, 'rb') as f:
                    sd = pkl.load(f)
                    training_loss_curve = sd['training_loss_curve']
                    training_accuracy_curve = sd['training_accuracy_curve']

            if os.path.exists(validation_log):
                with open(validation_log, 'rb') as f:
                    validation_accuracy_curve = pkl.load(f)

        # Combine train and feedback class frequencies
        species_frequencies = None
        activity_frequencies = None
        if self._class_frequencies is not None:
            train_species_frequencies,\
                train_activity_frequencies = self._class_frequencies
            train_species_frequencies = train_species_frequencies.to(device)
            train_activity_frequencies = train_activity_frequencies.to(device)
            
            species_frequencies = train_species_frequencies
            activity_frequencies = train_activity_frequencies

        if feedback_class_frequencies is not None:
            feedback_species_frequencies,\
                feedback_activity_frequencies = feedback_class_frequencies
            feedback_species_frequencies =\
                feedback_species_frequencies.to(device)
            feedback_activity_frequencies =\
                feedback_activity_frequencies.to(device)

            if species_frequencies is None:
                species_frequencies = feedback_species_frequencies
                activity_frequencies = feedback_activity_frequencies
            else:
                species_frequencies =\
                    species_frequencies + feedback_species_frequencies
                activity_frequencies =\
                    activity_frequencies + feedback_activity_frequencies

        if species_frequencies is not None:
            class_frequencies = (species_frequencies, activity_frequencies)
        else:
            class_frequencies = None

        # Train
        pre_ewc_path = '.temp/pre_ewc_path.pth'
        if self.just_finetune == False and self.ewc == True:
            self.ewc_calculation = EWC_All_Models(backbone, species_classifier, activity_classifier, train_loader, class_frequencies, self._loss_fn, self._label_smoothing, device, pre_ewc_path)

        if allow_print:
            print('lr:', self._lr)
            progress = tqdm(
                range(start_epoch, self._max_epochs),
                desc=gen_tqdm_description(
                    'Training backbone and classifiers...',
                    train_loss=mean_train_loss,
                    train_accuracy=mean_train_accuracy,
                    val_accuracy=mean_val_accuracy
                ),
                total=self._max_epochs,
                initial=start_epoch
            )
        else:
            progress = range(start_epoch, self._max_epochs)
        for epoch in progress:
            if self._patience is not None and\
                    epochs_since_improvement >= self._patience:
                # We haven't improved in several epochs. Time to stop
                # training.
                break
            
            mean_train_loss, species_loss, activity_loss, ewc_penalty, \
            mean_train_accuracy, mean_species_accuracy, mean_activity_accuracy = \
            self._train_epoch(
                backbone,
                feedback_loader,
                species_classifier,
                activity_classifier,
                optimizer,
                device,
                allow_print,
                class_frequencies
            )

            # print(f'Mean Train Loss: {mean_train_loss}, '
            #         f'Species Loss: {species_loss}, '
            #         f'Activity Loss: {activity_loss}, '
            #         f'EWC Penalty: {ewc_penalty}, '
            #         f'Mean Train Accuracy: {mean_train_accuracy}, '
            #         f'Mean Species Accuracy: {mean_species_accuracy}, '
            #         f'Mean Activity Accuracy: {mean_activity_accuracy}')

            if self._root_checkpoint_dir is not None and allow_write:
                checkpoint_dir = get_checkpoint_dir()
                training_checkpoint =\
                    os.path.join(checkpoint_dir, 'training.pth')
                os.makedirs(checkpoint_dir, exist_ok=True)

                sd = {}
                sd['backbone'] = backbone.state_dict()
                sd['species_classifier'] = species_classifier.state_dict()
                sd['activity_classifier'] = activity_classifier.state_dict()
                sd['optimizer'] = optimizer.state_dict()
                sd['start_epoch'] = epoch + 1
                sd['mean_train_loss'] = mean_train_loss
                sd['mean_train_accuracy'] = mean_train_accuracy
                if scheduler is not None:
                    sd['scheduler'] = scheduler.state_dict()
                torch.save(sd, training_checkpoint)

            if root_log_dir is not None and allow_write:
                training_loss_curve[epoch] = mean_train_loss
                training_accuracy_curve[epoch] = mean_train_accuracy
                training_species_loss_curve[epoch] = species_loss
                training_activity_loss_curve[epoch] = activity_loss
                training_mean_species_accuracy_curve[epoch] = mean_species_accuracy
                training_mean_activity_accuracy_curve[epoch] = mean_activity_accuracy
                training_ewc_penalty_curve[epoch] = ewc_penalty
                log_dir = get_log_dir()
                os.makedirs(log_dir, exist_ok=True)
                training_log = os.path.join(log_dir, 'training.pkl')

                with open(training_log, 'wb') as f:
                    sd = {}
                    sd['training_loss_curve'] = training_loss_curve
                    sd['training_accuracy_curve'] = training_accuracy_curve
                    sd['training_species_loss_curve'] = training_species_loss_curve
                    sd['training_activity_loss_curve'] = training_activity_loss_curve
                    sd['training_mean_species_accuracy_curve'] = training_mean_species_accuracy_curve
                    sd['training_mean_activity_accuracy_curve'] = training_mean_activity_accuracy_curve
                    sd['training_ewc_penalty_curve'] = training_ewc_penalty_curve
                    pkl.dump(sd, f)

            # Measure validation accuracy for early stopping / model selection.
            if epoch >= self._min_epochs - 1:
                mean_val_accuracy = self._val_epoch(
                    backbone,
                    val_loader,
                    species_classifier,
                    activity_classifier,
                    device
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict =\
                        deepcopy(backbone.state_dict())
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if self._root_checkpoint_dir is not None and allow_write:
                    checkpoint_dir = get_checkpoint_dir()
                    validation_checkpoint =\
                        os.path.join(checkpoint_dir, 'validation.pth')
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    sd = {}
                    sd['epochs_since_improvement'] = epochs_since_improvement
                    sd['accuracy'] = best_accuracy
                    sd['backbone_state_dict'] =\
                        best_accuracy_backbone_state_dict
                    sd['species_classifier_state_dict'] =\
                        best_accuracy_species_classifier_state_dict
                    sd['activity_classifier_state_dict'] =\
                        best_accuracy_activity_classifier_state_dict
                    sd['mean_val_accuracy'] = mean_val_accuracy
                    torch.save(sd, validation_checkpoint)

                if root_log_dir is not None and allow_write:
                    validation_accuracy_curve[epoch] = mean_val_accuracy
                    log_dir = get_log_dir()
                    os.makedirs(log_dir, exist_ok=True)
                    validation_log = os.path.join(log_dir, 'validation.pkl')
                    
                    with open(validation_log, 'wb') as f:
                        pkl.dump(validation_accuracy_curve, f)

            if allow_print:
                progress.set_description(
                    gen_tqdm_description(
                        'Training backbone and classifiers...',
                        train_loss=mean_train_loss,
                        train_accuracy=mean_train_accuracy,
                        val_accuracy=mean_val_accuracy
                    )
                )

        if allow_print:
            progress.close()

        # Load the best-accuracy state dict if configured to do so
        # NOTE To save GPU memory, we could temporarily move the models to the
        # CPU before copying or loading their state dicts.
        if self._load_best_after_training:
            backbone.load_state_dict(best_accuracy_backbone_state_dict)
            species_classifier.load_state_dict(
                best_accuracy_species_classifier_state_dict
            )
            activity_classifier.load_state_dict(
                best_accuracy_activity_classifier_state_dict
            )

    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        # backbone.zero_grad(set_to_none=True)
        # torch.cuda.empty_cache()
        pass
        # classifier.reset()
        # backbone.reset()
        # activation_statistical_model.reset()

    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        activation_stats_training_loader = DataLoader(
            val_known_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=gen_custom_collate(),
            num_workers=0
        )
        backbone.eval()

        all_features = []
        with torch.no_grad():
            for _, _, _, _, batch in activation_stats_training_loader:
                batch = batch.to(device)
                features = activation_statistical_model.compute_features(
                    backbone,
                    batch
                )
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        activation_statistical_model.fit(all_features)
        

class SLCAClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_dataset,
            val_known_dataset,
            box_transform,
            post_cache_train_transform,
            retraining_batch_size=32,
            root_checkpoint_dir=None,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            feedback_loss_weight=0.5,
            scheduler_type=SchedulerType.none,
            loss_fn=LossFnEnum.cross_entropy,
            class_frequencies=None,
            memory_cache=True,
            load_best_after_training=True,
            val_reduce_fn=None,
            sl_lr_scale=0.1,
            ca_epochs=5,
            num_samples_per_class=256,
            with_EWC = True):
        
        self.sl_lr_scale = sl_lr_scale
        self.ca_epochs = ca_epochs
        self.num_samples_per_class = num_samples_per_class
        self.class_means = {}
        self.class_covs = {}
        self._lr = lr
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._retraining_batch_size = retraining_batch_size
        self._root_checkpoint_dir = root_checkpoint_dir
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._feedback_loss_weight = feedback_loss_weight
        self._scheduler_type = scheduler_type
        self._loss_fn = loss_fn
        self._class_frequencies = class_frequencies
        self._memory_cache = memory_cache
        self._load_best_after_training = load_best_after_training
        self._val_reduce_fn = val_reduce_fn

        self.just_finetune = False 
        self.ewc = with_EWC

    def _train_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer,
            feedback_species_labels,
            feedback_activity_labels,
            feedback_box_images,
            device,
            allow_print,
            class_frequencies):
                
        if feedback_box_images is not None:
            feedback_species_labels = feedback_species_labels.to(device)
            feedback_activity_labels = feedback_activity_labels.to(device)
            feedback_box_images = [x.to(device) for x in feedback_box_images]

            feedback_box_counts = [len(x) for x in feedback_box_images]
            feedback_box_images = torch.cat(feedback_box_images, dim=0)
            print_nan(feedback_box_images, 'feedback_box_images')

            feedback_box_features = backbone(feedback_box_images)
            if allow_print:
                print_nan(feedback_box_features, 'feedback_box_features')

            feedback_species_logits =\
                species_classifier(feedback_box_features)
            feedback_activity_logits =\
                activity_classifier(feedback_box_features)
            if allow_print:
                print_nan(feedback_activity_logits, 'feedback_activity_logits')

            feedback_species_preds =\
                torch.nn.functional.softmax(feedback_species_logits, dim=1)
            feedback_activity_preds =\
                torch.nn.functional.softmax(feedback_activity_logits, dim=1)

            if allow_print:
                print_nan(feedback_activity_preds, 'feedback_activity_preds')

            if class_frequencies is not None:
                # Compute weights for class balancing
                species_frequencies, activity_frequencies = class_frequencies

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

                # Scale the gradients according to class weights
                feedback_species_preds = _balance_box_prediction_grads(
                    feedback_species_preds,
                    species_weights
                )
                feedback_activity_preds = _balance_box_prediction_grads(
                    feedback_activity_preds,
                    activity_weights
                )
            
            # Re-split feedback predictions for loss computation
            feedback_species_preds = torch.split(
                feedback_species_preds,
                feedback_box_counts,
                dim=0
            )
            feedback_activity_preds = torch.split(
                feedback_activity_preds,
                feedback_box_counts,
                dim=0
            )

            # We have image-level count feedback labels for species
            feedback_species_loss =\
                multiple_instance_count_cross_entropy_dyn(
                    feedback_species_preds,
                    feedback_species_labels,
                    allow_print=allow_print
                )
            if allow_print:
                print_nan(feedback_species_loss, "feedback_species_loss")

            # We have image-level presence feedback labels for activities
            feedback_activity_loss =\
                multiple_instance_presence_cross_entropy_dyn(
                    feedback_activity_preds,
                    feedback_activity_labels
                )
            if allow_print:
                print_nan(feedback_activity_loss, "feedback_activity_loss")

            feedback_loss = feedback_species_loss + feedback_activity_loss
            if allow_print:
                print_nan(feedback_loss, "feedback_loss")

            if self.ewc == True:
                ewc_penalty_ = self.ewc_calculation.penalty(backbone, species_classifier, activity_classifier)
                non_feedback_loss = 10000 * ewc_penalty_ 
            else:
                non_feedback_loss = 0.
            loss =  feedback_loss + non_feedback_loss
     
            if allow_print:
                if torch.any(torch.isnan(loss)):
                    print('feedback_species_loss', feedback_species_loss)
                    print('feedback_activity_loss', feedback_activity_loss)
                    print('feedback_loss', feedback_loss)
                    print('non_feedback_loss', non_feedback_loss)
                    print('loss', loss)
                print_nan(feedback_loss, 'feedback_loss')
                print_nan(non_feedback_loss, 'non_feedback_loss')
                print_nan(loss, 'loss 1')
        else:
            loss = non_feedback_loss
            if allow_print:
                print_nan(loss, 'loss 2')

        optimizer.zero_grad()
        if allow_print:
            print_nan(loss, 'loss 3')
            for param in backbone.parameters():
                print_nan(param, 'some parameter 1')

            for param in backbone.parameters():
                if param.grad is not None:
                    print_nan(param.grad.data, 'some parameter\'s gradient 1')
        loss.backward()

        for param in species_classifier.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of species_classifier. Skiping this batch ...')
                optimizer.zero_grad()
                break
                
        for param in activity_classifier.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of activity_classifier. Skiping this batch ...')
                optimizer.zero_grad()
                break
                
        for param in backbone.parameters():
            if param.grad is not None and \
            (torch.any(torch.isnan(param.grad.data)) or \
                torch.any(torch.isinf(param.grad.data))):
                # Found some NaNs in the gradients. "Skip" this batch by
                # zeroing the gradients. This should work in distributed
                # mode as well
                print('Some NaNs in the gradients of backbone. Skiping this batch ...')
                optimizer.zero_grad()
                break


        optimizer.step()
        indices_with_len_1 = [i for i, tensor in enumerate(feedback_species_preds) if tensor.size(0) == 1]

        if len(indices_with_len_1) > 0:
            feedback_species_preds_len_1 = torch.stack([feedback_species_preds[i] for i in indices_with_len_1]).squeeze(1)
            feedback_activity_preds_len_1 = torch.stack([feedback_activity_preds[i] for i in indices_with_len_1]).squeeze(1)
            numerical_feedback_species_labels = feedback_species_labels[indices_with_len_1].to(dtype=torch.int)
            numerical_feedback_activity_labels = feedback_activity_labels[indices_with_len_1].to(dtype=torch.int)

            species_correct = torch.argmax(feedback_species_preds_len_1, dim=1) == \
                torch.argmax(numerical_feedback_species_labels, dim=1)
            n_species_correct = int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(feedback_activity_preds_len_1, dim=1) == \
                torch.argmax(numerical_feedback_activity_labels, dim=1)
            n_activity_correct = int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )
        else:
            n_species_correct = 0.
            n_activity_correct = 0.
        return loss.detach().cpu().item(),\
            n_species_correct,\
            n_activity_correct, \
            feedback_species_loss, \
            feedback_activity_loss, \
            non_feedback_loss


    def _compute_class_statistics_pretrained(self, backbone, dataset, device):
        """
        Compute class statistics (mean and covariance) for both species and activity classifiers.
        If the statistics are already computed and saved in a file, load them instead of recalculating.
        """

        pretrained_stats_file = '.temp/class_stats.pth'
        # Check if statistics file exists
        if os.path.exists(pretrained_stats_file):
            print(f"Loading precomputed class statistics from {pretrained_stats_file}")
            stats = torch.load(pretrained_stats_file)
            self.class_means = stats["class_means"]
            self.class_covs = stats["class_covs"]
            return

        print("Computing pretrained class statistics...")
        dataloader = DataLoader(dataset, batch_size=self._retraining_batch_size, shuffle=False)
        backbone.eval()

        # Separate dictionaries for species and activity statistics
        species_features = []
        species_labels = []
        activity_features = []
        activity_labels = []

        with torch.no_grad():
            # Wrap the dataloader loop with tqdm for progress tracking
            for batch_species_labels, batch_activity_labels, box_images in tqdm(
                dataloader, desc="Computing Class Statistics", unit="batch"
            ):
                box_images = box_images.to(device)
                features_batch = backbone(box_images).detach()

                # Append features and labels for species
                species_features.append(features_batch)
                species_labels.append(batch_species_labels)

                # Append features and labels for activities
                activity_features.append(features_batch)
                activity_labels.append(batch_activity_labels)

        # Concatenate features and labels for both species and activity
        species_features = torch.cat(species_features, dim=0)
        species_labels = torch.cat(species_labels, dim=0)
        activity_features = torch.cat(activity_features, dim=0)
        activity_labels = torch.cat(activity_labels, dim=0)

        # Compute mean and covariance for species classifier
        for c in torch.unique(species_labels):
            class_features = species_features[species_labels == c]
            # Skip the class if it has only a single sample
            if class_features.size(0) <= 1:
                print(f"Skipping Species Class {c.item()} due to insufficient samples.")
                continue

            self.class_means["species", c.item()] = class_features.mean(dim=0)
            self.class_covs["species", c.item()] = torch.cov(class_features.T)

        # Compute mean and covariance for activity classifier
        for c in torch.unique(activity_labels):
            class_features = activity_features[activity_labels == c]
            # Skip the class if it has only a single sample
            if class_features.size(0) <= 1:
                print(f"Skipping Activity Class {c.item()} due to insufficient samples.")
                continue

            self.class_means["activity", c.item()] = class_features.mean(dim=0)
            self.class_covs["activity", c.item()] = torch.cov(class_features.T)

        # Save the computed statistics
        print(f"Saving computed class statistics to {pretrained_stats_file}")
        stats = {
            "class_means": self.class_means,
            "class_covs": self.class_covs,
        }
        torch.save(stats, pretrained_stats_file)



    def _compute_class_statistics_post_retraining(self, backbone, feedback_loader, device, allow_print):
        """
        Compute class statistics (mean and covariance) for both species and activity classifiers 
        using the feedback loader. Only one box per image is selected.
        """
        backbone.eval()

        # Separate dictionaries for species and activity statistics
        species_features = []
        species_labels = []
        activity_features = []
        activity_labels = []

        feedback_iter = iter(feedback_loader) if feedback_loader is not None else None

        if allow_print:
            train_feedback_loader_progress = tqdm(
                feedback_loader,
                desc="Computing Class Statistics for Feedback Data...",
                unit="batch"
            )
        else:
            train_feedback_loader_progress = feedback_loader

        for batch in train_feedback_loader_progress:
            if feedback_iter is not None:
                try:
                    feedback_species_labels, feedback_activity_labels, _, feedback_box_images, _ = next(feedback_iter)
                except StopIteration:
                    feedback_iter = iter(feedback_loader)
                    feedback_species_labels, feedback_activity_labels, _, feedback_box_images, _ = next(feedback_iter)

            gc.collect()

            # Select the first box and corresponding label per image
            selected_box_images = torch.stack([boxes[0] for boxes in feedback_box_images])

            # import ipdb; ipdb.set_trace()
            selected_species_labels = torch.argmax(feedback_species_labels.float(), dim=1).to(device)
            selected_activity_labels = torch.argmax(feedback_activity_labels.float(), dim=1).to(device)

            # Extract features using the backbone
            features_batch = backbone(selected_box_images).detach()

            # Append features and labels for species
            species_features.append(features_batch)
            species_labels.append(selected_species_labels)

            # Append features and labels for activities
            activity_features.append(features_batch)
            activity_labels.append(selected_activity_labels)

        # Concatenate features and labels for both species and activity
        species_features = torch.cat(species_features, dim=0)
        species_labels = torch.cat(species_labels, dim=0)
        activity_features = torch.cat(activity_features, dim=0)
        activity_labels = torch.cat(activity_labels, dim=0)

        # Compute mean and covariance for species classifier
        for c in torch.unique(species_labels):
            if ("species", c.item()) not in self.class_means:
                class_features = species_features[species_labels == c]
                # Skip the class if it has only a single sample
                if class_features.size(0) <= 1:
                    print(f"Skipping Species Class {c.item()} due to insufficient samples.")
                    continue

                self.class_means["species", c.item()] = class_features.mean(dim=0)
                self.class_covs["species", c.item()] = torch.cov(class_features.T)

        # Compute mean and covariance for activity classifier
        for c in torch.unique(activity_labels):
            if ("activity", c.item()) not in self.class_means:
                class_features = activity_features[activity_labels == c]
                # Skip the class if it has only a single sample
                if class_features.size(0) <= 1:
                    print(f"Skipping Activity Class {c.item()} due to insufficient samples.")
                    continue

                self.class_means["activity", c.item()] = class_features.mean(dim=0)
                self.class_covs["activity", c.item()] = torch.cov(class_features.T)

    def _compute_predicted_class_statistics_post_retraining(
        self, 
        backbone, 
        species_classifier, 
        activity_classifier, 
        feedback_loader, 
        device, 
        allow_print):
        """
        Compute class statistics (mean and covariance) for both species and activity classifiers 
        using the feedback loader and predicted labels. Only one box per image is selected.
        """
        backbone.eval()
        species_classifier.eval()
        activity_classifier.eval()

        # Separate dictionaries for species and activity statistics
        species_features = []
        species_labels = []
        activity_features = []
        activity_labels = []

        feedback_iter = iter(feedback_loader) if feedback_loader is not None else None

        if allow_print:
            train_feedback_loader_progress = tqdm(
                feedback_loader,
                desc="Computing Class Statistics for Feedback Data...",
                unit="batch"
            )
        else:
            train_feedback_loader_progress = feedback_loader

        for batch in train_feedback_loader_progress:
            if feedback_iter is not None:
                try:
                    feedback_species_labels, feedback_activity_labels, _, feedback_box_images, _ = next(feedback_iter)
                except StopIteration:
                    feedback_iter = iter(feedback_loader)
                    feedback_species_labels, feedback_activity_labels, _, feedback_box_images, _ = next(feedback_iter)

            gc.collect()

            feedback_species_labels = feedback_species_labels.to(device)
            feedback_activity_labels = feedback_activity_labels.to(device)
            feedback_box_images = [x.to(device) for x in feedback_box_images]

            feedback_box_counts = [len(x) for x in feedback_box_images]
            feedback_box_images = torch.cat(feedback_box_images, dim=0)

            # Extract features using the backbone
            features_batch = backbone(feedback_box_images).detach()

            # Predict species and activity labels
            predicted_species_scores = torch.nn.functional.softmax(species_classifier(features_batch), dim=1)
            predicted_activity_scores = torch.nn.functional.softmax(activity_classifier(features_batch), dim=1)

            # Get max scores and their corresponding labels
            species_max_scores, predicted_species_labels = predicted_species_scores.max(dim=1)
            activity_max_scores, predicted_activity_labels = predicted_activity_scores.max(dim=1)

            # Filter predictions by score threshold
            valid_species_indices = species_max_scores > 0.9
            valid_activity_indices = activity_max_scores > 0.9

            # Append valid features and predicted labels for species
            species_features.append(features_batch[valid_species_indices])
            species_labels.append(predicted_species_labels[valid_species_indices])

            # Append valid features and predicted labels for activities
            activity_features.append(features_batch[valid_activity_indices])
            activity_labels.append(predicted_activity_labels[valid_activity_indices])

        # Concatenate features and labels for both species and activity
        species_features = torch.cat(species_features, dim=0)
        species_labels = torch.cat(species_labels, dim=0)
        activity_features = torch.cat(activity_features, dim=0)
        activity_labels = torch.cat(activity_labels, dim=0)

        global_cov = torch.cov(species_features.T)

        # Compute mean and covariance for species classifier
        for c in torch.unique(species_labels):
            if ("species", c.item()) not in self.class_means:
                class_features = species_features[species_labels == c]
                print("species", c.item(), class_features.shape)

                if class_features.size(0) == 1:
                    print(f"Handling single sample for Species Class {c.item()} with global covariance.")
                    self.class_means["species", c.item()] = class_features.mean(dim=0)
                    self.class_covs["species", c.item()] = global_cov
                else:
                    self.class_means["species", c.item()] = class_features.mean(dim=0)
                    self.class_covs["species", c.item()] = torch.cov(class_features.T)

        # Compute mean and covariance for activity classifier
        global_cov = torch.cov(activity_features.T)

        for c in torch.unique(activity_labels):
            if ("activity", c.item()) not in self.class_means:
                class_features = activity_features[activity_labels == c]
                print("activity", c.item(), class_features.shape)

                if class_features.size(0) == 1:
                    print(f"Handling single sample for Activity Class {c.item()} with global covariance.")
                    self.class_means["activity", c.item()] = class_features.mean(dim=0)
                    self.class_covs["activity", c.item()] = global_cov
                else:
                    self.class_means["activity", c.item()] = class_features.mean(dim=0)
                    self.class_covs["activity", c.item()] = torch.cov(class_features.T)



    def _classifier_alignment(self, classifier, classifier_name, device):
        """
        Perform classifier alignment for the given classifier with progress tracking.

        Parameters:
        - classifier: The classifier model (species_classifier or activity_classifier).
        - classifier_name: A string, either 'species' or 'activity', indicating which classifier's statistics to use.
        - device: The device to run the computations on (CPU or GPU).
        """
        classifier.train()
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=self._lr,
            momentum=0.9,
            weight_decay=1e-3
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.ca_epochs)

        # Filter class_means and class_covs for the given classifier
        class_ids = [key[1] for key in self.class_means.keys() if key[0] == classifier_name]
        class_ids = sorted(class_ids)  # Ensure consistent ordering

        # Prepare the class statistics for the given classifier
        class_means = {class_id: self.class_means[(classifier_name, class_id)] for class_id in class_ids}
        class_covs = {class_id: self.class_covs[(classifier_name, class_id)] for class_id in class_ids}

        # Wrap the epoch loop with tqdm for progress tracking
        for epoch in tqdm(range(self.ca_epochs), desc=f"Classifier Alignment Progress ({classifier_name})", unit="epoch"):
            sampled_features = []
            sampled_labels = []

            for class_id in class_ids:
                mean = class_means[class_id]
                cov = class_covs[class_id]

                try:
                    cov += torch.eye(cov.size(0), device=cov.device) * 1e-6  # Add small regularization
                    m = torch.distributions.MultivariateNormal(mean, cov)
                    samples = m.sample((self.num_samples_per_class,))
                    sampled_features.append(samples)
                    sampled_labels += [class_id] * self.num_samples_per_class
                except Exception as e:
                    print(f"Error creating MultivariateNormal for class {class_id}: {e}")
                    continue
                    
            sampled_features = torch.cat(sampled_features).to(device)
            sampled_labels = torch.tensor(sampled_labels).to(device)


            optimizer.zero_grad()
            logits = classifier(sampled_features)

            logit_norm = 0.1
            norm = torch.norm(logits, p=2, dim=-1, keepdim=True) + 1e-7
            normalized_logits = logits / norm / logit_norm
            loss = F.cross_entropy(normalized_logits, sampled_labels)

            loss.backward()
            optimizer.step()
            scheduler.step()



    def _train_epoch(
            self,
            backbone,
            feedback_loader,
            species_classifier,
            activity_classifier,
            optimizer,
            device,
            allow_print,
            class_frequencies):
        # Set everything to train mode
        backbone.train()
        species_classifier.train()
        activity_classifier.train()

        backbone_params = [{'params': backbone.parameters(), 'lr': self._lr * self.sl_lr_scale}]
        classifier_params = [
            {'params': species_classifier.parameters(), 'lr': self._lr},
            {'params': activity_classifier.parameters(), 'lr': self._lr}
        ]
        optimizer = torch.optim.SGD(
            backbone_params + classifier_params,
            momentum=0.9,
            weight_decay=1e-3
        )

        print(f'Backbone Lr: {self._lr * self.sl_lr_scale} Classifiers LR: {self._lr}')

    
            
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_examples = 0
        n_species_correct = 0
        n_activity_correct = 0
        activity_loss = 0
        species_loss = 0
        ewc_penalty = 0

        feedback_iter =\
            iter(feedback_loader) if feedback_loader is not None else None
        #length = sum(1 for _ in feedback_iter)
        #print('feedback_iter 1', length)  # Outputs: 5
        #print(feedback_iter)
        feedback_species_labels = None
        feedback_activity_labels = None
        feedback_box_images = None
        batch_loss = None
        
        if allow_print:
            train_feedback_loader_progress = tqdm(
                feedback_loader,
                desc=gen_tqdm_description(
                    'Training batch...',
                    batch_loss=batch_loss
                )
            )
        else:
            train_feedback_loader_progress = feedback_loader
        for i in train_feedback_loader_progress:
            if feedback_iter is not None:
                try:
                    feedback_species_labels,\
                        feedback_activity_labels,\
                        _,\
                        feedback_box_images,\
                        _ = next(feedback_iter)
                except:
                    feedback_iter = iter(feedback_loader)
                    feedback_species_labels,\
                        feedback_activity_labels,\
                        _,\
                        feedback_box_images,\
                        _ = next(feedback_iter)
                gc.collect()
            batch_loss, batch_n_species_correct, batch_n_activity_correct, \
            species_l, activity_l, ewc_p =\
                self._train_batch(
                    backbone,
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    feedback_species_labels,
                    feedback_activity_labels,
                    feedback_box_images,
                    device,
                    allow_print,
                    class_frequencies
                )

            sum_loss += batch_loss
            activity_loss += activity_l
            species_loss += species_l  
            ewc_penalty += ewc_p
            n_iterations += 1
            n_examples += len(feedback_box_images)
            n_species_correct += batch_n_species_correct
            n_activity_correct += batch_n_activity_correct
            
            if allow_print:
                train_feedback_loader_progress.set_description(
                    gen_tqdm_description(
                        'Training batch...',
                        batch_loss=batch_loss
                    )
                )
            
        activity_loss /= n_iterations
        species_loss /= n_iterations  
        ewc_penalty /= n_iterations

        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, species_loss, activity_loss, ewc_penalty, \
        mean_accuracy, mean_species_accuracy, mean_activity_accuracy

    def _val_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images,
            device):
        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = backbone(box_images)

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_correct = torch.argmax(species_preds, dim=1) == \
            species_labels
        n_species_correct = species_correct.to(torch.int).sum()

        activity_correct = torch.argmax(activity_preds, dim=1) == \
            activity_labels
        n_activity_correct = activity_correct.to(torch.int).sum()

        return n_species_correct, n_activity_correct

    def _val_epoch(
            self,
            backbone,
            data_loader,
            species_classifier,
            activity_classifier,
            device):
        with torch.no_grad():
            backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = torch.zeros(1, device=device)
            n_species_correct = torch.zeros(1, device=device)
            n_activity_correct = torch.zeros(1, device=device)

            for species_labels, activity_labels, box_images in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        backbone,
                        species_classifier,
                        activity_classifier,
                        species_labels,
                        activity_labels,
                        box_images,
                        device
                    )
                n_examples += box_images.shape[0]
                n_species_correct += batch_n_species_correct
                n_activity_correct += batch_n_activity_correct

            if self._val_reduce_fn is not None:
                self._val_reduce_fn(n_examples)
                self._val_reduce_fn(n_species_correct)
                self._val_reduce_fn(n_activity_correct)
            mean_species_accuracy = float(n_species_correct.detach().cpu().item()) / float(n_examples.detach().cpu().item())
            mean_activity_accuracy = float(n_activity_correct.detach().cpu().item()) / float(n_examples.detach().cpu().item())

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        if feedback_sampling_configuration is not None:
            raise ValueError(('EWC training currently only supports '
                              'feedback_sampling_configuration=None'))

        train_dataset = self._train_dataset

        if self._memory_cache:
            train_dataset = \
                FlattenedBoxImageDataset(BoxImageMemoryDataset(train_dataset))
        else:
            train_dataset = FlattenedBoxImageDataset(train_dataset)
        
        if train_sampler_fn is not None:
            train_sampler = train_sampler_fn(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self._retraining_batch_size,
                num_workers=0,
                sampler=train_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self._retraining_batch_size,
                shuffle=True,
                num_workers=0
            )

        if feedback_dataset is not None:
            feedback_box_counts = [
                feedback_dataset.box_count(i) for\
                    i in range(len(feedback_dataset))
            ]
            if feedback_batch_sampler_fn is not None:
                feedback_batch_sampler = feedback_batch_sampler_fn(
                    feedback_box_counts
                )
            else:
                feedback_batch_sampler = DistributedRandomBoxImageBatchSampler(
                    feedback_box_counts,
                    self._retraining_batch_size,
                    1,
                    0
                )
            feedback_loader = DataLoader(
                feedback_dataset,
                num_workers=0,
                batch_sampler=feedback_batch_sampler,
                collate_fn=gen_custom_collate()
            )
        else:
            feedback_loader = None
        
        # Construct validation loaders for early stopping / model selection.
        # I'm assuming our model selection strategy will be based solely on the
        # validation classification accuracy and not based on novelty detection
        # capabilities in any way. Otherwise, we can use the novel validation
        # data to measure novelty detection performance. These currently aren't
        # being stored (except in a special form for the logistic regressions),
        # so we'd have to modify __init__().
        if self._memory_cache:
            val_dataset = FlattenedBoxImageDataset(BoxImageMemoryDataset(self._val_known_dataset))
        else:
            val_dataset = FlattenedBoxImageDataset(self._val_known_dataset)

        if train_sampler_fn is not None:
            val_sampler = train_sampler_fn(val_dataset)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._retraining_batch_size,
                num_workers=0,
                sampler=val_sampler
            )
        else:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self._retraining_batch_size,
                shuffle=False,
                num_workers=0
            )

        # Retrain the backbone and classifiers
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(backbone.parameters())\
                + list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            self._lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Init scheduler to None. It will be constructed after loading
        # the optimizer state dict, or after failing to do so
        scheduler = None
        scheduler_ctor = self._scheduler_type.ctor()

        # Define convergence parameters (early stopping + model selection)
        start_epoch = 0
        epochs_since_improvement = 0
        best_accuracy = None
        best_accuracy_backbone_state_dict = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None
        mean_train_loss = None
        mean_train_accuracy = None
        mean_val_accuracy = None

        def get_checkpoint_dir():
            return os.path.join(
                self._root_checkpoint_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )

        if self._root_checkpoint_dir is not None:
            checkpoint_dir = get_checkpoint_dir()
            training_checkpoint = os.path.join(checkpoint_dir, 'training.pth')
            validation_checkpoint =\
                os.path.join(checkpoint_dir, 'validation.pth')
            
            if os.path.exists(training_checkpoint):
                sd = torch.load(
                    training_checkpoint,
                    map_location=device
                )
                backbone.load_state_dict(sd['backbone'])
                species_classifier.load_state_dict(sd['species_classifier'])
                activity_classifier.load_state_dict(sd['activity_classifier'])
                optimizer.load_state_dict(sd['optimizer'])
                start_epoch = sd['start_epoch']
                mean_train_loss = sd['mean_train_loss']
                mean_train_accuracy = sd['mean_train_accuracy']

                if scheduler_ctor is not None:
                    scheduler = scheduler_ctor(
                        optimizer,
                        self._max_epochs
                    )
                    scheduler.load_state_dict(sd['scheduler'])
            if os.path.exists(validation_checkpoint):
                sd = torch.load(
                    validation_checkpoint,
                    map_location=device
                )
                epochs_since_improvement = sd['epochs_since_improvement']
                best_accuracy = sd['accuracy']
                best_accuracy_backbone_state_dict = sd['backbone_state_dict']
                best_accuracy_species_classifier_state_dict =\
                    sd['species_classifier_state_dict']
                best_accuracy_activity_classifier_state_dict =\
                    sd['activity_classifier_state_dict']
                mean_val_accuracy = sd['mean_val_accuracy']
        
        # If we didn't load an optimizer state dict, and so the scheduler
        # hasn't been constructed yet, then construct it
        if scheduler_ctor is not None and scheduler is None:
            scheduler = scheduler_ctor(
                optimizer,
                self._max_epochs
            )
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}
        training_species_loss_curve = {}
        training_activity_loss_curve = {}
        training_mean_species_accuracy_curve  = {}
        training_mean_activity_accuracy_curve  = {}
        training_ewc_penalty_curve = {}

        def get_log_dir():
            time_stamp = datetime.now().strftime('%H-%M')
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}',
                time_stamp  # Add just the time as the last component of the path
            )

        if root_log_dir is not None and allow_write:
            log_dir = get_log_dir()
            training_log = os.path.join(log_dir, 'training.pkl')
            validation_log =\
                os.path.join(log_dir, 'validation.pkl')

            if os.path.exists(training_log):
                with open(training_log, 'rb') as f:
                    sd = pkl.load(f)
                    training_loss_curve = sd['training_loss_curve']
                    training_accuracy_curve = sd['training_accuracy_curve']

            if os.path.exists(validation_log):
                with open(validation_log, 'rb') as f:
                    validation_accuracy_curve = pkl.load(f)

        # Combine train and feedback class frequencies
        species_frequencies = None
        activity_frequencies = None
        if self._class_frequencies is not None:
            train_species_frequencies,\
                train_activity_frequencies = self._class_frequencies
            train_species_frequencies = train_species_frequencies.to(device)
            train_activity_frequencies = train_activity_frequencies.to(device)
            
            species_frequencies = train_species_frequencies
            activity_frequencies = train_activity_frequencies

        if feedback_class_frequencies is not None:
            feedback_species_frequencies,\
                feedback_activity_frequencies = feedback_class_frequencies
            feedback_species_frequencies =\
                feedback_species_frequencies.to(device)
            feedback_activity_frequencies =\
                feedback_activity_frequencies.to(device)

            if species_frequencies is None:
                species_frequencies = feedback_species_frequencies
                activity_frequencies = feedback_activity_frequencies
            else:
                species_frequencies =\
                    species_frequencies + feedback_species_frequencies
                activity_frequencies =\
                    activity_frequencies + feedback_activity_frequencies

        if species_frequencies is not None:
            class_frequencies = (species_frequencies, activity_frequencies)
        else:
            class_frequencies = None

        # Train
        pre_ewc_path = '.temp/pre_ewc_path.pth'
        if self.just_finetune == False and self.ewc == True:
            print("Loading EWC Weights")
            self.ewc_calculation = EWC_All_Models(backbone, species_classifier, activity_classifier, train_loader, class_frequencies, self._loss_fn, self._label_smoothing, device, pre_ewc_path)

        if allow_print:
            print('lr:', self._lr)
            progress = tqdm(
                range(start_epoch, self._max_epochs),
                desc=gen_tqdm_description(
                    'Training backbone and classifiers...',
                    train_loss=mean_train_loss,
                    train_accuracy=mean_train_accuracy,
                    val_accuracy=mean_val_accuracy
                ),
                total=self._max_epochs,
                initial=start_epoch
            )
        else:
            progress = range(start_epoch, self._max_epochs)
        
        # # Compute class statistics for pre-trained model
        self._compute_class_statistics_pretrained(backbone, train_dataset, device)

        for epoch in progress:
            if self._patience is not None and\
                    epochs_since_improvement >= self._patience:
                # We haven't improved in several epochs. Time to stop
                # training.
                break
            
            mean_train_loss, species_loss, activity_loss, ewc_penalty, \
            mean_train_accuracy, mean_species_accuracy, mean_activity_accuracy = \
            self._train_epoch(
                backbone,
                feedback_loader,
                species_classifier,
                activity_classifier,
                optimizer,
                device,
                allow_print,
                class_frequencies
            )



            # print(f'Mean Train Loss: {mean_train_loss}, '
            #         f'Species Loss: {species_loss}, '
            #         f'Activity Loss: {activity_loss}, '
            #         f'EWC Penalty: {ewc_penalty}, '
            #         f'Mean Train Accuracy: {mean_train_accuracy}, '
            #         f'Mean Species Accuracy: {mean_species_accuracy}, '
            #         f'Mean Activity Accuracy: {mean_activity_accuracy}')

            if self._root_checkpoint_dir is not None and allow_write:
                checkpoint_dir = get_checkpoint_dir()
                training_checkpoint =\
                    os.path.join(checkpoint_dir, 'training.pth')
                os.makedirs(checkpoint_dir, exist_ok=True)

                sd = {}
                sd['backbone'] = backbone.state_dict()
                sd['species_classifier'] = species_classifier.state_dict()
                sd['activity_classifier'] = activity_classifier.state_dict()
                sd['optimizer'] = optimizer.state_dict()
                sd['start_epoch'] = epoch + 1
                sd['mean_train_loss'] = mean_train_loss
                sd['mean_train_accuracy'] = mean_train_accuracy
                if scheduler is not None:
                    sd['scheduler'] = scheduler.state_dict()
                torch.save(sd, training_checkpoint)

            if root_log_dir is not None and allow_write:
                training_loss_curve[epoch] = mean_train_loss
                training_accuracy_curve[epoch] = mean_train_accuracy
                training_species_loss_curve[epoch] = species_loss
                training_activity_loss_curve[epoch] = activity_loss
                training_mean_species_accuracy_curve[epoch] = mean_species_accuracy
                training_mean_activity_accuracy_curve[epoch] = mean_activity_accuracy
                training_ewc_penalty_curve[epoch] = ewc_penalty
                log_dir = get_log_dir()
                os.makedirs(log_dir, exist_ok=True)
                training_log = os.path.join(log_dir, 'training.pkl')

                with open(training_log, 'wb') as f:
                    sd = {}
                    sd['training_loss_curve'] = training_loss_curve
                    sd['training_accuracy_curve'] = training_accuracy_curve
                    sd['training_species_loss_curve'] = training_species_loss_curve
                    sd['training_activity_loss_curve'] = training_activity_loss_curve
                    sd['training_mean_species_accuracy_curve'] = training_mean_species_accuracy_curve
                    sd['training_mean_activity_accuracy_curve'] = training_mean_activity_accuracy_curve
                    sd['training_ewc_penalty_curve'] = training_ewc_penalty_curve
                    pkl.dump(sd, f)

            # Measure validation accuracy for early stopping / model selection.
            if epoch >= self._min_epochs - 1:
                mean_val_accuracy = self._val_epoch(
                    backbone,
                    val_loader,
                    species_classifier,
                    activity_classifier,
                    device
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict =\
                        deepcopy(backbone.state_dict())
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if self._root_checkpoint_dir is not None and allow_write:
                    checkpoint_dir = get_checkpoint_dir()
                    validation_checkpoint =\
                        os.path.join(checkpoint_dir, 'validation.pth')
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    sd = {}
                    sd['epochs_since_improvement'] = epochs_since_improvement
                    sd['accuracy'] = best_accuracy
                    sd['backbone_state_dict'] =\
                        best_accuracy_backbone_state_dict
                    sd['species_classifier_state_dict'] =\
                        best_accuracy_species_classifier_state_dict
                    sd['activity_classifier_state_dict'] =\
                        best_accuracy_activity_classifier_state_dict
                    sd['mean_val_accuracy'] = mean_val_accuracy
                    torch.save(sd, validation_checkpoint)

                if root_log_dir is not None and allow_write:
                    validation_accuracy_curve[epoch] = mean_val_accuracy
                    log_dir = get_log_dir()
                    os.makedirs(log_dir, exist_ok=True)
                    validation_log = os.path.join(log_dir, 'validation.pkl')
                    
                    with open(validation_log, 'wb') as f:
                        pkl.dump(validation_accuracy_curve, f)

            if allow_print:
                progress.set_description(
                    gen_tqdm_description(
                        'Training backbone and classifiers...',
                        train_loss=mean_train_loss,
                        train_accuracy=mean_train_accuracy,
                        val_accuracy=mean_val_accuracy
                    )
                )
        
        



        # Load the best-accuracy state dict if configured to do so
        # NOTE To save GPU memory, we could temporarily move the models to the
        # CPU before copying or loading their state dicts.
        if self._load_best_after_training:
            backbone.load_state_dict(best_accuracy_backbone_state_dict)
            species_classifier.load_state_dict(
                best_accuracy_species_classifier_state_dict
            )
            activity_classifier.load_state_dict(
                best_accuracy_activity_classifier_state_dict
            )


        # Compute class statistics for pre-trained model
        # self._compute_class_statistics_post_retraining(backbone, feedback_loader, device, allow_print)
        # self._compute_predicted_class_statistics_post_retraining(backbone, species_classifier, activity_classifier, feedback_loader, device, allow_print)
        # import ipdb; ipdb.set_trace()
        # # Perform classifier alignment for both classifiers
        # self._classifier_alignment(species_classifier, 'species', device)
        # self._classifier_alignment(activity_classifier, 'activity', device)


        if allow_print:
            progress.close()

        

    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        # backbone.zero_grad(set_to_none=True)
        # torch.cuda.empty_cache()
        pass
        # classifier.reset()
        # backbone.reset()
        # activation_statistical_model.reset()

    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        activation_stats_training_loader = DataLoader(
            val_known_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=gen_custom_collate(),
            num_workers=0
        )
        backbone.eval()

        all_features = []
        with torch.no_grad():
            for _, _, _, _, batch in activation_stats_training_loader:
                batch = batch.to(device)
                features = activation_statistical_model.compute_features(
                    backbone,
                    batch
                )
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        activation_statistical_model.fit(all_features)
        
class SideTuningClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_dataset,
            val_known_dataset,
            train_feature_file,
            val_feature_file,
            box_transform,
            post_cache_train_transform,
            feedback_batch_size=32,
            retraining_batch_size=32,
            val_interval=20,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            feedback_loss_weight=0.5,
            scheduler_type=SchedulerType.none,
            loss_fn=LossFnEnum.cross_entropy,
            class_frequencies=None):
        self._lr = lr
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._train_feature_file = train_feature_file
        self._val_feature_file = val_feature_file
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._feedback_batch_size = feedback_batch_size
        self._retraining_batch_size = retraining_batch_size
        self._val_interval = val_interval
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._feedback_loss_weight = feedback_loss_weight
        self._scheduler_type = scheduler_type
        self._loss_fn = loss_fn
        self._class_frequencies = class_frequencies
        self._focal_loss = torch.hub.load(
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            alpha=torch.tensor([.75, .25]),
            gamma=2,
            reduction='none',
            force_reload=False
        )

    def _train_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer,
            train_species_labels,
            train_activity_labels,
            train_box_images,
            train_box_backbone_features,
            feedback_species_labels,
            feedback_activity_labels,
            feedback_box_images,
            feedback_box_backbone_features,
            device,
            feedback_class_frequencies):
        # Move to device
        train_species_labels = train_species_labels.to(device)
        train_activity_labels = train_activity_labels.to(device)
        train_box_images = train_box_images.to(device)
        train_box_backbone_features = train_box_backbone_features.to(device)
        feedback_species_labels = feedback_species_labels.to(device)
        feedback_activity_labels = feedback_activity_labels.to(device)
        feedback_box_images = [x.to(device) for x in feedback_box_images]
        feedback_box_backbone_features = [x.to(device) for x in feedback_box_backbone_features]

        # Record feedback box counts, and temporarily flatten feedback box
        # images and backbone features
        feedback_box_counts = [len(x) for x in feedback_box_images]
        feedback_box_images = torch.cat(feedback_box_images, dim=0)
        feedback_box_backbone_features =\
            torch.cat(feedback_box_backbone_features, dim=0)

        # Extract side network box features
        train_box_side_features =\
            backbone.compute_side_features(train_box_images)
        feedback_box_side_features =\
            backbone.compute_side_features(feedback_box_images)

        # Concatenate backbone and side features
        train_box_features = torch.cat(
            (train_box_backbone_features, train_box_side_features),
            dim=1
        )
        feedback_box_features = torch.cat(
            (feedback_box_backbone_features, feedback_box_side_features),
            dim=1
        )

        # Compute logits by passing the features through the appropriate
        # classifiers
        train_species_preds = species_classifier(train_box_features)
        train_activity_preds = activity_classifier(train_box_features)
        feedback_species_logits = species_classifier(feedback_box_features)
        feedback_activity_logits = activity_classifier(feedback_box_features)

        # Compute softmax predictions for feedback data
        feedback_species_preds =\
            torch.nn.functional.softmax(feedback_species_logits, dim=1)
        feedback_activity_preds =\
            torch.nn.functional.softmax(feedback_activity_logits, dim=1)

        # Compute class balancing weights
        species_weights = None
        activity_weights = None
        species_frequencies = None
        activity_frequencies = None
        if self._class_frequencies is not None:
            train_species_frequencies,\
                train_activity_frequencies = self._class_frequencies
            train_species_frequencies = train_species_frequencies.to(device)
            train_activity_frequencies = train_activity_frequencies.to(device)
            
            species_frequencies = train_species_frequencies
            activity_frequencies = train_activity_frequencies

        if feedback_class_frequencies is not None:
            feedback_species_frequencies,\
                feedback_activity_frequencies = feedback_class_frequencies
            feedback_species_frequencies =\
                feedback_species_frequencies.to(device)
            feedback_activity_frequencies =\
                feedback_activity_frequencies.to(device)

            if species_frequencies is None:
                species_frequencies = feedback_species_frequencies
                activity_frequencies = feedback_activity_frequencies
            else:
                species_frequencies =\
                    species_frequencies + feedback_species_frequencies
                activity_frequencies =\
                    activity_frequencies + feedback_activity_frequencies

        if species_frequencies is not None:
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

        # If class balancing, scale the gradients according
        # to class weights
        if species_weights is not None:
            feedback_species_preds = _balance_box_prediction_grads(
                feedback_species_preds,
                species_weights
            )
            feedback_activity_preds = _balance_box_prediction_grads(
                feedback_activity_preds,
                activity_weights
            )
        
        # Re-split feedback predictions for loss computation
        feedback_species_preds = torch.split(
            feedback_species_preds,
            feedback_box_counts,
            dim=0
        )
        feedback_activity_preds = torch.split(
            feedback_activity_preds,
            feedback_box_counts,
            dim=0
        )

        # Compute losses
        if self._loss_fn == LossFnEnum.cross_entropy:
            train_species_loss = torch.nn.functional.cross_entropy(
                train_species_preds,
                train_species_labels,
                weight=species_weights,
                label_smoothing=self._label_smoothing
            )
            train_activity_loss = torch.nn.functional.cross_entropy(
                train_activity_preds,
                train_activity_labels,
                weight=activity_weights,
                label_smoothing=self._label_smoothing
            )
        else:
            ex_species_weights = species_weights[train_species_labels]
            train_species_loss_all = self._focal_loss(
                train_species_preds,
                train_species_labels
            )
            train_species_loss =\
                (train_species_loss_all * ex_species_weights).mean()
            
            ex_activity_weights = activity_weights[train_activity_labels]
            train_activity_loss_all = self._focal_loss(
                train_activity_preds,
                train_activity_labels
            )
            train_activity_loss =\
                (train_activity_loss_all * ex_activity_weights).mean()

        # We have image-level count feedback labels for species
        feedback_species_loss = multiple_instance_count_cross_entropy_dyn(
            feedback_species_preds,
            feedback_species_labels
        )
        # We have image-level presence feedback labels for activities
        feedback_activity_loss = multiple_instance_presence_cross_entropy_dyn(
            feedback_activity_preds,
            feedback_activity_labels
        )

        # Compute non-feedback and feedback loss sums
        non_feedback_loss = train_species_loss + train_activity_loss
        feedback_loss = feedback_species_loss + feedback_activity_loss

        loss = (1 - self._feedback_loss_weight) * non_feedback_loss +\
            self._feedback_loss_weight * feedback_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_species_correct = torch.argmax(train_species_preds, dim=1) == \
            train_species_labels
        train_n_species_correct = int(
            train_species_correct.to(torch.int).sum().detach().cpu().item()
        )

        train_activity_correct = torch.argmax(train_activity_preds, dim=1) == \
            train_activity_labels
        train_n_activity_correct = int(
            train_activity_correct.to(torch.int).sum().detach().cpu().item()
        )
        
        return loss.detach().cpu().item(),\
            train_n_species_correct,\
            train_n_activity_correct

    def _train_epoch(
            self,
            backbone,
            data_loader,
            feedback_feature_loader,
            species_classifier,
            activity_classifier,
            optimizer,
            device,
            feedback_class_frequencies):
        # Set everything to train mode
        backbone.train()
        species_classifier.train()
        activity_classifier.train()
        
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_examples = 0
        n_species_correct = 0
        n_activity_correct = 0

        # For each batch of feedback data...
        train_data_iter = iter(data_loader)
        for feedback_species_labels, feedback_activity_labels, _,\
                feedback_box_images, _, feedback_box_backbone_features in\
                    feedback_feature_loader:
            # Sample a batch of regular training data
            try:
                train_species_labels, train_activity_labels, train_box_images,\
                    train_box_backbone_features = next(train_data_iter)
            except:
                train_data_iter = iter(data_loader)
                train_species_labels, train_activity_labels, train_box_images,\
                    train_box_backbone_features = next(train_data_iter)
            gc.collect()
            # Train on both the feedback and regular training batch
            batch_loss, batch_n_species_correct, batch_n_activity_correct =\
                self._train_batch(
                    backbone,
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    train_species_labels,
                    train_activity_labels,
                    train_box_images,
                    train_box_backbone_features,
                    feedback_species_labels,
                    feedback_activity_labels,
                    feedback_box_images,
                    feedback_box_backbone_features,
                    device,
                    feedback_class_frequencies
                )

            sum_loss += batch_loss
            n_iterations += 1
            n_examples += train_box_images.shape[0]
            n_species_correct += batch_n_species_correct
            n_activity_correct += batch_n_activity_correct

        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, mean_accuracy

    def _val_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images,
            box_backbone_features,
            device):
        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)
        box_backbone_features = box_backbone_features.to(device)

        # Extract side network box features
        box_side_features = backbone.compute_side_features(box_images)

        # Concatenate backbone and side features
        box_features = torch.cat(
            (box_backbone_features, box_side_features),
            dim=1
        )

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_correct = torch.argmax(species_preds, dim=1) == \
            species_labels
        n_species_correct = int(
            species_correct.to(torch.int).sum().detach().cpu().item()
        )

        activity_correct = torch.argmax(activity_preds, dim=1) == \
            activity_labels
        n_activity_correct = int(
            activity_correct.to(torch.int).sum().detach().cpu().item()
        )

        return n_species_correct, n_activity_correct

    def _val_epoch(
            self,
            backbone,
            data_loader,
            species_classifier,
            activity_classifier,
            device):
        with torch.no_grad():
            backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = 0
            n_species_correct = 0
            n_activity_correct = 0

            for species_labels, activity_labels, box_images,\
                    box_backbone_features in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        backbone,
                        species_classifier,
                        activity_classifier,
                        species_labels,
                        activity_labels,
                        box_images,
                        box_backbone_features,
                        device
                    )
                n_examples += box_images.shape[0]
                n_species_correct += batch_n_species_correct
                n_activity_correct += batch_n_activity_correct

            mean_species_accuracy = float(n_species_correct) / n_examples
            mean_activity_accuracy = float(n_activity_correct) / n_examples

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    def train(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration):
        if feedback_sampling_configuration is not None:
            raise ValueError(('Side tuning currently only supports '
                              'feedback_sampling_configuration=None'))

        # Load training and validation backbone features from feature
        # files
        train_box_features, train_species_labels, train_activity_labels =\
            torch.load(
                self._train_feature_file,
                map_location='cpu'
                # map_location= device
            )
        val_box_features, val_species_labels, val_activity_labels =\
            torch.load(
                self._val_feature_file,
                map_location='cpu'
                # map_location= device
            )

        if feedback_dataset is not None:
            feedback_box_counts = [
                feedback_dataset.box_count(i) for\
                    i in range(len(feedback_dataset))
            ]
            feedback_batch_sampler = DistributedRandomBoxImageBatchSampler(
                feedback_box_counts,
                self._retraining_batch_size,
                1,
                0
            )
            feedback_loader = DataLoader(
                feedback_dataset,
                num_workers=0,
                batch_sampler=feedback_batch_sampler,
                collate_fn=gen_custom_collate()
            )
            # Precompute feedback backbone features
            backbone.eval_backbone()
            feedback_box_features = []
            with torch.no_grad():
                for _, _, _, batch_box_images, _ in feedback_loader:
                    # Move batch to device
                    batch_box_images =\
                        [b.to(device) for b in batch_box_images]

                    # Get list of per-image box counts
                    batch_box_counts = [len(x) for x in batch_box_images]

                    # Flatten box images and compute features
                    flattened_box_images = torch.cat(batch_box_images, dim=0)
                    batch_box_features = backbone.compute_backbone_features(flattened_box_images)
                    batch_box_features = batch_box_features.detach().cpu()

                    # Use batch_box_counts to split the computed features back
                    # to per-image feature tensors and concatenate to
                    # feedback_box_features
                    feedback_box_features += torch.split(
                        batch_box_features,
                        batch_box_counts,
                        dim=0
                    )

            # Construct FeatureConcatDataset for the feedback dataset
            feedback_feature_batch_sampler =\
                DistributedRandomBoxImageBatchSampler(
                    feedback_box_counts,
                    self._retraining_batch_size,
                    1,
                    0
                )
            feedback_feature_loader = DataLoader(
                FeatureConcatDataset(
                    feedback_dataset,
                    feedback_box_features
                ),
                num_workers=0,
                batch_sampler=feedback_feature_batch_sampler,
                collate_fn=gen_custom_collate()
            )
        else:
            feedback_feature_loader = None

        # Construct training and validation
        # FeatureConcatDataset objects from known features.

        known_train_dataset = FeatureConcatDataset(
            FlattenedBoxImageDataset(self._train_dataset),
            train_box_features
        )
        known_val_dataset = FeatureConcatDataset(
            FlattenedBoxImageDataset(self._val_known_dataset),
            val_box_features
        )

        # For now, we just have a single loader for training and a single loader
        # for validation (until feedback is implemented)
        train_dataset = known_train_dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._retraining_batch_size,
            shuffle=True,
            num_workers=0
        )
        val_dataset = known_val_dataset
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._retraining_batch_size,
            shuffle=False,
            num_workers=0
        )

        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(backbone.retrainable_parameters())\
                + list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            self._lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Init scheduler to None. It will be constructed after loading
        # the optimizer state dict, or after failing to do so
        scheduler = None
        scheduler_ctor = self._scheduler_type.ctor()

        # Define convergence parameters (early stopping + model selection)
        start_epoch = 0
        epochs_since_improvement = 0
        best_accuracy = None
        best_accuracy_backbone_state_dict = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None
        mean_train_loss = None
        mean_train_accuracy = None
        mean_val_accuracy = None

        # If we didn't load an optimizer state dict, and so the scheduler
        # hasn't been constructed yet, then construct it
        if scheduler_ctor is not None and scheduler is None:
            scheduler = scheduler_ctor(
                optimizer,
                self._max_epochs
            )
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}

        def get_log_dir():
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                'end-to-end-trainer',
                f'lr={self._lr}',
                f'label_smoothing={self._label_smoothing:.2f}'
            )

        if root_log_dir is not None and allow_write:
            log_dir = get_log_dir()
            training_log = os.path.join(log_dir, 'training.pkl')
            validation_log =\
                os.path.join(log_dir, 'validation.pkl')

            if os.path.exists(training_log):
                with open(training_log, 'rb') as f:
                    sd = pkl.load(f)
                    training_loss_curve = sd['training_loss_curve']
                    training_accuracy_curve = sd['training_accuracy_curve']

            if os.path.exists(validation_log):
                with open(validation_log, 'rb') as f:
                    validation_accuracy_curve = pkl.load(f)

        # Train
        progress = tqdm(
            range(start_epoch, self._max_epochs),
            desc=gen_tqdm_description(
                'Training backbone and classifiers...',
                train_loss=mean_train_loss,
                train_accuracy=mean_train_accuracy,
                val_accuracy=mean_val_accuracy
            ),
            total=self._max_epochs,
            initial=start_epoch
        )
        for epoch in progress:
            if self._patience is not None and\
                    epochs_since_improvement >= self._patience:
                # We haven't improved in several epochs. Time to stop
                # training.
                break

            # Train for one full epoch
            mean_train_loss, mean_train_accuracy = self._train_epoch(
                backbone,
                train_loader,
                feedback_feature_loader,
                species_classifier,
                activity_classifier,
                optimizer,
                device,
                feedback_class_frequencies
            )

            if root_log_dir is not None and allow_write:
                training_loss_curve[epoch] = mean_train_loss
                training_accuracy_curve[epoch] = mean_train_accuracy
                log_dir = get_log_dir()
                os.makedirs(log_dir, exist_ok=True)
                training_log = os.path.join(log_dir, 'training.pkl')
                
                with open(training_log, 'wb') as f:
                    sd = {}
                    sd['training_loss_curve'] = training_loss_curve
                    sd['training_accuracy_curve'] = training_accuracy_curve
                    pkl.dump(sd, f)

            # Measure validation accuracy for early stopping / model selection.
            if epoch >= self._min_epochs - 1 and\
                    (epoch + 1) % self._val_interval == 0:
                mean_val_accuracy = self._val_epoch(
                    backbone,
                    val_loader,
                    species_classifier,
                    activity_classifier,
                    device
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict =\
                        deepcopy(backbone.state_dict())
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if root_log_dir is not None and allow_write:
                    validation_accuracy_curve[epoch] = mean_val_accuracy
                    log_dir = get_log_dir()
                    os.makedirs(log_dir, exist_ok=True)
                    validation_log = os.path.join(log_dir, 'validation.pkl')
                    
                    with open(validation_log, 'wb') as f:
                        pkl.dump(validation_accuracy_curve, f)

            progress.set_description(
                gen_tqdm_description(
                    'Training backbone and classifiers...',
                    train_loss=mean_train_loss,
                    train_accuracy=mean_train_accuracy,
                    val_accuracy=mean_val_accuracy
                )
            )

        progress.close()

        # Load the best-accuracy state dicts
        # NOTE To save GPU memory, we could temporarily move the models to the
        # CPU before copying or loading their state dicts.
        # NOTE we could also make the state dicts here a little more efficient
        # by only saving and loading the state dict of the side network, rather
        # than working with the fixed backbone as well.
        backbone.load_state_dict(best_accuracy_backbone_state_dict)
        species_classifier.load_state_dict(
            best_accuracy_species_classifier_state_dict
        )
        activity_classifier.load_state_dict(
            best_accuracy_activity_classifier_state_dict
        )

    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            activation_statistical_model):
        pass
        # Reset only the side network's weights
        # backbone.reset()

        # Update classifier's bottleneck dim to account for side network's
        # features before resetting
        # classifier.reset(bottleneck_dim=512)

    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model,
            val_known_dataset,
            device):
        pass

def get_transforms(augmentation):
    box_transform = ResizePad(224)
    normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    augmentation_ctor = augmentation.ctor()
    post_cache_train_transform =\
        Compose((augmentation_ctor(), normalize))
    post_cache_val_transform = normalize
    return box_transform, post_cache_train_transform, post_cache_val_transform


def get_datasets(
        data_root,
        train_csv_path,
        val_csv_path,
        n_species_cls,
        n_activity_cls,
        static_label_mapper,
        dynamic_label_mapper,
        box_transform,
        post_cache_train_transform,
        post_cache_val_transform,
        root_cache_dir=None,
        n_known_val=4068):
    if root_cache_dir is not None:
        train_cache_dir = os.path.join(root_cache_dir, 'train')
        val_cache_dir = os.path.join(root_cache_dir, 'val')
    else:
        train_cache_dir = None
        val_cache_dir = None

    raw_train_dataset = BoxImageDataset(
        name = 'Custom',
        data_root = data_root,
        csv_path = train_csv_path,
        training = True,
        n_species_cls=n_species_cls,
        n_activity_cls=n_activity_cls,
        label_mapper=dynamic_label_mapper,
        box_transform=box_transform,
        cache_dir=train_cache_dir,
        write_cache=True
    )
    raw_train_dataset.commit_cache()

    val_known_indices_gen = np.random.Generator(np.random.PCG64(0))
    val_known_indices = val_known_indices_gen.choice(
        list(range(len(raw_train_dataset))),
        size=n_known_val,
        replace=False
    ).tolist()
    val_known_indices_set = set(val_known_indices)
    training_indices = [x for x in range(len(raw_train_dataset)) if\
        not x in val_known_indices_set]
    
    val_known_dataset = TransformingBoxImageDataset(
        Subset(raw_train_dataset, val_known_indices),
        post_cache_val_transform
    )
    train_dataset = TransformingBoxImageDataset(
        Subset(raw_train_dataset, training_indices),
        post_cache_train_transform
    )

    raw_val_dataset = BoxImageDataset(
        name = 'Custom',
        data_root = data_root,
        csv_path = val_csv_path,
        training = False,
        n_species_cls=n_species_cls,
        n_activity_cls=n_activity_cls,
        label_mapper=static_label_mapper,
        box_transform=box_transform,
        cache_dir=val_cache_dir,
        write_cache=True
    )
    raw_val_dataset.commit_cache()

    val_dataset = ConcatDataset((
        val_known_dataset,
        TransformingBoxImageDataset(
            raw_val_dataset,
            post_cache_val_transform
        )
    ))

    return train_dataset,\
        val_known_dataset,\
        val_dataset


def compute_features(
        backbone,
        root_save_dir,
        box_transform,
        post_cache_train_transform,
        train_dataset,
        val_known_dataset,
        retraining_batch_size):
    backbone.eval()
    print('Computing Features Set....')
    flattened_train_dataset = FlattenedBoxImageDataset(train_dataset)
    train_loader = DataLoader(
        flattened_train_dataset,
        batch_size=retraining_batch_size,
        shuffle=False,
        num_workers=0
    )
    os.makedirs(root_save_dir, exist_ok=True)

    # Construct validation loaders for early stopping / model selection.
    # I'm assuming our model selection strategy will be based solely on the
    # validation classification accuracy and not based on novelty detection
    # capabilities in any way. Otherwise, we can use the novel validation
    # data to measure novelty detection performance. These currently aren't
    # being stored (except in a special form for the logistic regressions),
    # so we'd have to modify __init__().

    flattened_val_dataset = FlattenedBoxImageDataset(val_known_dataset)
    val_loader = DataLoader(
        flattened_val_dataset,
        batch_size=retraining_batch_size,
        shuffle=False,
        num_workers=0
    )

    save_dir = os.path.join(
        root_save_dir,
        box_transform.path(),
        post_cache_train_transform.path(),
    )

    training_features_path = os.path.join(save_dir, 'training.pth')
    os.makedirs(training_features_path, exist_ok=True)
    validation_features_path = os.path.join(save_dir, 'validation.pth')
    os.makedirs(validation_features_path, exist_ok=True)

    device = backbone.device

    train_box_features = []
    train_species_labels = []
    train_activity_labels = []

    with torch.no_grad():
        for species_labels, activity_labels, box_images in tqdm(train_loader, desc="Computing Training Set Features"):
            species_labels = species_labels.to(device)
            activity_labels = activity_labels.to(device)
            box_images = box_images.to(device)

            box_features = backbone(box_images)

            train_box_features.append(box_features)
            train_species_labels.append(species_labels)
            train_activity_labels.append(activity_labels)
            break

        train_box_features = torch.cat(train_box_features, dim=0)
        train_species_labels = torch.cat(train_species_labels, dim=0)
        train_activity_labels = torch.cat(train_activity_labels, dim=0)

    val_box_features = []
    val_species_labels = []
    val_activity_labels = []

    with torch.no_grad():
        for species_labels, activity_labels, box_images in tqdm(val_loader, desc="Computing Validation Set Features"):
            species_labels = species_labels.to(device)
            activity_labels = activity_labels.to(device)
            box_images = box_images.to(device)

            box_features = backbone(box_images)

            val_box_features.append(box_features)
            val_species_labels.append(species_labels)
            val_activity_labels.append(activity_labels)
            break

        val_box_features = torch.cat(val_box_features, dim=0)
        val_species_labels = torch.cat(val_species_labels, dim=0)
        val_activity_labels = torch.cat(val_activity_labels, dim=0)
    torch.save(
        (train_box_features, train_species_labels, train_activity_labels),
        training_features_path
    )
    torch.save(
        (val_box_features, val_species_labels, val_activity_labels),
        validation_features_path
    )



class TuplePredictorTrainer:
    def __init__(
            self,
            train_dataset,
            val_known_dataset,
            val_dataset,
            box_transform,
            post_cache_train_transform,
            n_species_cls,
            n_activity_cls,
            dynamic_label_mapper,
            classifier_trainer):
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._val_dataset = val_dataset
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._n_species_cls = n_species_cls
        self._n_activity_cls = n_activity_cls
        self._dynamic_label_mapper = dynamic_label_mapper
        self._classifier_trainer = classifier_trainer
        self._feedback_data = []
        self._n_feedback_examples = 0

    def add_feedback_data(self, data_root, csv_path):
        # Construct feedback dataset
        new_novel_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = csv_path,
            training = True,
            n_species_cls=self._n_species_cls,
            n_activity_cls=self._n_activity_cls,
            label_mapper=self._dynamic_label_mapper,
            box_transform=self._box_transform
        )
        # Add in-memory caching layer (feedback data shouldn't be cached
        # to disk)
        new_novel_dataset = BoxImageMemoryDataset(new_novel_dataset)
        # Apply post-cache train transforms (augment, normalize)
        new_novel_dataset = TransformingBoxImageDataset(
            new_novel_dataset,
            self._post_cache_train_transform
        )

        # Put new feedback data in list
        self._feedback_data.append(new_novel_dataset)
        self._n_feedback_examples += len(new_novel_dataset)

    def n_feedback_examples(self):
        return self._n_feedback_examples

    # Should be called before train_novelty_detection_module(), except when
    # training for the very first time manually. This prepares the
    # backbone, classifier, and novelty type logistic regressions for
    # retraining.
    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model):
        # Reset the classifiers (and possibly certain backbone components,
        # depending on the classifier retraining method) if appropriate
        self._classifier_trainer.prepare_for_retraining(
            backbone,
            classifier,
            activation_statistical_model
        )
        # pass
        # Reset the confidence calibrator
        confidence_calibrator.reset()

        # Reset logistic regressions and statistical model
        novelty_type_classifier.reset()

    def calibrate_temperature_scalers(
            self,
            device,
            backbone,
            species_classifier,
            activity_classifier,
            species_calibrator,
            activity_calibrator,
            allow_print):
        cal_loader = DataLoader(
            self._val_known_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=gen_custom_collate(),
            num_workers=0
        )

        # Set everything to eval mode for calibration, except the calibrators
        backbone.eval()
        species_classifier.eval()
        activity_classifier.eval()
        species_calibrator.train()
        activity_calibrator.train()

        if allow_print:
            print('Calibrating temperature scalers...')
        start = time.time()
        # Extract logits to fit confidence calibration temperatures
        with torch.no_grad():
            species_logits = []
            species_labels = []
            activity_logits = []
            activity_labels = []
            for batch_species_labels, batch_activity_labels, _, box_images, _ in\
                    cal_loader:
                # Flatten the boxes across images and extract per-image box
                # counts.
                box_counts = [x.shape[0] for x in box_images]
                flattened_box_images = torch.cat(box_images, dim=0)

                # Move to device
                batch_species_labels = batch_species_labels.to(device)
                batch_activity_labels = batch_activity_labels.to(device)
                flattened_box_images = flattened_box_images.to(device)

                # Construct per-box labels.
                one_hot_species_labels = torch.argmax(batch_species_labels, dim=1)
                flattened_species_labels = torch.cat([
                    torch.full((box_count,), species_label, device=device)\
                        for species_label, box_count in\
                            zip(one_hot_species_labels, box_counts)
                ])
                one_hot_activity_labels = torch.argmax(batch_activity_labels.float(), dim=1)
                flattened_activity_labels = torch.cat([
                    torch.full((box_count,), activity_label, device=device)\
                        for activity_label, box_count in\
                            zip(one_hot_activity_labels, box_counts)
                ])

                # Compute box features and logits
                box_features = backbone(flattened_box_images)
                batch_species_logits = species_classifier(box_features)
                species_logits.append(batch_species_logits)
                species_labels.append(flattened_species_labels)
                batch_activity_logits = activity_classifier(box_features)
                activity_logits.append(batch_activity_logits)
                activity_labels.append(flattened_activity_labels)

            species_logits = torch.cat(species_logits, dim = 0)
            species_labels = torch.cat(species_labels, dim = 0)
            activity_logits = torch.cat(activity_logits, dim = 0)
            activity_labels = torch.cat(activity_labels, dim = 0)

        optimizer = torch.optim.SGD(
            list(species_calibrator.parameters()) +\
                list(activity_calibrator.parameters()),
            0.001,
            momentum=0.9)

        progress = tqdm(range(10000), desc = 'Training calibrators...')
        for epoch in progress:
            scaled_species_logits = species_calibrator(species_logits)
            species_loss = torch.nn.functional.cross_entropy(
                scaled_species_logits,
                species_labels
            )
            scaled_activity_logits = activity_calibrator(activity_logits)
            activity_loss = torch.nn.functional.cross_entropy(
                scaled_activity_logits,
                activity_labels
            )
            loss = species_loss + activity_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_description(
                f'Training calibrators... | Loss: {loss.detach().cpu().item()}'
            )

        end = time.time()
        t = end - start
        if allow_print:
            print(f'Took {t} seconds')

    def fit_logistic_regression(
            self,
            logistic_regression,
            scores,
            labels,
            epochs,
            allow_print):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            logistic_regression.parameters(),
            lr = 0.01,
            momentum = 0.9
        )
        logistic_regression.fit_standardization_statistics(scores)

        loss_item = None
        if allow_print:
            progress = tqdm(
                range(epochs),
                desc=gen_tqdm_description(
                    'Fitting logistic regression...',
                    loss=loss_item
                )
            )
        else:
            progress = range(epochs)
        for epoch in progress:
            optimizer.zero_grad()
            logits = logistic_regression(scores)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            loss_item = loss.detach().cpu().item()
            if allow_print:
                progress.set_description(
                    gen_tqdm_description(
                        'Fitting logistic regression...',
                        loss=loss_item
                    )
                )
        if allow_print:
            progress.close()

    def train_novelty_type_logistic_regressions(
            self,
            device,
            backbone,
            species_classifier,
            activity_classifier,
            novelty_type_classifier,
            activation_statistical_model,
            scorer,
            allow_print):
        # Set the backbone and classifiers to eval(), but set the logistic
        # regressions to train()
        backbone.eval()
        species_classifier.eval()
        activity_classifier.eval()
        novelty_type_classifier.train()

        val_loader = DataLoader(
            self._val_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=gen_custom_collate(),
            num_workers=0
        )

        if allow_print:
            print('Training novelty type classifier...')
        start = time.time()
        with torch.no_grad():
            # Extract novelty scores and labels
            scores = []
            labels = []
            for _, _, batch_labels, box_images, whole_images in val_loader:
                # Flatten the boxes across images and extract per-image box
                # counts.
                box_counts = [x.shape[0] for x in box_images]
                flattened_box_images = torch.cat(box_images, dim=0)

                # Move to device
                flattened_box_images = flattened_box_images.to(device)
                whole_images = whole_images.to(device)
                batch_labels = batch_labels.to(device)

                # Extract box features
                box_features = backbone(flattened_box_images)

                # Compute logits
                species_logits = species_classifier(box_features)
                activity_logits = activity_classifier(box_features)

                # Compute whole-image features activation statistic scores
                whole_image_features =\
                    activation_statistical_model.compute_features(
                        backbone,
                        whole_images
                    )

                # Compute novelty scores
                batch_scores = scorer.score(
                    species_logits,
                    activity_logits,
                    whole_image_features,
                    box_counts
                )

                scores.append(batch_scores)
                labels.append(batch_labels)
            
            scores = torch.cat(scores, dim = 0)
            labels = torch.cat(labels, dim = 0)

        # Fit the logistic regression
        self.fit_logistic_regression(
            novelty_type_classifier,
            scores,
            labels,
            3000,
            allow_print
        )
        end = time.time()
        t = end - start
        if allow_print:
            print(f'Took {t} seconds')

    def train_novelty_detection_module(
            self,
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model,
            scorer,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_sampling_configuration,
            root_log_dir=None,
            model_unwrap_fn=None):
        species_classifier = classifier.species_classifier
        activity_classifier = classifier.activity_classifier
        species_calibrator = confidence_calibrator.species_calibrator
        activity_calibrator = confidence_calibrator.activity_calibrator
        
        # Retrain the backbone and classifiers
        feedback_dataset = None
        feedback_class_frequencies = None
        if len(self._feedback_data) > 0:
            feedback_dataset = ConcatDataset(self._feedback_data)
            feedback_label_dataset = feedback_dataset.label_dataset()

            species_labels = []
            activity_labels = []
            for species_label, activity_label, _ in feedback_label_dataset:
                species_labels.append(species_label.to(torch.long))
                activity_labels.append(activity_label.to(torch.long))
            species_labels = torch.stack(species_labels, dim=0).to(device)
            activity_labels = torch.stack(activity_labels, dim=0).to(device)

            species_frequencies = species_labels.sum(dim=0)
            activity_frequencies = activity_labels.sum(dim=0)

            feedback_class_frequencies = (
                species_frequencies,
                activity_frequencies
            )

        feedback_sampling_configuration_ctor =\
            feedback_sampling_configuration.ctor()

        self._classifier_trainer.train(
            backbone,
            species_classifier,
            activity_classifier,
            root_log_dir,
            feedback_dataset,
            device,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            feedback_class_frequencies,
            feedback_sampling_configuration_ctor()
        )

        # Unwrap backbone and classifiers (e.g., from DDP adapters) if
        # appropriate
        if model_unwrap_fn is not None:
            backbone, classifier = model_unwrap_fn(backbone, classifier)
            species_classifier = classifier.species_classifier
            activity_classifier = classifier.activity_classifier

        self._classifier_trainer.fit_activation_statistics(
            backbone,
            activation_statistical_model,
            self._val_known_dataset,
            device
        )

        # Retrain the classifier's temperature scaling calibrators
        self.calibrate_temperature_scalers(
            device,
            backbone,
            species_classifier,
            activity_classifier,
            species_calibrator,
            activity_calibrator,
            allow_print
        )

        # Retrain the logistic regressions
        self.train_novelty_type_logistic_regressions(
            device,
            backbone,
            species_classifier,
            activity_classifier,
            novelty_type_classifier,
            activation_statistical_model,
            scorer,
            allow_print
        )
