import itertools
from copy import deepcopy
import numpy as np
import os
from enum import Enum
import pickle as pkl
from abc import ABC, abstractmethod

from tqdm import tqdm
from torch.utils.data import\
    Dataset,\
    DataLoader,\
    Subset as TorchSubset,\
    ConcatDataset as TorchConcatDataset
import torch
from torch.nn import Module
from torch.optim.lr_scheduler import CosineAnnealingLR

from boximagedataset import BoxImageDataset
from utils import custom_collate, gen_tqdm_description
from labelmapping import LabelMapper
from transforms import\
    Compose,\
    Normalize,\
    ResizePad,\
    RandAugment,\
    RandomHorizontalFlip,\
    NoOpTransform


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
    # For each image
    for img_predictions, img_targets in zip(predictions, targets):
        # We're working with img_predictions of shape [M_i, K] and img_targets
        # of shape [K]. Index the predictions whose classes are present in
        # the ground truth labels
        present_classes = torch.nonzero(img_targets)[0]
        remaining_predictions = img_predictions[:, present_classes]
        remaining_targets = img_targets[present_classes]
        combination_pred_running_product = None

        # The general strategy is as follows: remaining_predictions will be
        # iteratively updated, and in general its shape will be
        # [B, K, *c], where *c = (C_{N-1}, C_{N-2}, ..., C_1). C_i denotes
        # the number of combinations selected when considering present class
        # i (i.e., torch.nonzero(image_targets)[0][i]). The combination
        # dimensions are prepended rather than appended to simplify broadcasting
        # to keep track of the running cross product predictions.

        # At each iteration, we'll consider combinations for the class whose
        # predictions are at remaining_predictions[:, 0] and whose counts are
        # at target[0]. Those index-0 dimensions are sliced off at each
        # iteration until all present classes have been considered.

        # Note that the last class is trivial since there's only one
        # possible combination: all remaining boxes must be assigned the one
        # remaining label.

        # For each present class
        for _ in range(len(present_classes)):
            ## Consider all of the ways in which we can assign this class label
            ## to the correct number of the remaining boxes
            cur_combinations = torch.combinations(
                torch.arange(
                    len(remaining_predictions),
                    device=targets.device
                ),
                r=remaining_targets[0]
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
                *(len(cur_combinations), remaining_predictions.shape[1]),
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
                remaining_predictions.shape[1],
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
        sum_prob = combination_pred_running_products.sum()

        ## Compute per-image loss
        losses.append(-torch.log(sum_prob))

    ## Aggregate per-image losses and return
    losses = torch.stack(losses, dim=0)
    return losses.mean()


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
def multiple_instance_count_cross_entropy(predictions, targets):
    losses = []
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
        # but it's left out the notation for brevity.
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
        absent_predictions = img_predictions[:, ~img_targets]
        prob_others_absent = torch.prod(1 - absent_predictions.sum(dim=1))

        # Step 2: Filter out present-class predictions and condition on
        # the event <others absent>
        # TODO handle case where denominator == 0. This might require moving
        # computations to the log space
        present_predictions = img_predictions[:, img_targets]
        cond_present_predictions = present_predictions / \
            present_predictions.sum(dim=1, keepdim=True)

        # Step 3: For each i in 1, ..., N, alternate adding and subtracting
        # the relevant N choose i conjunction probabilities to compute
        # P(A, B, or C absent)
        prob_any_absent = 0
        add_iteration = True
        for i in range(1, cond_present_predictions.shape[1] + 1):
            # Compute the class indices for the N choose i conjunctions
            cur_combinations = torch.combinations(
                torch.arange(
                    cond_present_predictions.shape[1],
                    device=targets.device
                ),
                r=i
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

        # Step 4: We have
        # P(any of the ground-truth present classes are absent | others absent).
        # Compute the compliment to get
        # P(present classes are predicted present | others absent).
        prob_all_present = 1 - prob_any_absent

        # Step 5: Compute P(A, B, ... C present, others Absent)
        #       = P(A, B, ..., C present | others absent) * P(others absent)
        prob_conform = prob_all_present * prob_others_absent

        ## Compute per-image loss
        losses.append(-torch.log(prob_conform))

    ## Aggregate per-image losses and return
    losses = torch.stack(losses, dim=0)
    return losses.mean()


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


class BackboneTrainingType(Enum):
    end_to_end = 'end-to-end'
    classifiers = 'classifiers'
    side_tuning = 'side-tuning'

    def __str__(self):
        return self.value


class ClassifierTrainer(ABC):
    @abstractmethod
    def train(self, species_classifier, activity_classifier, root_log_dir):
        return NotImplemented

    def prepare_for_retraining(
            self,
            classifier):
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
        one_hot_activity_label = torch.argmax(activity_labels, dim=0)
        box_image = box_images[local_box_idx]

        return one_hot_species_label, one_hot_activity_label, box_image


'''
Wraps a dataset around a FlattenedBoxImageDataset and a precomputed feature
tensor for the whole dataset. Indexing at i returns the underlying data
from the FlattenedBoxImageDataset as well as sub-tensor of the feature tensor
indexed at i along dim 0 (most likely the ith datapoint's feature vector).
'''
class FeatureConcatFlattenedBoxImageDataset(Dataset):
    def __init__(self, dataset, box_features):
        self._dataset = dataset
        self._box_features = box_features

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        species_label, activity_label, box_image = self._dataset[idx]
        box_features = self._box_features[idx]
        return species_label, activity_label, box_image, box_features


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


def fit_logistic_regression(logistic_regression, scores, labels, epochs = 3000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        logistic_regression.parameters(),
        lr = 0.01,
        momentum = 0.9
    )
    logistic_regression.fit_standardization_statistics(scores)
    
    loss_item = None
    progress = tqdm(
        range(epochs),
        desc=gen_tqdm_description(
            'Fitting logistic regression...',
            loss=loss_item
        )
    )
    for epoch in progress:
        optimizer.zero_grad()
        logits = logistic_regression(scores)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        loss_item = loss.detach().cpu().item()
        progress.set_description(
            gen_tqdm_description(
                'Fitting logistic regression...',
                loss=loss_item
            )
        )
    progress.close()


class LogitLayerClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            lr,
            train_feature_file,
            val_feature_file,
            box_transform,
            post_cache_train_transform,
            device,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0):
        self._lr = lr
        self._train_feature_file = train_feature_file
        self._val_feature_file = val_feature_file
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._device = device
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing

    def _train_epoch(
            self,
            box_features,
            species_labels,
            activity_labels,
            species_classifier,
            activity_classifier,
            optimizer):
        # Set everything to train mode
        species_classifier.train()
        activity_classifier.train()

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_loss = torch.nn.functional.cross_entropy(
            species_preds,
            species_labels,
            label_smoothing=self._label_smoothing
        )
        activity_loss = torch.nn.functional.cross_entropy(
            activity_preds,
            activity_labels,
            label_smoothing=self._label_smoothing
        )

        loss = species_loss + activity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return loss.detach().cpu().item(), mean_accuracy

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
            species_classifier,
            activity_classifier,
            root_log_dir):
        # TODO new feedback setup will require a separate dataloader since
        # feedback data has several multiple instance issues and thus requires
        # image-level labels and special loss functions. Pass that dataloader
        # in as an argument

        # TODO Class balancing

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
                map_location=self._device
            )
        val_box_features, val_species_labels, val_activity_labels =\
            torch.load(
                self._val_feature_file,
                map_location=self._device
            )

        # TODO Precompute feedback backbone features

        # TODO Construct custom feedback dataset; separate class from known data
        # datasets because of the multiple instance problems. Weighted sampling
        # will come from explicitly drawing batches from each loader (known
        # and feedback) separately, and then balancing them mid-iteration

        # TODO Construct custom weighted sampling concatenated dataset between
        # known and feedback data, and use that rather than the full-batch
        # approach

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
                species_classifier,
                activity_classifier,
                optimizer
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
            classifier):
        classifier.reset()

class EndToEndClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            backbone,
            lr,
            train_dataset,
            val_known_dataset,
            box_transform,
            post_cache_train_transform,
            retraining_batch_size=32,
            train_sampler_fn=None,
            root_checkpoint_dir=None,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            scheduler_type=SchedulerType.none,
            allow_write=False):
        self._backbone = backbone
        self._lr = lr
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._retraining_batch_size = retraining_batch_size
        self._train_sampler_fn = train_sampler_fn
        self._root_checkpoint_dir = root_checkpoint_dir
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._scheduler_type = scheduler_type
        self._allow_write = allow_write

    def _train_batch(
            self,
            species_classifier,
            activity_classifier,
            optimizer,
            species_labels,
            activity_labels,
            box_images):
        # Determine the device to use based on the backbone's fc weights
        device = self._backbone.device

        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = self._backbone(box_images)

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_loss = torch.nn.functional.cross_entropy(
            species_preds,
            species_labels,
            label_smoothing=self._label_smoothing
        )
        activity_loss = torch.nn.functional.cross_entropy(
            activity_preds,
            activity_labels,
            label_smoothing=self._label_smoothing
        )

        loss = species_loss + activity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        
        return loss.detach().cpu().item(),\
            n_species_correct,\
            n_activity_correct

    def _train_epoch(
            self,
            data_loader,
            species_classifier,
            activity_classifier,
            optimizer):
        # Set everything to train mode
        self._backbone.train()
        species_classifier.train()
        activity_classifier.train()
        
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_examples = 0
        n_species_correct = 0
        n_activity_correct = 0

        for species_labels, activity_labels, box_images in data_loader:
            batch_loss, batch_n_species_correct, batch_n_activity_correct =\
                self._train_batch(
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    species_labels,
                    activity_labels,
                    box_images
                )

            sum_loss += batch_loss
            n_iterations += 1
            n_examples += box_images.shape[0]
            n_species_correct += batch_n_species_correct
            n_activity_correct += batch_n_activity_correct

        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, mean_accuracy

    def _val_batch(
            self,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images):
        device = self._backbone.device

        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)

        # Extract box features
        box_features = self._backbone(box_images)

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
            data_loader,
            species_classifier,
            activity_classifier):
        with torch.no_grad():
            self._backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = 0
            n_species_correct = 0
            n_activity_correct = 0

            for species_labels, activity_labels, box_images in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        species_classifier,
                        activity_classifier,
                        species_labels,
                        activity_labels,
                        box_images
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
            species_classifier,
            activity_classifier,
            root_log_dir):
        # TODO new feedback setup will require a separate dataloader since
        # feedback data has several multiple instance issues and thus requires
        # image-level labels and special loss functions

        # TODO Class balancing
        train_dataset = FlattenedBoxImageDataset(self._train_dataset)
        if self._train_sampler_fn is not None:
            train_sampler = self._train_sampler_fn(train_dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self._retraining_batch_size,
                num_workers=2,
                sampler=train_sampler
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self._retraining_batch_size,
                shuffle=True,
                num_workers=2
            )

        # Construct validation loaders for early stopping / model selection.
        # I'm assuming our model selection strategy will be based solely on the
        # validation classification accuracy and not based on novelty detection
        # capabilities in any way. Otherwise, we can use the novel validation
        # data to measure novelty detection performance. These currently aren't
        # being stored (except in a special form for the logistic regressions),
        # so we'd have to modify __init__().
        val_dataset = FlattenedBoxImageDataset(self._val_known_dataset)
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._retraining_batch_size,
            shuffle=False,
            num_workers=2
        )

        # Retrain the backbone and classifiers
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(self._backbone.parameters())\
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
                    map_location=self._backbone.device
                )
                self._backbone.load_state_dict(sd['backbone'])
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
                    map_location=self._backbone.device
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

        if root_log_dir is not None and self._allow_write:
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

            if train_loader.sampler is not None:
                # Set the sampler epoch for shuffling when running in
                # distributed mode
                train_loader.sampler.set_epoch(epoch)

            # Train for one full epoch
            mean_train_loss, mean_train_accuracy = self._train_epoch(
                train_loader, 
                species_classifier,
                activity_classifier,
                optimizer
            )

            if self._root_checkpoint_dir is not None and self._allow_write:
                checkpoint_dir = get_checkpoint_dir()
                training_checkpoint =\
                    os.path.join(checkpoint_dir, 'training.pth')
                os.makedirs(checkpoint_dir, exist_ok=True)

                sd = {}
                sd['backbone'] = self._backbone.state_dict()
                sd['species_classifier'] = species_classifier.state_dict()
                sd['activity_classifier'] = activity_classifier.state_dict()
                sd['optimizer'] = optimizer.state_dict()
                sd['start_epoch'] = epoch + 1
                sd['mean_train_loss'] = mean_train_loss
                sd['mean_train_accuracy'] = mean_train_accuracy
                if scheduler is not None:
                    sd['scheduler'] = scheduler.state_dict()
                torch.save(sd, training_checkpoint)

            if root_log_dir is not None and self._allow_write:
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
                    val_loader,
                    species_classifier,
                    activity_classifier
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict =\
                        deepcopy(self._backbone.state_dict())
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if self._root_checkpoint_dir is not None and self._allow_write:
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

                if root_log_dir is not None and self._allow_write:
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
        self._backbone.load_state_dict(best_accuracy_backbone_state_dict)
        species_classifier.load_state_dict(
            best_accuracy_species_classifier_state_dict
        )
        activity_classifier.load_state_dict(
            best_accuracy_activity_classifier_state_dict
        )

    def prepare_for_retraining(
            self,
            classifier):
        classifier.reset()


class SideTuningClassifierTrainer(ClassifierTrainer):
    def __init__(
            self,
            side_tuning_backbone,
            lr,
            train_dataset,
            val_known_dataset,
            train_feature_file,
            val_feature_file,
            box_transform,
            post_cache_train_transform,
            retraining_batch_size=32,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            scheduler_type=SchedulerType.none,
            allow_write=False):
        self._backbone = side_tuning_backbone
        self._lr = lr
        self._train_dataset = train_dataset
        self._val_known_dataset = val_known_dataset
        self._train_feature_file = train_feature_file
        self._val_feature_file = val_feature_file
        self._box_transform = box_transform
        self._post_cache_train_transform = post_cache_train_transform
        self._retraining_batch_size = retraining_batch_size
        self._patience = patience
        self._min_epochs = min_epochs
        self._max_epochs = max_epochs
        self._label_smoothing = label_smoothing
        self._scheduler_type = scheduler_type
        self._allow_write = allow_write

    def _train_batch(
            self,
            species_classifier,
            activity_classifier,
            optimizer,
            species_labels,
            activity_labels,
            box_images,
            box_backbone_features):
        # Determine the device to use based on the backbone's fc weights
        device = self._backbone.device

        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)
        box_backbone_features = box_backbone_features.to(device)

        # Extract side network box features
        box_side_features = self._backbone.compute_side_features(box_images)

        # Concatenate backbone and side features
        box_features = torch.cat(
            (box_backbone_features, box_side_features),
            dim=1
        )

        # Compute logits by passing the features through the appropriate
        # classifiers
        species_preds = species_classifier(box_features)
        activity_preds = activity_classifier(box_features)

        species_loss = torch.nn.functional.cross_entropy(
            species_preds,
            species_labels,
            label_smoothing=self._label_smoothing
        )
        activity_loss = torch.nn.functional.cross_entropy(
            activity_preds,
            activity_labels,
            label_smoothing=self._label_smoothing
        )

        loss = species_loss + activity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
        
        return loss.detach().cpu().item(),\
            n_species_correct,\
            n_activity_correct

    def _train_epoch(
            self,
            data_loader,
            species_classifier,
            activity_classifier,
            optimizer):
        # Set everything to train mode
        self._backbone.train()
        species_classifier.train()
        activity_classifier.train()
        
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_examples = 0
        n_species_correct = 0
        n_activity_correct = 0

        for species_labels, activity_labels, box_images, box_backbone_features\
                in data_loader:
            batch_loss, batch_n_species_correct, batch_n_activity_correct =\
                self._train_batch(
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    species_labels,
                    activity_labels,
                    box_images,
                    box_backbone_features
                )

            sum_loss += batch_loss
            n_iterations += 1
            n_examples += box_images.shape[0]
            n_species_correct += batch_n_species_correct
            n_activity_correct += batch_n_activity_correct

        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, mean_accuracy

    def _val_batch(
            self,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images,
            box_backbone_features):
        device = self._backbone.device

        # Move to device
        species_labels = species_labels.to(device)
        activity_labels = activity_labels.to(device)
        box_images = box_images.to(device)
        box_backbone_features = box_backbone_features.to(device)

        # Extract side network box features
        box_side_features = self._backbone.compute_side_features(box_images)

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
            data_loader,
            species_classifier,
            activity_classifier):
        with torch.no_grad():
            self._backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = 0
            n_species_correct = 0
            n_activity_correct = 0

            for species_labels, activity_labels, box_images,\
                    box_backbone_features in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        species_classifier,
                        activity_classifier,
                        species_labels,
                        activity_labels,
                        box_images,
                        box_backbone_features
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
            species_classifier,
            activity_classifier,
            root_log_dir):
        # TODO new feedback setup will require a separate dataloader since
        # feedback data has several multiple instance issues and thus requires
        # image-level labels and special loss functions.

        # TODO Class balancing

        # Load training and validation backbone features from feature
        # files
        train_box_features, train_species_labels, train_activity_labels =\
            torch.load(
                self._train_feature_file,
                map_location=self._device
            )
        val_box_features, val_species_labels, val_activity_labels =\
            torch.load(
                self._val_feature_file,
                map_location=self._device
            )

        # TODO Precompute feedback backbone features

        # Construct training and validation
        # FeatureConcatFlattenedBoxImageDataset objects from known features

        known_train_dataset = FeatureConcatFlattenedBoxImageDataset(
            FlattenedBoxImageDataset(self._train_dataset),
            train_box_features
        )
        known_val_dataset = FeatureConcatFlattenedBoxImageDataset(
            FlattenedBoxImageDataset(self._val_known_dataset),
            val_box_features
        )

        # TODO Construct custom feedback dataset; separate class from known data
        # datasets because of the multiple instance problems. Weighted sampling
        # will come from explicitly drawing batches from each loader (known
        # and feedback) separately, and then balancing them mid-iteration

        # For now, we just have a single loader for training and a single loader
        # for validation (until feedback is implemented)
        train_dataset = known_train_dataset
        train_loader = DataLoader(
            train_dataset,
            batch_size=self._retraining_batch_size,
            shuffle=True,
            num_workers=2
        )
        val_dataset = known_val_dataset
        val_loader = DataLoader(
            val_dataset,
            batch_size=self._retraining_batch_size,
            shuffle=False,
            num_workers=2
        )

        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(self._backbone.retrainable_parameters())\
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

        if root_log_dir is not None and self._allow_write:
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
                train_loader, 
                species_classifier,
                activity_classifier,
                optimizer
            )

            if root_log_dir is not None and self._allow_write:
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
                    val_loader,
                    species_classifier,
                    activity_classifier
                )

                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict =\
                        deepcopy(self._backbone.state_dict())
                    best_accuracy_species_classifier_state_dict =\
                        deepcopy(species_classifier.state_dict())
                    best_accuracy_activity_classifier_state_dict =\
                        deepcopy(activity_classifier.state_dict())
                else:
                    epochs_since_improvement += 1

                if root_log_dir is not None and self._allow_write:
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
        self._backbone.load_state_dict(best_accuracy_backbone_state_dict)
        species_classifier.load_state_dict(
            best_accuracy_species_classifier_state_dict
        )
        activity_classifier.load_state_dict(
            best_accuracy_activity_classifier_state_dict
        )

    def prepare_for_retraining(
            self,
            classifier):
        # Reset only the side network's weights
        self._backbone.reset()

        # Update classifier's bottleneck dim to account for side network's
        # features before resetting
        classifier.reset(bottleneck_dim=512)


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
        label_mapping,
        box_transform,
        post_cache_train_transform,
        post_cache_val_transform,
        root_cache_dir=None,
        allow_write=False,
        n_known_val=4068):
    static_label_mapper =\
        LabelMapper(label_mapping=deepcopy(label_mapping), update=False)
    dynamic_label_mapper =\
        LabelMapper(label_mapping, update=True)

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
        write_cache=allow_write
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
        write_cache=allow_write
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
        val_dataset,\
        dynamic_label_mapper,\
        static_label_mapper


def compute_features(
        backbone,
        root_save_dir,
        box_transform,
        post_cache_train_transform,
        train_dataset,
        val_known_dataset,
        retraining_batch_size):
    backbone.eval()
    flattened_train_dataset = FlattenedBoxImageDataset(train_dataset)
    train_loader = DataLoader(
        flattened_train_dataset,
        batch_size=retraining_batch_size,
        shuffle=False,
        num_workers=2
    )

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
        num_workers=2
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

    # Determine the device to use based on the backbone's fc weights
    device = backbone.device

    train_box_features = []
    train_species_labels = []
    train_activity_labels = []

    with torch.no_grad():
        for species_labels, activity_labels, box_images in train_loader:
            # Move to device
            species_labels = species_labels.to(device)
            activity_labels = activity_labels.to(device)
            box_images = box_images.to(device)

            # Extract box features
            box_features = backbone(box_images)

            # Store
            train_box_features.append(box_features)
            train_species_labels.append(species_labels)
            train_activity_labels.append(activity_labels)

        train_box_features = torch.cat(train_box_features, dim=0)
        train_species_labels = torch.cat(train_species_labels, dim=0)
        train_activity_labels = torch.cat(train_activity_labels, dim=0)

    val_box_features = []
    val_species_labels = []
    val_activity_labels = []

    with torch.no_grad():
        for species_labels, activity_labels, box_images in val_loader:
            # Move to device
            species_labels = species_labels.to(device)
            activity_labels = activity_labels.to(device)
            box_images = box_images.to(device)

            # Extract box features
            box_features = backbone(box_images)

            # Store
            val_box_features.append(box_features)
            val_species_labels.append(species_labels)
            val_activity_labels.append(activity_labels)

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

        # TODO class balancing? In the SVO system, we balanced 50/50 known
        # and novel examples to avoid biasing P(N_i) toward 1. But maybe it
        # doesn't matter here since we aren't using P(N_i) for merging
        # SCG / non-SCG predictions. We also previously would sample a batch
        # from each of 6 data loaders, which naturally balanced them, when
        # training the classifier: S/V/O x known/novel
        self._feedback_data = None

    def add_feedback_data(self, data_root, csv_path):
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
        new_novel_dataset = TransformingBoxImageDataset(
            new_novel_dataset,
            self._post_cache_train_transform
        )

        # Put new feedback data in list
        if self._feedback_data is None:
            self._feedback_data = new_novel_dataset
        else:
            self._feedback_data = ConcatDataset(
                [self._feedback_data, new_novel_dataset]
            )
    
    # Should be called before train_novelty_detection_module(), except when
    # training for the very first time manually. This prepares the
    # backbone, classifier, and novelty type logistic regressions for
    # retraining. Most likely this is done by fully randomizing them, but in
    # the future we might change the process to be e.g. a warm-start,
    # shrink-and-perturb, or crashing a single layer.
    def prepare_for_retraining(
            self,
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model):
        # Reset the classifiers (and possibly certain backbone components,
        # depending on the classifier retraining method) if appropriate
        self._classifier_trainer.prepare_for_retraining(classifier)
        
        # Reset the confidence calibrator
        confidence_calibrator.reset()

        # Reset logistic regressions and statistical model
        novelty_type_classifier.reset()
        activation_statistical_model.reset()

    def fit_activation_statistics(
            self,
            backbone,
            activation_statistical_model):
        activation_stats_training_loader = DataLoader(
            self._val_known_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=custom_collate,
            num_workers=2
        )
        backbone.eval()

        device = backbone.device

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

    def calibrate_temperature_scalers(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            species_calibrator,
            activity_calibrator):
        cal_loader = DataLoader(
            self._val_known_dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=custom_collate,
            num_workers=2
        )

        # Set everything to eval mode for calibration, except the calibrators
        backbone.eval()
        species_classifier.eval()
        activity_classifier.eval()
        species_calibrator.train()
        activity_calibrator.train()

        device = backbone.device

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
                one_hot_activity_labels = torch.argmax(batch_activity_labels, dim=1)
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

    def train_novelty_type_logistic_regressions(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            novelty_type_classifier,
            activation_statistical_model,
            scorer):
        # Set the backbone and classifiers to eval(), but set the logistic
        # regressions to train()
        backbone.eval()
        species_classifier.eval()
        activity_classifier.eval()
        novelty_type_classifier.train()

        device = backbone.device

        val_loader = DataLoader(
            self._val_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=custom_collate,
            num_workers=2
        )
        
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
        fit_logistic_regression(
            novelty_type_classifier,
            scores,
            labels,
            epochs=3000
        )

    def train_novelty_detection_module(
            self,
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model,
            scorer,
            lr=0.0005,
            root_checkpoint_dir=None,
            root_log_dir=None,
            train_sampler_fn=None,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            scheduler_type=SchedulerType.none):
        species_classifier = classifier.species_classifier
        activity_classifier = classifier.activity_classifier
        species_calibrator = confidence_calibrator.species_calibrator
        activity_calibrator = confidence_calibrator.activity_calibrator
        
        # Retrain the backbone and classifiers
        self._classifier_trainer.train(
            species_classifier,
            activity_classifier,
            root_log_dir
        )

        self.fit_activation_statistics(
            backbone,
            activation_statistical_model
        )

        # Retrain the classifier's temperature scaling calibrators
        self.calibrate_temperature_scalers(
            backbone,
            species_classifier,
            activity_classifier,
            species_calibrator,
            activity_calibrator
        )

        # Retrain the logistic regressions
        self.train_novelty_type_logistic_regressions(
            backbone,
            species_classifier,
            activity_classifier,
            novelty_type_classifier,
            activation_statistical_model,
            scorer
        )
