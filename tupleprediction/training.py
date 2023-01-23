from copy import deepcopy
import numpy as np
import os
from enum import Enum
import pickle as pkl

from tqdm import tqdm
from torch.utils.data import\
    Dataset,\
    DataLoader,\
    Subset as TorchSubset,\
    ConcatDataset as TorchConcatDataset
import torch
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


class Augmentation(Enum):
    rand_augment = 'rand-augment'
    horizontal_flip = 'horizontal-flip'
    none = 'none'

    def __str__(self):
        return self.value

class SchedulerType(Enum):
    cosine = 'cosine'
    none = 'none'

    def __str__(self):
        return self.value

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


def separate_known_images(dataset, n_known_species_cls, n_known_activity_cls):
    known_indices = []
    for idx, (_, _, novelty_type_label, _, _) in enumerate(dataset):
        if novelty_type_label == 0:
            known_indices.append(idx)

    return Subset(dataset, known_indices)


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


class TuplePredictorTrainer:
    augmentation_dict = {
        Augmentation.rand_augment: RandAugment,
        Augmentation.horizontal_flip: RandomHorizontalFlip,
        Augmentation.none: NoOpTransform
    }

    scheduler_dict = {
        SchedulerType.cosine: CosineAnnealingLR,
        SchedulerType.none: None
    }

    def __init__(
            self,
            data_root,
            train_csv_path,
            val_csv_path,
            retraining_batch_size,
            n_species_cls,
            n_activity_cls,
            n_known_species_cls,
            n_known_activity_cls,
            label_mapping,
            augmentation=Augmentation.rand_augment,
            allow_write=False,
            n_known_val=1000,
            root_cache_dir=None):
        self._n_species_cls = n_species_cls
        self._n_activity_cls = n_activity_cls
        self._static_label_mapper =\
            LabelMapper(label_mapping=deepcopy(label_mapping), update=False)
        self._dynamic_label_mapper =\
            LabelMapper(label_mapping, update=True)
        self._box_transform = ResizePad(224)
        normalize = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        augmentation_ctor = self.augmentation_dict[augmentation]
        self._post_cache_train_transform =\
            Compose((augmentation_ctor(), normalize))
        self._post_cache_val_transform = normalize

        if root_cache_dir is not None:
            train_cache_dir = os.path.join(root_cache_dir, 'train')
            val_cache_dir = os.path.join(root_cache_dir, 'val')
        else:
            train_cache_dir = None
            val_cache_dir = None

        train_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = train_csv_path,
            training = True,
            n_species_cls=n_species_cls,
            n_activity_cls=n_activity_cls,
            label_mapper=self._dynamic_label_mapper,
            box_transform=self._box_transform,
            cache_dir=train_cache_dir,
            write_cache=allow_write
        )
        train_dataset.commit_cache()

        val_known_indices_gen = np.random.Generator(np.random.PCG64(0))
        val_known_indices = val_known_indices_gen.choice(
            list(range(len(train_dataset))),
            size=n_known_val,
            replace=False
        ).tolist()
        val_known_indices_set = set(val_known_indices)
        training_indices = [x for x in range(len(train_dataset)) if\
            not x in val_known_indices_set]
        
        self._val_known_dataset = TransformingBoxImageDataset(
            Subset(train_dataset, val_known_indices),
            self._post_cache_val_transform
        )
        self._train_dataset = TransformingBoxImageDataset(
            Subset(train_dataset, training_indices),
            self._post_cache_train_transform
        )

        val_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = val_csv_path,
            training = False,
            n_species_cls=n_species_cls,
            n_activity_cls=n_activity_cls,
            label_mapper=self._static_label_mapper,
            box_transform=self._box_transform,
            cache_dir=val_cache_dir,
            write_cache=allow_write
        )
        val_dataset.commit_cache()

        self._val_dataset = ConcatDataset((
            self._val_known_dataset,
            TransformingBoxImageDataset(
                val_dataset,
                self._post_cache_val_transform
            )
        ))

        # TODO class balancing? In the SVO system, we balanced 50/50 known
        # and novel examples to avoid biasing P(N_i) toward 1. But maybe it
        # doesn't matter here since we aren't using P(N_i) for merging
        # SCG / non-SCG predictions. We also previously would sample a batch
        # from each of 6 data loaders, which naturally balanced them, when
        # training the classifier: S/V/O x known/novel
        self._feedback_data = None

        self._retraining_batch_size = retraining_batch_size 
        self._allow_write = allow_write

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
    def prepare_for_retraining(self,
            backbone,
            classifier,
            confidence_calibrator,
            novelty_type_classifier,
            activation_statistical_model):
        # Reset the backbone
        backbone.reset()
        
        # Reset the classifier and confidence calibrator
        classifier.reset()
        confidence_calibrator.reset()

        # Reset logistic regressions and statistical model
        novelty_type_classifier.reset()
        activation_statistical_model.reset()

    def _train_batch(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer,
            species_labels,
            activity_labels,
            box_images,
            label_smoothing):
        # Determine the device to use based on the backbone's fc weights
        device = backbone.device

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

        species_loss = torch.nn.functional.cross_entropy(
            species_preds,
            species_labels,
            label_smoothing=label_smoothing
        )
        activity_loss = torch.nn.functional.cross_entropy(
            activity_preds,
            activity_labels,
            label_smoothing=label_smoothing
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

    def _train_epoch(self,
            data_loader,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer,
            label_smoothing):
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

        for species_labels, activity_labels, box_images in data_loader:
            batch_loss, batch_n_species_correct, batch_n_activity_correct =\
                self._train_batch(
                    backbone,
                    species_classifier,
                    activity_classifier,
                    optimizer,
                    species_labels,
                    activity_labels,
                    box_images,
                    label_smoothing
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
            backbone,
            species_classifier,
            activity_classifier,
            species_labels,
            activity_labels,
            box_images):
        device = backbone.device

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
            backbone,
            species_classifier,
            activity_classifier):
        with torch.no_grad():
            backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            n_examples = 0
            n_species_correct = 0
            n_activity_correct = 0

            for species_labels, activity_labels, box_images in data_loader:
                batch_n_species_correct, batch_n_activity_correct =\
                    self._val_batch(
                        backbone,
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

    def train_backbone_and_classifiers(
            self,
            backbone,
            species_classifier,
            activity_classifier,
            lr,
            train_sampler_fn=None,
            root_checkpoint_dir=None,
            root_log_dir=None,
            patience=3,
            min_epochs=3,
            max_epochs=30,
            label_smoothing=0.0,
            scheduler_type=SchedulerType.none):
        if self._feedback_data is not None:
            train_dataset = ConcatDataset((
                self._train_dataset, self._feedback_data
            ))
        else:
            train_dataset = self._train_dataset

        train_dataset = FlattenedBoxImageDataset(train_dataset)
        if train_sampler_fn is not None:
            train_sampler = train_sampler_fn(train_dataset)
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
            list(backbone.parameters())\
                + list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            lr,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Init scheduler to None. It will be constructed after loading
        # the optimizer state dict, or after failing to do so
        scheduler = None
        scheduler_ctor = self.scheduler_dict[scheduler_type]

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
                root_checkpoint_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                f'lr={lr}',
                f'label_smoothing={label_smoothing:.2f}'
            )

        if root_checkpoint_dir is not None:
            checkpoint_dir = get_checkpoint_dir()
            training_checkpoint = os.path.join(checkpoint_dir, 'training.pth')
            validation_checkpoint =\
                os.path.join(checkpoint_dir, 'validation.pth')
            
            if os.path.exists(training_checkpoint):
                sd = torch.load(
                    training_checkpoint,
                    map_location=backbone.device
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
                        max_epochs
                    )
                    scheduler.load_state_dict(sd['scheduler'])
            if os.path.exists(validation_checkpoint):
                sd = torch.load(
                    validation_checkpoint,
                    map_location=backbone.device
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
                max_epochs
            )
        training_loss_curve = {}
        training_accuracy_curve = {}
        validation_accuracy_curve = {}

        def get_log_dir():
            return os.path.join(
                root_log_dir,
                self._box_transform.path(),
                self._post_cache_train_transform.path(),
                f'lr={lr}',
                f'label_smoothing={label_smoothing:.2f}'
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
            range(start_epoch, max_epochs),
            desc=gen_tqdm_description(
                'Training backbone and classifiers...',
                train_loss=mean_train_loss,
                train_accuracy=mean_train_accuracy,
                val_accuracy=mean_val_accuracy
            ),
            total=max_epochs,
            initial=start_epoch
        )
        for epoch in progress:
            if patience is not None and epochs_since_improvement >= patience:
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
                backbone,
                species_classifier,
                activity_classifier,
                optimizer,
                label_smoothing
            )

            if root_checkpoint_dir is not None and self._allow_write:
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
            if epoch >= min_epochs - 1:
                mean_val_accuracy = self._val_epoch(
                    val_loader,
                    backbone,
                    species_classifier,
                    activity_classifier
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

                if root_checkpoint_dir is not None and self._allow_write:
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
        backbone.load_state_dict(best_accuracy_backbone_state_dict)
        species_classifier.load_state_dict(
            best_accuracy_species_classifier_state_dict
        )
        activity_classifier.load_state_dict(
            best_accuracy_activity_classifier_state_dict
        )

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
            batch_size=256,
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
        self.train_backbone_and_classifiers(
            backbone,
            species_classifier,
            activity_classifier,
            lr,
            train_sampler_fn=train_sampler_fn,
            root_checkpoint_dir=root_checkpoint_dir,
            root_log_dir=root_log_dir,
            patience=patience,
            min_epochs=min_epochs,
            max_epochs=max_epochs,
            label_smoothing=label_smoothing,
            scheduler_type=scheduler_type
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
