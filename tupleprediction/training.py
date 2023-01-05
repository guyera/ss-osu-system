from copy import deepcopy

from tqdm import tqdm
from torch.utils.data import Dataset
import torch

from boximagedataset import BoxImageDataset
from utils import custom_collate
from labelmapping import LabelMapper

def separate_known_images(dataset, n_known_species_cls, n_known_activity_cls):
    known_indices = []
    for idx, (_, _, novelty_type_label, _, _) in enumerate(dataset):
        if novelty_type_label == 0:
            known_indices.append(idx)

    return torch.utils.data.Subset(dataset, known_indices)


def fit_logistic_regression(logistic_regression, scores, labels, epochs = 3000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        logistic_regression.parameters(),
        lr = 0.01,
        momentum = 0.9
    )
    logistic_regression.fit_standardization_statistics(scores)
    
    progress = tqdm(range(epochs), desc = 'Fitting logistic regression...')
    for epoch in progress:
        optimizer.zero_grad()
        logits = logistic_regression(scores)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        progress.set_description((
            f'Fitting logistic regression... | Loss: '
            f'{loss.detach().cpu().item()}'
        ))
    progress.close()


class TuplePredictorTrainer:
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
            label_mapping):
        self._n_species_cls = n_species_cls
        self._n_activity_cls = n_activity_cls
        self._static_label_mapper =\
            LabelMapper(label_mapping=deepcopy(label_mapping), update=False)
        self._dynamic_label_mapper =\
            LabelMapper(label_mapping, update=True)
        self._train_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = train_csv_path,
            training = True,
            n_species_cls=n_species_cls,
            n_activity_cls=n_activity_cls,
            label_mapper=self._dynamic_label_mapper
        )

        self._val_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = val_csv_path,
            training = False,
            n_species_cls=n_species_cls,
            n_activity_cls=n_activity_cls,
            label_mapper=self._static_label_mapper
        )

        # TODO class balancing? In the SVO system, we balanced 50/50 known
        # and novel examples to avoid biasing P(N_i) toward 1. But maybe it
        # doesn't matter here since we aren't using P(N_i) for merging
        # SCG / non-SCG predictions. We also previously would sample a batch
        # from each of 6 data loaders, which naturally balanced them, when
        # training the classifier: S/V/O x known/novel

        # Extract whole images
        self._activation_stats_training_dataset = separate_known_images(
            self._val_dataset,
            n_known_species_cls,
            n_known_activity_cls
        )

        self._feedback_data = None

        self._retraining_batch_size = retraining_batch_size 

    def add_feedback_data(self, data_root, csv_path):
        new_novel_dataset = BoxImageDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = csv_path,
            training = True,
            n_species_cls=self._n_species_cls,
            n_activity_cls=self._n_activity_cls,
            label_mapper=self._dynamic_label_mapper
        )

        # Put new feedback data in list
        if self._feedback_data is None:
            self._feedback_data = new_novel_dataset
        else:
            self._feedback_data = torch.utils.data.ConcatDataset(
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

    def train_epoch(self,
            data_loader,
            backbone,
            species_classifier,
            activity_classifier,
            optimizer):
        # Determine the device to use based on the backbone's fc weights
        device = backbone.device
        
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

        for species_labels, activity_labels, _, box_images, whole_images in\
                data_loader:
            # Flatten the boxes across images and extract per-image box
            # counts.
            box_counts = [x.shape[0] for x in box_images]
            flattened_box_images = torch.cat(box_images, dim=0)

            # Move to device
            species_labels = species_labels.to(device)
            activity_labels = activity_labels.to(device)
            flattened_box_images = flattened_box_images.to(device)
            whole_images = whole_images.to(device)

            # Construct per-box labels.
            one_hot_species_labels = torch.argmax(species_labels, dim=1)
            flattened_species_labels = torch.cat([
                torch.full((box_count,), species_label, device=device)\
                    for species_label, box_count in\
                        zip(one_hot_species_labels, box_counts)
            ])
            one_hot_activity_labels = torch.argmax(activity_labels, dim=1)
            flattened_activity_labels = torch.cat([
                torch.full((box_count,), activity_label, device=device)\
                    for activity_label, box_count in\
                        zip(one_hot_activity_labels, box_counts)
            ])

            # Extract box features
            box_features = backbone(flattened_box_images)

            # Compute logits by passing the features through the appropriate
            # classifiers
            species_preds = species_classifier(box_features)
            activity_preds = activity_classifier(box_features)

            species_loss = torch.nn.functional.cross_entropy(
                species_preds,
                flattened_species_labels
            )
            activity_loss = torch.nn.functional.cross_entropy(
                activity_preds,
                flattened_activity_labels
            )

            batch_loss = species_loss + activity_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            sum_loss += batch_loss.detach().cpu().item()

            n_iterations += 1
            n_examples += flattened_box_images.shape[0]

            species_correct = torch.argmax(species_preds, dim=1) == \
                flattened_species_labels
            n_species_correct += int(
                species_correct.to(torch.int).sum().detach().cpu().item()
            )

            activity_correct = torch.argmax(activity_preds, dim=1) == \
                flattened_activity_labels
            n_activity_correct += int(
                activity_correct.to(torch.int).sum().detach().cpu().item()
            )

        mean_loss = sum_loss / n_iterations

        mean_species_accuracy = float(n_species_correct) / n_examples
        mean_activity_accuracy = float(n_activity_correct) / n_examples

        mean_accuracy = (mean_species_accuracy + mean_activity_accuracy) / 2.0

        return mean_loss, mean_accuracy

    def val_epoch(
            self,
            data_loader,
            backbone,
            species_classifier,
            activity_classifier):
        with torch.no_grad():
            backbone.eval()
            species_classifier.eval()
            activity_classifier.eval()

            device = backbone.device

            n_examples = 0
            n_species_correct = 0
            n_activity_correct = 0

            for species_labels, activity_labels, _, box_images, whole_images in\
                    data_loader:
                # Flatten the boxes across images and extract per-image box
                # counts.
                box_counts = [x.shape[0] for x in box_images]
                flattened_box_images = torch.cat(box_images, dim=0)

                # Move to device
                species_labels = species_labels.to(device)
                activity_labels = activity_labels.to(device)
                flattened_box_images = flattened_box_images.to(device)
                whole_images = whole_images.to(device)

                # Construct per-box labels.
                one_hot_species_labels = torch.argmax(species_labels, dim=1)
                flattened_species_labels = torch.cat([
                    torch.full((box_count,), species_label, device=device)\
                        for species_label, box_count in\
                            zip(one_hot_species_labels, box_counts)
                ])
                one_hot_activity_labels = torch.argmax(activity_labels, dim=1)
                flattened_activity_labels = torch.cat([
                    torch.full((box_count,), activity_label, device=device)\
                        for activity_label, box_count in\
                            zip(one_hot_activity_labels, box_counts)
                ])

                # Extract box features
                box_features = backbone(flattened_box_images)

                # Compute logits by passing the features through the appropriate
                # classifiers
                species_preds = species_classifier(box_features)
                activity_preds = activity_classifier(box_features)

                n_examples += flattened_box_images.shape[0]

                species_correct = torch.argmax(species_preds, dim=1) == \
                    flattened_species_labels
                n_species_correct += int(
                    species_correct.to(torch.int).sum().detach().cpu().item()
                )

                activity_correct = torch.argmax(activity_preds, dim=1) == \
                    flattened_activity_labels
                n_activity_correct += int(
                    activity_correct.to(torch.int).sum().detach().cpu().item()
                )

            mean_species_accuracy = float(n_species_correct) / n_examples
            mean_activity_accuracy = float(n_activity_correct) / n_examples

            mean_accuracy = \
                (mean_species_accuracy + mean_activity_accuracy) / 2.0

            return mean_accuracy

    def train_backbone_and_classifiers(
            self,
            backbone,
            species_classifier,
            activity_classifier):
        if self._feedback_data is not None:
            train_dataset = torch.utils.data.ConcatDataset((
                self._train_dataset, self._feedback_data
            ))
        else:
            train_dataset = self._train_dataset

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._retraining_batch_size,
            shuffle=True,
            collate_fn=custom_collate
        )

        # Construct validation loaders for early stopping / model selection.
        # I'm assuming our model selection strategy will be based solely on the
        # validation classification accuracy and not based on novelty detection
        # capabilities in any way. Otherwise, we can use the novel validation
        # data to measure novelty detection performance. These currently aren't
        # being stored (except in a special form for the logistic regressions),
        # so we'd have to modify __init__().
        val_loader = torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=custom_collate
        )

        # Retrain the backbone and classifiers
        # Construct the optimizer
        optimizer = torch.optim.SGD(
            list(backbone.parameters())\
                + list(species_classifier.parameters())\
                + list(activity_classifier.parameters()),
            0.0005,
            momentum=0.9,
            weight_decay=1e-3
        )

        # Define convergence parameters (early stopping + model selection)
        patience = 3
        epochs_since_improvement = 0
        max_epochs = 30
        min_epochs = 4
        best_accuracy = None
        best_accuracy_backbone_state_dict = None
        best_accuracy_species_classifier_state_dict = None
        best_accuracy_activity_classifier_state_dict = None

        # Train
        progress = tqdm(
            range(max_epochs),
            desc='Training backbone and classifiers...'
        )
        for epoch in progress:
            # Train for one full epoch
            mean_train_loss, mean_train_accuracy = self.train_epoch(
                train_loader, 
                backbone,
                species_classifier,
                activity_classifier,
                optimizer
            )
                
            # Measure validation accuracy for early stopping / model selection.
            if epoch >= min_epochs - 1:
                mean_val_accuracy = self.val_epoch(
                    val_loader,
                    backbone,
                    species_classifier,
                    activity_classifier
                )
                
                progress.set_description((
                    f'Training backbone and classifiers... | Train Loss: '
                    f'{mean_train_loss} | Train Acc: '
                    f'{mean_known_train_accuracy} | Val Acc: '
                    f'{mean_val_accuracy}'
                ))
                
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
                    if epochs_since_improvement >= patience:
                        # We haven't improved in several epochs. Time to stop
                        # training.
                        break
        progress.close()

        # Load the best accuracy state dicts
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
        activation_stats_training_loader = torch.utils.data.DataLoader(
            self._activation_stats_training_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=custom_collate
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
        cal_loader = torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size=256,
            shuffle=False,
            collate_fn=custom_collate
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

        val_loader = torch.utils.data.DataLoader(
            self._val_dataset,
            batch_size = 32,
            shuffle = False,
            collate_fn=custom_collate
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
                batch_scores = scorer(
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
        print('Fitting novelty type logistic regression...')
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
            scorer):
        species_classifier = classifier.species_classifier
        activity_classifier = classifier.activity_classifier
        species_calibrator = confidence_calibrator.species_calibrator
        activity_calibrator = confidence_calibrator.activity_calibrator
        
        # Retrain the backbone and classifiers
        self.train_backbone_and_classifiers(
            backbone,
            species_classifier,
            activity_classifier
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
