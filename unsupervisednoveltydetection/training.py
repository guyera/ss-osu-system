import copy

from tqdm import tqdm
import torch
from torchvision.models import resnet50, swin_t, swin_b

import noveltydetectionfeatures
import noveltydetection

class SubjectBoxImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        self.novelty_feature_dataset = novelty_feature_dataset

    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        _, _, _, _, subject_label, _, _, subject_box_image, _, _, _ = self.novelty_feature_dataset[idx]
        return subject_box_image, subject_label

class ObjectBoxImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        self.novelty_feature_dataset = novelty_feature_dataset

    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        _, _, _, _, _, object_label, _, _, object_box_image, _, _ = self.novelty_feature_dataset[idx]
        return object_box_image, object_label

class VerbBoxImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        self.novelty_feature_dataset = novelty_feature_dataset

    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        spatial_encodings, _, _, _, _, _, verb_label, _, _, verb_box_image, _ = self.novelty_feature_dataset[idx]
        return verb_box_image, spatial_encodings, verb_label

class WholeImageDataset(torch.utils.data.Dataset):
    def __init__(self, novelty_feature_dataset):
        self.novelty_feature_dataset = novelty_feature_dataset

    def __len__(self):
        return len(self.novelty_feature_dataset)

    def __getitem__(self, idx):
        _, _, _, _, _, _, _, _, _, _, whole_image = self.novelty_feature_dataset[idx]
        return whole_image

def custom_cross_entropy(predictions, targets):
    log_softmaxes = predictions - torch.logsumexp(predictions, dim = 1, keepdim = True)
    products = targets * log_softmaxes
    negations = -products
    sums = negations.sum(dim = 1)
    reduction = sums.mean()
    return reduction

def separate_data(dataset):
    known_subject_indices = []
    novel_subject_indices = []
    known_object_indices = []
    novel_object_indices = []
    known_verb_indices = []
    novel_verb_indices = []
    for idx, (_, _, _, _, subject_label, object_label, verb_label, _, _, _, _) in enumerate(dataset):
        if subject_label is not None:
            if subject_label == 0:
                # Novel subject
                novel_subject_indices.append(idx)
            else:
                # Known subject
                known_subject_indices.append(idx)

        if object_label is not None:
            if object_label == 0:
                # Novel object
                novel_object_indices.append(idx)
            else:
                # Known object
                known_object_indices.append(idx)

        if verb_label is not None:
            if verb_label == 0:
                # Novel verb
                novel_verb_indices.append(idx)
            else:
                # Known verb
                known_verb_indices.append(idx)

    known_subject_dataset = SubjectBoxImageDataset(torch.utils.data.Subset(dataset, known_subject_indices))
    novel_subject_dataset = SubjectBoxImageDataset(torch.utils.data.Subset(dataset, novel_subject_indices))
    known_object_dataset = ObjectBoxImageDataset(torch.utils.data.Subset(dataset, known_object_indices))
    novel_object_dataset = ObjectBoxImageDataset(torch.utils.data.Subset(dataset, novel_object_indices))
    known_verb_dataset = VerbBoxImageDataset(torch.utils.data.Subset(dataset, known_verb_indices))
    novel_verb_dataset = VerbBoxImageDataset(torch.utils.data.Subset(dataset, novel_verb_indices))

    return known_subject_dataset,\
        novel_subject_dataset,\
        known_object_dataset,\
        novel_object_dataset,\
        known_verb_dataset,\
        novel_verb_dataset

def separate_known_whole_images(dataset):
    known_indices = []
    for idx, (_, _, _, _, subject_label, object_label, verb_label, _, _, _, _) in enumerate(dataset):
        if subject_label is not None:
            if subject_label == 0:
                # Novel subject
                continue

        if object_label is not None:
            if object_label == 0:
                # Novel object
                continue

        if verb_label is not None:
            if verb_label == 0:
                # Novel verb
                continue
    
        known_indices.append(idx)
    
    return WholeImageDataset(torch.utils.data.Subset(dataset, known_indices))


def class_balance(dataset):
    # Determine the class with the fewest instances
    class_counts = {}
    for item_tuple in dataset:
        label = item_tuple[-1]
        if torch.is_tensor(label):
            numeric_label = int(label.item())
        else:
            numeric_label = label
        
        if numeric_label not in class_counts:
            class_counts[numeric_label] = 0

        class_counts[numeric_label] += 1

    min_class_count = min([v for k, v in class_counts.items()])
    
    # Iterate over dataset in random order, accumulating data for each class
    # until the min_class_count is reached.
    class_counts = {}
    selected_indices = []
    generator = torch.Generator()
    generator.manual_seed(0)
    indices = torch.randperm(len(dataset), generator = generator)
    for idx in indices:
        item_tuple = dataset[idx]
        label = item_tuple[-1]

        if torch.is_tensor(label):
            numeric_label = int(label.item())
        else:
            numeric_label = label
        
        if numeric_label not in class_counts:
            class_counts[numeric_label] = 0
        
        # If we haven't yet selected min_class_count data points belonging to
        # this class, then select this one as well
        if class_counts[numeric_label] < min_class_count:
            class_counts[numeric_label] += 1
            selected_indices.append(idx)

    return torch.utils.data.Subset(dataset, selected_indices)

def custom_class_balance(dataset, class_proportions):
    class_counts = {}
    for item_tuple in dataset:
        label = item_tuple[-1]
        if torch.is_tensor(label):
            numeric_label = int(label.item())
        else:
            numeric_label = label
        
        if numeric_label not in class_counts:
            class_counts[numeric_label] = 0

        class_counts[numeric_label] += 1
    
    total_data_points = sum([v for k, v in class_counts.items()])
    target_class_counts = {k: int(total_data_points * class_proportions[k]) for k, v in class_counts.items()}
    
    # The target class counts need to be "refined", since they'll only be
    # possible to achieve if the full dataset balance already exactly matches
    # the class_proportions (otherwise there will be an insufficient amount
    # of data for some classes and extra data for others). To refine them,
    # we simply have to check each target class count as see if it's greater
    # than the corresponding actual class count. If so, we compute a scale 
    # factor by which all target class counts must be multiplied in order to
    # make the target class count for the given class achievable (and preserving
    # the target class count proportions by using the same scale factor across
    # all target class counts). We do this for all such target class counts
    # and find the minimum such scale factor. We then use this scale factor
    # to do the actual scaling. After doing this, the limiting class's target
    # class count will be exactly equal to its actual class count, which uses
    # the maximal amount of data possible.
    
    # Discover the minimum scale factor
    min_scale_factor = None
    for key in target_class_counts:
        if target_class_counts[key] > class_counts[key]:
            scale_factor = float(class_counts[key]) / target_class_counts[key]
            if min_scale_factor is None or scale_factor < min_scale_factor:
                min_scale_factor = scale_factor
    
    # Apply the minimum scale factor, if appropriate
    if min_scale_factor is not None:
        for inner_key in target_class_counts:
            target_class_counts[inner_key] = int(target_class_counts[inner_key] * min_scale_factor)
    
    # Iterate over dataset in random order, accumulating data for each class
    # until the corresponding target class count is reached
    class_counts = {}
    selected_indices = []
    generator = torch.Generator()
    generator.manual_seed(0)
    indices = torch.randperm(len(dataset), generator = generator)
    for idx in indices:
        item_tuple = dataset[idx]
        label = item_tuple[-1]

        if torch.is_tensor(label):
            numeric_label = int(label.item())
        else:
            numeric_label = label
        
        if numeric_label not in class_counts:
            class_counts[numeric_label] = 0
        
        # If we haven't yet selected target number of data points belonging to
        # this class, then select this one as well
        if class_counts[numeric_label] < target_class_counts[numeric_label]:
            class_counts[numeric_label] += 1
            selected_indices.append(idx)

    return torch.utils.data.Subset(dataset, selected_indices)

'''
def separate_by_case(novelty_feature_dataset):
    case_1_images = []
    case_2_images = []
    case_3_images = []
    case_1_spatial_encodings = []
    case_2_spatial_encodings = []
    case_1_labels = []
    case_2_labels = []
    case_3_labels = []

    for spatial_encodings, _, _, _, subject_label, object_label, verb_label, subject_box_image, object_box_image, verb_box_image in novelty_feature_dataset:
        # Determine the novelty type
        if subject_label == 0:
            if object_label == 0 or verb_label == 0:
                # Invalid novel example; multiple novelty types. Filter it out.
                continue
            # Type 1
            type_label = 1
        elif subject_label is not None and verb_label == 0:
            if subject_label == 0 or object_label == 0:
                # Invalid novel example; multiple novelty types. Filter it out.
                continue
            # Type 2
            type_label = 2
        elif object_label == 0:
            if subject_label == 0 or (subject_label is not None and verb_label == 0):
                # Invalid novel example; multiple novelty types. Filter it out.
                continue
            # Type 3
            type_label = 3
        else:
            # Type 0
            type_label = 0

        # Add the images and labels to the case inputs
        if subject_box_image is not None and object_box_image is not None:
            # All boxes present; append images to case 1
            case_1_images.append(torch.stack((subject_box_image, object_box_image, verb_box_image), dim = 0))
            case_1_spatial_encodings.append(spatial_encodings)
            case_1_labels.append(torch.tensor(type_label, dtype = torch.long, device = subject_box_image.device))
        if subject_box_image is not None and (type_label == 0 or type_label == 1 or type_label == 2):
            # At least the subject box is present; append subject and verb images
            # to case 2
            case_2_images.append(torch.stack((subject_box_image, verb_box_image), dim = 0))
            case_2_spatial_encodings.append(spatial_encodings)
            case_2_labels.append(torch.tensor(type_label, dtype = torch.long, device = subject_box_image.device))
        if object_box_image is not None and (type_label == 0 or type_label == 3):
            # At least the object box is present; append object box image to case 3
            case_3_images.append(object_box_image)
            cur_type_label = 0 if type_label == 0 else 1
            case_3_labels.append(torch.tensor(cur_type_label, dtype = torch.long, device = object_box_image.device))

    case_1_images = torch.stack(case_1_images, dim = 0)
    case_1_spatial_encodings = torch.stack(case_1_spatial_encodings, dim = 0)
    case_1_labels = torch.stack(case_1_labels, dim = 0)
    case_2_images = torch.stack(case_2_images, dim = 0)
    case_2_spatial_encodings = torch.stack(case_2_spatial_encodings, dim = 0)
    case_2_labels = torch.stack(case_2_labels, dim = 0)
    case_3_images = torch.stack(case_3_images, dim = 0)
    case_3_labels = torch.stack(case_3_labels, dim = 0)
    
    case_1_dataset = torch.utils.data.TensorDataset(case_1_images, case_1_spatial_encodings, case_1_labels)
    case_2_dataset = torch.utils.data.TensorDataset(case_2_images, case_2_spatial_encodings, case_2_labels)
    case_3_dataset = torch.utils.data.TensorDataset(case_3_images, case_3_labels)

    return case_1_dataset, case_2_dataset, case_3_dataset
'''

def separate_by_case(novelty_feature_dataset):
    case_1_images = []
    case_2_images = []
    case_3_images = []
    case_1_spatial_encodings = []
    case_2_spatial_encodings = []
    case_1_labels = []
    case_2_labels = []
    case_3_labels = []

    # TODO for a lack of case 2 / 3 data, we use case 1 data additionally as
    # case 2 / 3 (just leaving out the additional boxes). Previously, we would
    # determine the novelty type of the case 1 image. If it was a novel object,
    # then we would NOT use it as additional case 2 data. If it was a novel
    # subject or verb, then we would NOT use it as additional case 3 data.
    # The concern was that a novel subject / verb could result in something
    # seeming novel about the object box as well, and that would be fed to
    # the case 3 novelty type logistic regression during calibration. This
    # check was eliminated when updating to phase 3. We could reimplement it,
    # but it's not clear whether it's helpful.
    
    for spatial_encodings, _, _, _, subject_label, object_label, verb_label, subject_box_image, object_box_image, verb_box_image, whole_image in novelty_feature_dataset:
        # Add the images and labels to the case inputs
        
        if subject_box_image is not None and object_box_image is not None:
            # All boxes present; append images to case 1
            case_1_images.append(torch.stack((subject_box_image, object_box_image, verb_box_image, whole_image), dim = 0))
            case_1_spatial_encodings.append(spatial_encodings)
            case_1_labels.append(
                torch.stack(
                    (
                        subject_label,
                        verb_label,
                        object_label
                    ),
                    dim=0
                )
            )
        if subject_box_image is not None:
            # At least the subject box is present; append subject and verb images
            # to case 2
            case_2_images.append(torch.stack((subject_box_image, verb_box_image, whole_image), dim = 0))
            case_2_spatial_encodings.append(spatial_encodings)
            case_2_labels.append(
                torch.stack(
                    (
                        subject_label,
                        verb_label
                    ),
                    dim=0
                )
            )
        if object_box_image is not None:
            # At least the object box is present; append object box image to case 3
            case_3_images.append(torch.stack((object_box_image, whole_image), dim = 0))
            case_3_labels.append(object_label)
       
    
    case_1_images = torch.stack(case_1_images, dim = 0)
    case_1_spatial_encodings = torch.stack(case_1_spatial_encodings, dim = 0)
    case_1_labels = torch.stack(case_1_labels, dim = 0)
    case_2_images = torch.stack(case_2_images, dim = 0)
    case_2_spatial_encodings = torch.stack(case_2_spatial_encodings, dim = 0)
    case_2_labels = torch.stack(case_2_labels, dim = 0)
    case_3_images = torch.stack(case_3_images, dim = 0)
    case_3_labels = torch.stack(case_3_labels, dim = 0)
    
    case_1_dataset = torch.utils.data.TensorDataset(case_1_images, case_1_spatial_encodings, case_1_labels)
    case_2_dataset = torch.utils.data.TensorDataset(case_2_images, case_2_spatial_encodings, case_2_labels)
    case_3_dataset = torch.utils.data.TensorDataset(case_3_images, case_3_labels)

    return case_1_dataset, case_2_dataset, case_3_dataset

def to_case_1_novelty_type_dataset(case_1_dataset):
    # Get all data from the TensorDataset
    images, spatial_encodings, labels = case_1_dataset[:]

    # Determine novel boxes
    s_labels = labels[:, 0]
    v_labels = labels[:, 1]
    o_labels = labels[:, 2]
    novel_s = s_labels == 0
    novel_v = v_labels == 0
    novel_o = o_labels == 0

    # Disqualify images with more than one type of novelty
    total_novel = novel_s.to(torch.int) + novel_v.to(torch.int) + \
        novel_o.to(torch.int)
    valid = total_novel <= 1
    valid_images = images[valid]
    valid_spatial_encodings = spatial_encodings[valid]
    valid_novel_s = novel_s[valid]
    valid_novel_v = novel_v[valid]
    valid_novel_o = novel_o[valid]

    # Determine novelty types
    type_labels = torch.zeros_like(valid_novel_s, dtype=torch.long)
    type_labels[valid_novel_s] = 1
    type_labels[valid_novel_v] = 2
    type_labels[valid_novel_o] = 3

    return torch.utils.data.TensorDataset(
        valid_images,
        valid_spatial_encodings,
        type_labels
    )

def to_case_2_novelty_type_dataset(case_2_dataset):
    # Get all data from the TensorDataset
    images, spatial_encodings, labels = case_2_dataset[:]

    # Determine novel boxes
    s_labels = labels[:, 0]
    v_labels = labels[:, 1]
    novel_s = s_labels == 0
    novel_v = v_labels == 0

    # Disqualify images with more than one type of novelty
    total_novel = novel_s.to(torch.int) + novel_v.to(torch.int)
    valid = total_novel <= 1
    valid_images = images[valid]
    valid_spatial_encodings = spatial_encodings[valid]
    valid_novel_s = novel_s[valid]
    valid_novel_v = novel_v[valid]

    # Determine novelty types
    type_labels = torch.zeros_like(valid_novel_s, dtype=torch.long)
    type_labels[valid_novel_s] = 1
    type_labels[valid_novel_v] = 2

    return torch.utils.data.TensorDataset(
        valid_images,
        valid_spatial_encodings,
        type_labels
    )

def to_case_3_novelty_type_dataset(case_3_dataset):
    # Get all data from the TensorDataset
    images, o_labels = case_3_dataset[:]

    # Determine novel boxes
    novel_o = o_labels == 0

    # Determine novelty types
    type_labels = torch.zeros_like(novel_o, dtype=torch.long)
    type_labels[novel_o] = 1

    return torch.utils.data.TensorDataset(
        images,
        type_labels
    )

def to_case_1_phase_3_novelty_type_dataset(
        case_1_incident_dataset,
        incident_or_environment):
    assert incident_or_environment in ['incident', 'environment']
    if incident_or_environment == 'incident':
        incident = True
    else:
        incident = False

    # Get all data from the TensorDataset
    images, spatial_encodings, labels = case_1_incident_dataset[:]
    
    # Determine novel boxes
    s_labels = labels[:, 0]
    v_labels = labels[:, 1]
    o_labels = labels[:, 2]
    novel_s = s_labels == 0
    novel_v = v_labels == 0
    novel_o = o_labels == 0

    # Disqualify images with any novel boxes
    total_novel = novel_s.to(torch.int) + novel_v.to(torch.int) + \
        novel_o.to(torch.int)
    valid = total_novel == 0
    valid_images = images[valid]
    valid_spatial_encodings = spatial_encodings[valid]

    # Construct incident novelty type labels
    type_labels = torch.full(
        (valid_images.shape[0],),
        4 if incident else 5,
        dtype=torch.long,
        device=valid_images.device
    )

    return torch.utils.data.TensorDataset(
        valid_images,
        valid_spatial_encodings,
        type_labels
    )

def to_case_2_phase_3_novelty_type_dataset(
        case_2_incident_dataset,
        incident_or_environment):
    assert incident_or_environment in ['incident', 'environment']
    if incident_or_environment == 'incident':
        incident = True
    else:
        incident = False
    # Get all data from the TensorDataset
    images, spatial_encodings, labels = case_2_incident_dataset[:]
    
    # Determine novel boxes
    s_labels = labels[:, 0]
    v_labels = labels[:, 1]
    novel_s = s_labels == 0
    novel_v = v_labels == 0

    # Disqualify images with any novel boxes
    total_novel = novel_s.to(torch.int) + novel_v.to(torch.int)
    valid = total_novel == 0
    valid_images = images[valid]
    valid_spatial_encodings = spatial_encodings[valid]

    # Construct incident novelty type labels
    type_labels = torch.full(
        (valid_images.shape[0],),
        3 if incident else 4,
        dtype=torch.long,
        device=valid_images.device
    )

    return torch.utils.data.TensorDataset(
        valid_images,
        valid_spatial_encodings,
        type_labels
    )

def to_case_3_phase_3_novelty_type_dataset(
        case_3_incident_dataset,
        incident_or_environment):
    assert incident_or_environment in ['incident', 'environment']
    if incident_or_environment == 'incident':
        incident = True
    else:
        incident = False
    # Get all data from the TensorDataset
    images, o_labels = case_3_incident_dataset[:]
    
    # Determine novel boxes
    novel_o = o_labels == 0

    # Disqualify images with any novel boxes
    valid = ~novel_o
    valid_images = images[valid]

    # Construct incident novelty type labels
    type_labels = torch.full(
        (valid_images.shape[0],),
        2 if incident else 3,
        dtype=torch.long,
        device=valid_images.device
    )

    return torch.utils.data.TensorDataset(
        valid_images,
        type_labels
    )

def fit_logistic_regression(logistic_regression, scores, labels, epochs = 3000):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(logistic_regression.parameters(), lr = 0.01, momentum = 0.9)
    logistic_regression.fit_standardization_statistics(scores)
    
    progress = tqdm(range(epochs), desc = 'Fitting logistic regression...')
    for epoch in progress:
        optimizer.zero_grad()
        logits = logistic_regression(scores)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        progress.set_description(f'Fitting logistic regression... | Loss: {loss.detach().cpu().item()}')
    progress.close()
    
class NoveltyDetectorTrainer:
    def __init__(self, data_root, train_csv_path, val_csv_path, val_incident_csv_path, val_environment_csv_path, retraining_batch_size, model_ = 'resnet'):
        train_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = train_csv_path,
            training = True,
            image_batch_size = 512,
            backbone = None,
            cache_to_disk = True
        )

        self.model_ = model_

        self.known_train_subject_dataset,\
            self.novel_train_subject_dataset,\
            self.known_train_object_dataset,\
            self.novel_train_object_dataset,\
            self.known_train_verb_dataset,\
            self.novel_train_verb_dataset = separate_data(train_dataset)

        val_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = val_csv_path,
            training = False,
            image_batch_size = 512,
            backbone = None,
            cache_to_disk = True
        )
        
        # Separate the known val data out for temperature scaling calibration
        self.known_val_subject_dataset,\
            _,\
            self.known_val_object_dataset,\
            _,\
            self.known_val_verb_dataset,\
            _ = separate_data(val_dataset)

        # Class-balance them for temperature scaling calibration
        self.calibration_subject_dataset = class_balance(self.known_val_subject_dataset)
        self.calibration_object_dataset = class_balance(self.known_val_object_dataset)
        self.calibration_verb_dataset = class_balance(self.known_val_verb_dataset)

        # Next, also using the val dataset, separate the instances by case
        # and label them by type.

        case_1_val_dataset, case_2_val_dataset, case_3_val_dataset = separate_by_case(val_dataset)
        case_1_val_novelty_type_dataset = to_case_1_novelty_type_dataset(case_1_val_dataset)
        case_2_val_novelty_type_dataset = to_case_2_novelty_type_dataset(case_2_val_dataset)
        case_3_val_novelty_type_dataset = to_case_3_novelty_type_dataset(case_3_val_dataset)
        self.activation_stats_training_dataset = separate_known_whole_images(val_dataset)

        val_incident_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = val_incident_csv_path,
            training = False,
            image_batch_size = 512,
            backbone = None,
            cache_to_disk = True
        )
        
        case_1_val_incident_dataset, case_2_val_incident_dataset, case_3_val_incident_dataset = separate_by_case(val_incident_dataset)
        case_1_val_incident_novelty_type_dataset = to_case_1_phase_3_novelty_type_dataset(case_1_val_incident_dataset, 'incident')
        case_2_val_incident_novelty_type_dataset = to_case_2_phase_3_novelty_type_dataset(case_2_val_incident_dataset, 'incident')
        case_3_val_incident_novelty_type_dataset = to_case_3_phase_3_novelty_type_dataset(case_3_val_incident_dataset, 'incident')

        val_environment_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = val_environment_csv_path,
            training = False,
            image_batch_size = 512,
            backbone = None,
            cache_to_disk = True
        )
        
        case_1_val_environment_dataset, case_2_val_environment_dataset, case_3_val_environment_dataset = separate_by_case(val_environment_dataset)
        case_1_val_environment_novelty_type_dataset = to_case_1_phase_3_novelty_type_dataset(case_1_val_environment_dataset, 'environment')
        case_2_val_environment_novelty_type_dataset = to_case_2_phase_3_novelty_type_dataset(case_2_val_environment_dataset, 'environment')
        case_3_val_environment_novelty_type_dataset = to_case_3_phase_3_novelty_type_dataset(case_3_val_environment_dataset, 'environment')

        case_1_all_novelty_type_dataset = torch.utils.data.ConcatDataset((
            case_1_val_novelty_type_dataset,
            case_1_val_incident_novelty_type_dataset,
            case_1_val_environment_novelty_type_dataset
        ))
        case_2_all_novelty_type_dataset = torch.utils.data.ConcatDataset((
            case_2_val_novelty_type_dataset,
            case_2_val_incident_novelty_type_dataset,
            case_2_val_environment_novelty_type_dataset
        ))
        case_3_all_novelty_type_dataset = torch.utils.data.ConcatDataset((
            case_3_val_novelty_type_dataset,
            case_3_val_incident_novelty_type_dataset,
            case_3_val_environment_novelty_type_dataset
        ))

        # Balance the case-separated data by novelty type
        # TODO switch to a class-weighted sampler in the data loader rather
        # than taking a constant subset of the data points
        case_1_novelty_type_proportions = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.case_1_novelty_type_dataset = custom_class_balance(case_1_all_novelty_type_dataset, case_1_novelty_type_proportions)
        case_2_novelty_type_proportions = [0.5, 0.125, 0.125, 0.125, 0.125]
        self.case_2_novelty_type_dataset = custom_class_balance(case_2_all_novelty_type_dataset, case_2_novelty_type_proportions)
        case_3_novelty_type_proportions = [0.5, 0.167, 0.167, 0.167]
        self.case_3_novelty_type_dataset = custom_class_balance(case_3_all_novelty_type_dataset, case_3_novelty_type_proportions)

        self.feedback_data = None

        self.retraining_batch_size = retraining_batch_size 

    def add_feedback_data(self, data_root, csv_path):
        new_novel_dataset = noveltydetectionfeatures.NoveltyFeatureDataset(
            name = 'Custom',
            data_root = data_root,
            csv_path = csv_path,
            training = True,
            image_batch_size = 8,
            backbone = None,
            cache_to_disk = False
        )

        # Put new feedback data in list
        if self.feedback_data is None:
            self.feedback_data = new_novel_dataset
        else:
            self.feedback_data = torch.utils.data.ConcatDataset([self.feedback_data, new_novel_dataset])
    
    # Should be called before train_novelty_detection_module(), except when
    # training for the very first time manually by Alex. This prepares the
    # backbone, detector, and novelty type logistic regressions for retraining.
    # Most likely this is done by fully randomizing them, but in the future
    # we might change the process to be e.g. a warm-start, shrink-and-perturb,
    # or crashing a single layer.
    def prepare_for_retraining(self, backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model):
        # Construct a randomly initialized ResNet50 and transfer the weights to
        # backbone. Do the actual weight transfer on the CPU, since it's a big
        # network with a lot of weights and we don't want to run out of GPU
        # memory
        # random_backbone = resnet50(pretrained = True)
        # random_backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], backbone.fc.weight.shape[0])
        
        if self.model_ == 'resnet':
            random_backbone = resnet50(weights="IMAGENET1K_V1") # pretrained = True, 
            random_backbone.fc = torch.nn.Linear(backbone.fc.weight.shape[1], 256)
            device = backbone.fc.weight.device

        if self.model_ == 'swin_t':
            random_backbone = swin_t(weights="IMAGENET1K_V1") # pretrained = True, 
            random_backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
            device = backbone.head.weight.device

        if self.model_ == 'swin_b':
            random_backbone = swin_b(weights="IMAGENET1K_V1") # pretrained = True, 
            random_backbone.head = torch.nn.Linear(backbone.head.weight.shape[1], 256)
            device = backbone.head.weight.device

        state_dict = random_backbone.state_dict()
        backbone = backbone.to('cpu')
        backbone.load_state_dict(state_dict)
        backbone = backbone.to(device)
        
        # Randomize the classifiers and temperature scalers
        detector.classifier.subject_classifier = torch.nn.Linear(detector.classifier.subject_classifier.weight.shape[1], detector.classifier.subject_classifier.weight.shape[0]).to(device)
        detector.classifier.object_classifier = torch.nn.Linear(detector.classifier.object_classifier.weight.shape[1], detector.classifier.object_classifier.weight.shape[0]).to(device)
        detector.classifier.verb_classifier = torch.nn.Linear(detector.classifier.verb_classifier.weight.shape[1], detector.classifier.verb_classifier.weight.shape[0]).to(device)
        detector.confidence_calibrator.reset()

        # Construct random logistic regressions and transfer the weights
        random_case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(device)
        case_1_logistic_regression.load_state_dict(random_case_1_logistic_regression.state_dict())
        random_case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(device)
        case_2_logistic_regression.load_state_dict(random_case_2_logistic_regression.state_dict())
        random_case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(device)
        case_3_logistic_regression.load_state_dict(random_case_3_logistic_regression.state_dict())

        activation_statistical_model.reset()

    def train_epoch(self, known_subject_loader, novel_subject_loader, known_object_loader, novel_object_loader, known_verb_loader, novel_verb_loader, backbone, subject_classifier, object_classifier, verb_classifier, optimizer):
        # Determine the device to use based on the backbone's fc weights
        if self.model_ == 'resnet':
            device = backbone.fc.weight.device
        if self.model_ == 'swin_t' or self.model_ == 'swin_b':        
            device = backbone.head.weight.device
        
        # Set everything to train mode
        backbone.train()
        subject_classifier.train()
        object_classifier.train()
        verb_classifier.train()
        
        # Keep track of epoch statistics
        sum_loss = 0.0
        n_iterations = 0
        n_known_subject_examples = 0
        n_known_subject_correct = 0
        n_known_verb_examples = 0
        n_known_verb_correct = 0
        n_known_object_examples = 0
        n_known_object_correct = 0
        
        # Number of known classes for each
        n_known_subject_classes = 4
        n_known_object_classes  = 11
        n_known_verb_classes    = 7

        # The data loaders have different lengths, and so if we just zip them
        # all together, the zipped loader's length will be equal to that of the
        # minimum length of all the data loaders. So instead, we will iterate
        # over each data separately by constructing iterators manually and
        # iterating via next()
        known_subject_iter = iter(known_subject_loader)
        known_object_iter = iter(known_object_loader)
        known_verb_iter = iter(known_verb_loader)
        novel_subject_iter = iter(novel_subject_loader)
        novel_object_iter  = iter(novel_object_loader)
        novel_verb_iter    = iter(novel_verb_loader)
        
        # We need to iterate until we've covered all of the data in the largest
        # iterator. That's what we'll call an "epoch". We might iterate over
        # the smaller datasets multiple times, but that's fine. This keeps the
        # gradients in the feature extractor balanced across the tasks while
        # allowing it to see all the data in-between each validation step. So
        # keep track of which iterators we've fully iterated over at least once.
        finished_known_subject_iter = False
        finished_known_object_iter = False
        finished_known_verb_iter = False
        finished_novel_subject_iter = False
        finished_novel_object_iter  = False
        finished_novel_verb_iter    = False

        # Loop until all iterators have been covered
        while not (finished_known_subject_iter and\
                finished_known_object_iter and\
                finished_known_verb_iter and\
                finished_novel_subject_iter and\
                finished_novel_object_iter and\
                finished_novel_verb_iter):
            # Sample a batch from each iterator
            try:
                batch_known_subject_images, batch_known_subject_labels = next(known_subject_iter)
            except StopIteration:
                finished_known_subject_iter = True
                known_subject_iter = iter(known_subject_loader)
                batch_known_subject_images, batch_known_subject_labels = next(known_subject_iter)

            try:
                batch_known_object_images, batch_known_object_labels = next(known_object_iter)
            except StopIteration:
                finished_known_object_iter = True
                known_object_iter = iter(known_object_loader)
                batch_known_object_images, batch_known_object_labels = next(known_object_iter)

            try:
                batch_known_verb_images, batch_known_spatial_encodings, batch_known_verb_labels = next(known_verb_iter)
            except StopIteration:
                finished_known_verb_iter = True
                known_verb_iter = iter(known_verb_loader)
                batch_known_verb_images, batch_known_spatial_encodings, batch_known_verb_labels = next(known_verb_iter)

            try:
                batch_novel_subject_images, batch_novel_subject_labels = next(novel_subject_iter)
            except StopIteration:
                finished_novel_subject_iter = True
                novel_subject_iter = iter(novel_subject_loader)
                batch_novel_subject_images, batch_novel_subject_labels = next(novel_subject_iter)

            try:
                batch_novel_object_images, batch_novel_object_labels = next(novel_object_iter)
            except StopIteration:
                finished_novel_object_iter = True
                novel_object_iter = iter(novel_object_loader)
                batch_novel_object_images, batch_novel_object_labels = next(novel_object_iter)

            try:
                batch_novel_verb_images, batch_novel_spatial_encodings, batch_novel_verb_labels = next(novel_verb_iter)
            except StopIteration:
                finished_novel_verb_iter = True
                novel_verb_iter = iter(novel_verb_loader)
                batch_novel_verb_images, batch_novel_spatial_encodings, batch_novel_verb_labels = next(novel_verb_iter)

            # Shift labels down by 1 for closed-set classifier, since known
            # class indices start at 1 but the classifiers outputs the number
            # of logits equal to the corresponding number of known classes
            batch_known_subject_images = batch_known_subject_images.to(device)
            batch_known_subject_labels = batch_known_subject_labels.to(device) - 1
            batch_known_object_images = batch_known_object_images.to(device)
            batch_known_object_labels = batch_known_object_labels.to(device) - 1
            batch_known_verb_images = batch_known_verb_images.to(device)
            batch_known_spatial_encodings = batch_known_spatial_encodings.to(device)
            batch_known_verb_labels = batch_known_verb_labels.to(device) - 1
            
            # Extract features from each image batch, making sure to prepend
            # the spatial encodings to the verb features, since the verb
            # classifier operates over the spatial encodings and the verb
            # box features.
            batch_known_subject_features = backbone(batch_known_subject_images)
            batch_known_object_features = backbone(batch_known_object_images)
            batch_known_verb_features = backbone(batch_known_verb_images)
            batch_known_spatial_encodings = torch.flatten(batch_known_spatial_encodings, start_dim = 1)
            batch_known_verb_features = torch.cat((batch_known_spatial_encodings, batch_known_verb_features), dim = 1)

            # All novel labels should be uniform since we want to use outlier exposure
            batch_novel_subject_images = batch_novel_subject_images.to(device)
            batch_novel_subject_labels = (1/n_known_subject_classes) * torch.ones((batch_novel_subject_images.shape[0],n_known_subject_classes))
            batch_novel_subject_labels = batch_novel_subject_labels.to(device)

            batch_novel_object_images = batch_novel_object_images.to(device)
            batch_novel_object_labels = (1/n_known_object_classes) * torch.ones((batch_novel_object_images.shape[0],n_known_object_classes))
            batch_novel_object_labels = batch_novel_object_labels.to(device)

            batch_novel_verb_images = batch_novel_verb_images.to(device)
            batch_novel_spatial_encodings = batch_novel_spatial_encodings.to(device)
            batch_novel_verb_labels = (1/n_known_verb_classes) * torch.ones((batch_novel_verb_images.shape[0],n_known_verb_classes))
            batch_novel_verb_labels = batch_novel_verb_labels.to(device)

            # Extract features from each novel image batch, making sure to prepend
            # the spatial encodings to the verb features, since the verb
            # classifier operates over the spatial encodings and the verb
            # box features.
            batch_novel_subject_features = backbone(batch_novel_subject_images)
            batch_novel_object_features = backbone(batch_novel_object_images)
            batch_novel_verb_features = backbone(batch_novel_verb_images)
            batch_novel_spatial_encodings = torch.flatten(batch_novel_spatial_encodings, start_dim = 1)
            batch_novel_verb_features = torch.cat((batch_novel_spatial_encodings, batch_novel_verb_features), dim = 1)

            # Compute logits by passing the features through the appropriate
            # classifiers
            batch_known_subject_preds = subject_classifier(batch_known_subject_features)
            batch_known_object_preds = object_classifier(batch_known_object_features)
            batch_known_verb_preds = verb_classifier(batch_known_verb_features)
            
            batch_known_subject_loss = torch.nn.functional.cross_entropy(batch_known_subject_preds, batch_known_subject_labels)
            batch_known_object_loss = torch.nn.functional.cross_entropy(batch_known_object_preds, batch_known_object_labels)
            batch_known_verb_loss = torch.nn.functional.cross_entropy(batch_known_verb_preds, batch_known_verb_labels)

            # Compute logits by passing the features through the appropriate
            # classifiers
            batch_novel_subject_preds = subject_classifier(batch_novel_subject_features)
            batch_novel_object_preds = object_classifier(batch_novel_object_features)
            batch_novel_verb_preds = verb_classifier(batch_novel_verb_features)

            # batch_novel_subject_loss = torch.nn.functional.cross_entropy(batch_novel_subject_preds, batch_novel_subject_labels)
            # batch_novel_object_loss = torch.nn.functional.cross_entropy(batch_novel_object_preds, batch_novel_object_labels)
            # batch_novel_verb_loss = torch.nn.functional.cross_entropy(batch_novel_verb_preds, batch_novel_verb_labels)
            batch_novel_subject_loss = custom_cross_entropy(batch_novel_subject_preds, batch_novel_subject_labels)
            batch_novel_object_loss = custom_cross_entropy(batch_novel_object_preds, batch_novel_object_labels)
            batch_novel_verb_loss = custom_cross_entropy(batch_novel_verb_preds, batch_novel_verb_labels)

            batch_loss = batch_known_subject_loss + batch_known_verb_loss + batch_known_object_loss \
                         + batch_novel_subject_loss + batch_novel_object_loss + batch_novel_verb_loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            sum_loss += batch_loss.detach().cpu().item()
            n_iterations += 1

            n_known_subject_examples += batch_known_subject_images.shape[0]
            n_known_object_examples += batch_known_object_images.shape[0]
            n_known_verb_examples += batch_known_verb_images.shape[0]

            n_known_subject_correct += int((torch.argmax(batch_known_subject_preds, dim = 1) == batch_known_subject_labels).to(torch.int).sum().detach().cpu().item())
            n_known_object_correct += int((torch.argmax(batch_known_object_preds, dim = 1) == batch_known_object_labels).to(torch.int).sum().detach().cpu().item())
            n_known_verb_correct += int((torch.argmax(batch_known_verb_preds, dim = 1) == batch_known_verb_labels).to(torch.int).sum().detach().cpu().item())

        mean_loss = sum_loss / n_iterations

        mean_known_subject_accuracy = float(n_known_subject_correct) / n_known_subject_examples
        mean_known_object_accuracy = float(n_known_object_correct) / n_known_object_examples
        mean_known_verb_accuracy = float(n_known_verb_correct) / n_known_verb_examples

        mean_known_accuracy = (mean_known_subject_accuracy + mean_known_object_accuracy + mean_known_verb_accuracy) / 3.0

        return mean_loss, mean_known_accuracy

    def val_epoch(self, subject_loader, object_loader, verb_loader, backbone, subject_classifier, object_classifier, verb_classifier):
        with torch.no_grad():
            backbone.eval()
            subject_classifier.eval()
            object_classifier.eval()
            verb_classifier.eval()

            if self.model_ == 'resnet':
                device = backbone.fc.weight.device
            if self.model_ == 'swin_t' or self.model_ == 'swin_b':        
                device = backbone.head.weight.device

            
            n_subject_examples = 0
            n_subject_correct = 0
            n_object_examples = 0
            n_object_correct = 0
            n_verb_examples = 0
            n_verb_correct = 0
            
            for batch_subject_images, batch_subject_labels in subject_loader:
                batch_subject_images = batch_subject_images.to(device)
                batch_subject_labels = batch_subject_labels.to(device) - 1
                
                batch_subject_features = backbone(batch_subject_images)
                batch_subject_preds = subject_classifier(batch_subject_features)
                n_subject_examples += batch_subject_images.shape[0]
                n_subject_correct += int((torch.argmax(batch_subject_preds, dim = 1) == batch_subject_labels).to(torch.int).sum().detach().cpu().item())

            mean_subject_accuracy = float(n_subject_correct) / n_subject_examples

            for batch_object_images, batch_object_labels in object_loader:
                batch_object_images = batch_object_images.to(device)
                batch_object_labels = batch_object_labels.to(device) - 1
                
                batch_object_features = backbone(batch_object_images)
                batch_object_preds = object_classifier(batch_object_features)
                n_object_examples += batch_object_images.shape[0]
                n_object_correct += int((torch.argmax(batch_object_preds, dim = 1) == batch_object_labels).to(torch.int).sum().detach().cpu().item())

            mean_object_accuracy = float(n_object_correct) / n_object_examples

            for batch_verb_images, batch_spatial_encodings, batch_verb_labels in verb_loader:
                batch_verb_images = batch_verb_images.to(device)
                batch_spatial_encodings = batch_spatial_encodings.to(device)
                batch_verb_labels = batch_verb_labels.to(device) - 1
                
                batch_verb_features = backbone(batch_verb_images)
                batch_spatial_encodings = torch.flatten(batch_spatial_encodings, start_dim = 1)
                batch_verb_features = torch.cat((batch_spatial_encodings, batch_verb_features), dim = 1)
                batch_verb_preds = verb_classifier(batch_verb_features)
                n_verb_examples += batch_verb_images.shape[0]
                n_verb_correct += int((torch.argmax(batch_verb_preds, dim = 1) == batch_verb_labels).to(torch.int).sum().detach().cpu().item())

            mean_verb_accuracy = float(n_verb_correct) / n_verb_examples

            mean_accuracy = (mean_subject_accuracy + mean_verb_accuracy + mean_object_accuracy) / 3.0
            
            return mean_accuracy

    def train_backbone_and_classifiers(self, backbone, subject_classifier, object_classifier, verb_classifier):
        # NOTE Can potentially refactor for speedup (by calling separate_data in add_feedback
        # and keeping track of novel feedback s,v, and o datasets as class attributes)
        if self.feedback_data is not None:
            _, novel_feedback_subject_dataset, \
                _, novel_feedback_object_dataset, \
                _, novel_feedback_verb_dataset = separate_data(self.feedback_data)
            
            novel_subject_dataset = torch.utils.data.ConcatDataset([self.novel_train_subject_dataset, novel_feedback_subject_dataset])
            novel_object_dataset  = torch.utils.data.ConcatDataset([self.novel_train_object_dataset, novel_feedback_object_dataset])
            novel_verb_dataset    = torch.utils.data.ConcatDataset([self.novel_train_verb_dataset, novel_feedback_verb_dataset])
        else:
            novel_subject_dataset = self.novel_train_subject_dataset
            novel_object_dataset  = self.novel_train_object_dataset
            novel_verb_dataset    = self.novel_train_verb_dataset

        # Construct known dataloaders. Since we're not using any known feedback
        # data, these are constructed solely from the known OSU training data.

        ks_size = len(self.known_train_subject_dataset)
        ko_size = len(self.known_train_object_dataset)
        kv_size = len(self.known_train_verb_dataset)

        ns_size = len(novel_subject_dataset)
        no_size = len(novel_object_dataset)
        nv_size = len(novel_verb_dataset)

        total_batch_size = self.retraining_batch_size

        ks_batch_size, ko_batch_size, kv_batch_size, \
            ns_batch_size, no_batch_size, nv_batch_size = self.compute_balanced_batch_sizes(total_batch_size, ks_size, ko_size, kv_size, ns_size, no_size, nv_size)

        known_train_subject_loader = torch.utils.data.DataLoader(
            self.known_train_subject_dataset,
            batch_size = ks_batch_size,
            shuffle = True
        )
        known_train_object_loader = torch.utils.data.DataLoader(
            self.known_train_object_dataset,
            batch_size = ko_batch_size,
            shuffle = True
        )
        known_train_verb_loader = torch.utils.data.DataLoader(
            self.known_train_verb_dataset,
            batch_size = kv_batch_size,
            shuffle = True
        )

        novel_subject_loader = torch.utils.data.DataLoader(
            novel_subject_dataset,
            batch_size = ns_batch_size,
            shuffle = True
        )
        novel_object_loader = torch.utils.data.DataLoader(
            novel_object_dataset,
            batch_size = no_batch_size,
            shuffle = True
        )
        novel_verb_loader = torch.utils.data.DataLoader(
            novel_verb_dataset,
            batch_size = nv_batch_size,
            shuffle = True
        )

        # Construct validation loaders for early stopping / model selection.
        # I'm assuming our model selection strategy will be based solely on the
        # validation classification accuracy and not based on novelty detection
        # capabilities in any way. Otherwise, we can use the novel validation
        # data to measure novelty detection performance. These currently aren't
        # being stored (except in a special form for the logistic regressions),
        # so we'd have to modify __init__().
        known_val_subject_loader = torch.utils.data.DataLoader(
            self.known_val_subject_dataset,
            batch_size = 256,
            shuffle = False
        )
        known_val_object_loader = torch.utils.data.DataLoader(
            self.known_val_object_dataset,
            batch_size = 256,
            shuffle = False
        )
        known_val_verb_loader = torch.utils.data.DataLoader(
            self.known_val_verb_dataset,
            batch_size = 256,
            shuffle = False
        )

        # Retrain the backbone and classifiers
        # Construct the optimizer
        optimizer = torch.optim.SGD(list(backbone.parameters()) + list(subject_classifier.parameters()) + list(object_classifier.parameters()) + list(verb_classifier.parameters()),
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
        best_accuracy_subject_classifier_state_dict = None
        best_accuracy_object_classifier_state_dict = None
        best_accuracy_verb_classifier_state_dict = None
        
        # Retrain
        progress = tqdm(range(max_epochs), desc = 'Training backbone and classifiers...')
        for epoch in progress:
            # Train for one full epoch
            mean_train_loss, mean_known_train_accuracy = self.train_epoch(known_train_subject_loader, novel_subject_loader, 
                known_train_object_loader, novel_object_loader,
                known_train_verb_loader, novel_verb_loader, 
                backbone, subject_classifier, object_classifier, verb_classifier, 
                optimizer)
                
            # progress.set_description(f'Training backbone and classifiers... | Train Loss: {mean_train_loss} | Train Acc: {mean_known_train_accuracy}')
            
            # Measure validation accuracy for early stopping / model selection.
            # I'm assuming we don't need to use the novel data here.
            if epoch >= min_epochs - 1:
                mean_val_accuracy = self.val_epoch(known_val_subject_loader,
                    known_val_object_loader, known_val_verb_loader, backbone,
                    subject_classifier, object_classifier, verb_classifier)
                
                progress.set_description(f'Training backbone and classifiers... | Train Loss: {mean_train_loss} | Train Acc: {mean_known_train_accuracy} | Val Acc: {mean_val_accuracy}')
                
                if best_accuracy is None or mean_val_accuracy > best_accuracy:
                    epochs_since_improvement = 0
                    best_accuracy = mean_val_accuracy
                    best_accuracy_backbone_state_dict = copy.deepcopy(backbone.state_dict())
                    best_accuracy_subject_classifier_state_dict = copy.deepcopy(subject_classifier.state_dict())
                    best_accuracy_object_classifier_state_dict = copy.deepcopy(object_classifier.state_dict())
                    best_accuracy_verb_classifier_state_dict = copy.deepcopy(verb_classifier.state_dict())
                else:
                    epochs_since_improvement += 1
                    if epochs_since_improvement >= patience:
                        # We haven't improved in several epochs. Time to stop
                        # training.
                        break
        progress.close()

        # Load the best accuracy state dicts
        backbone.load_state_dict(best_accuracy_backbone_state_dict)
        subject_classifier.load_state_dict(best_accuracy_subject_classifier_state_dict)
        object_classifier.load_state_dict(best_accuracy_object_classifier_state_dict)
        verb_classifier.load_state_dict(best_accuracy_verb_classifier_state_dict)

    def compute_balanced_batch_sizes(self, total_batch_size, ks_size, ko_size, kv_size, ns_size, no_size, nv_size):
        total_size = ks_size + ko_size + kv_size + ns_size + no_size + nv_size
        ks_batch_size = (total_batch_size * ks_size) // total_size
        ko_batch_size = (total_batch_size * ko_size) // total_size
        kv_batch_size = (total_batch_size * kv_size) // total_size
        ns_batch_size = (total_batch_size * ns_size) // total_size
        no_batch_size = (total_batch_size * no_size) // total_size
        nv_batch_size = (total_batch_size * nv_size) // total_size
        
        if ks_batch_size == 0 and ks_size > 0:
            ks_batch_size = 1
        if ko_batch_size == 0 and ko_size > 0:
            ks_batch_size = 1
        if kv_batch_size == 0 and kv_size > 0:
            kv_batch_size = 1
        if ns_batch_size == 0 and ns_size > 0:
            ns_batch_size = 1
        if no_batch_size == 0 and no_size > 0:
            no_batch_size = 1
        if nv_batch_size == 0 and nv_size > 0:
            nv_batch_size = 1

        return ks_batch_size, ko_batch_size, kv_batch_size, ns_batch_size, no_batch_size, nv_batch_size

    def fit_activation_statistics(self, backbone, activation_statistical_model):
        activation_stats_training_loader = torch.utils.data.DataLoader(
            self.activation_stats_training_dataset,
            batch_size = 32,
            shuffle = False
        )
        backbone.eval()

        if self.model_ == 'resnet':
            device = backbone.fc.weight.device
        if self.model_ == 'swin_t' or self.model_ == 'swin_b':        
            device = backbone.head.weight.device

        all_features = []
        with torch.no_grad():
            for batch in activation_stats_training_loader:
                batch = batch.to(device)
                features = activation_statistical_model.compute_features(
                    backbone,
                    batch
                )
                all_features.append(features)
        all_features = torch.cat(all_features, dim=0)
        activation_statistical_model.fit(all_features)

    def calibrate_temperature_scalers(self, backbone, subject_classifier, object_classifier, verb_classifier, subject_calibrator, object_calibrator, verb_calibrator):
        calibration_subject_loader = torch.utils.data.DataLoader(
            self.calibration_subject_dataset,
            batch_size = 32,
            shuffle = False
        )
        calibration_object_loader = torch.utils.data.DataLoader(
            self.calibration_object_dataset,
            batch_size = 32,
            shuffle = False
        )
        calibration_verb_loader = torch.utils.data.DataLoader(
            self.calibration_verb_dataset,
            batch_size = 32,
            shuffle = False
        )

        # Set everything to eval mode for calibration, except the calibrators
        backbone.eval()
        subject_classifier.eval()
        object_classifier.eval()
        verb_classifier.eval()
        subject_calibrator.train()
        object_calibrator.train()
        verb_calibrator.train()

        if self.model_ == 'resnet':
            device = backbone.fc.weight.device
        if self.model_ == 'swin_t' or self.model_ == 'swin_b':        
            device = backbone.head.weight.device

        # Extract logits to fit confidence calibration temperatures
        with torch.no_grad():
            subject_logits = []
            subject_labels = []
            for batch_subject_images, batch_subject_labels in calibration_subject_loader:
                batch_subject_images = batch_subject_images.to(device)
                batch_subject_labels = batch_subject_labels.to(device) - 1
                batch_subject_features = backbone(batch_subject_images)
                batch_subject_logits = subject_classifier(batch_subject_features)
                subject_logits.append(batch_subject_logits)
                subject_labels.append(batch_subject_labels)
            subject_logits = torch.cat(subject_logits, dim = 0)
            subject_labels = torch.cat(subject_labels, dim = 0)

            object_logits = []
            object_labels = []
            for batch_object_images, batch_object_labels in calibration_object_loader:
                batch_object_images = batch_object_images.to(device)
                batch_object_labels = batch_object_labels.to(device) - 1
                batch_object_features = backbone(batch_object_images)
                batch_object_logits = object_classifier(batch_object_features)
                object_logits.append(batch_object_logits)
                object_labels.append(batch_object_labels)
            object_logits = torch.cat(object_logits, dim = 0)
            object_labels = torch.cat(object_labels, dim = 0)

            verb_logits = []
            verb_labels = []
            for batch_verb_images, batch_spatial_encodings, batch_verb_labels in calibration_verb_loader:
                batch_verb_images = batch_verb_images.to(device)
                batch_spatial_encodings = batch_spatial_encodings.to(device)
                batch_verb_labels = batch_verb_labels.to(device) - 1
                batch_verb_features = backbone(batch_verb_images)
                batch_spatial_encodings = torch.flatten(batch_spatial_encodings, start_dim = 1)
                batch_verb_features = torch.cat((batch_spatial_encodings, batch_verb_features), dim = 1)
                batch_verb_logits = verb_classifier(batch_verb_features)
                verb_logits.append(batch_verb_logits)
                verb_labels.append(batch_verb_labels)
            verb_logits = torch.cat(verb_logits, dim = 0)
            verb_labels = torch.cat(verb_labels, dim = 0)

        subject_calibrator_optimizer = torch.optim.SGD(subject_calibrator.parameters(),
            0.001,
            momentum=0.9)
        object_calibrator_optimizer = torch.optim.SGD(object_calibrator.parameters(),
            0.001,
            momentum=0.9)
        verb_calibrator_optimizer = torch.optim.SGD(verb_calibrator.parameters(),
            0.001,
            momentum=0.9)
        
        progress = tqdm(range(10000), desc = 'Training calibrators...')
        for epoch in progress:
            scaled_subject_logits = subject_calibrator(subject_logits)
            subject_loss = torch.nn.functional.cross_entropy(scaled_subject_logits, subject_labels)
            subject_calibrator_optimizer.zero_grad()
            subject_loss.backward()
            subject_calibrator_optimizer.step()

            scaled_object_logits = object_calibrator(object_logits)
            object_loss = torch.nn.functional.cross_entropy(scaled_object_logits, object_labels)
            object_calibrator_optimizer.zero_grad()
            object_loss.backward()
            object_calibrator_optimizer.step()

            scaled_verb_logits = verb_calibrator(verb_logits)
            verb_loss = torch.nn.functional.cross_entropy(scaled_verb_logits, verb_labels)
            verb_calibrator_optimizer.zero_grad()
            verb_loss.backward()
            verb_calibrator_optimizer.step()

            avg_loss = (subject_loss + verb_loss + object_loss) / 3.0
            progress.set_description(f'Training calibrators... | Loss: {avg_loss.detach().cpu().item()}')

    def train_novelty_type_logistic_regressions(self, backbone, subject_classifier, object_classifier, verb_classifier, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model):
        # Set the backbone and classifiers to eval(), but set the logistic
        # regressions to train()
        backbone.eval()
        subject_classifier.eval()
        object_classifier.eval()
        verb_classifier.eval()
        case_1_logistic_regression.train()
        case_2_logistic_regression.train()
        case_3_logistic_regression.train()

        if self.model_ == 'resnet':
            device = backbone.fc.weight.device
        if self.model_ == 'swin_t' or self.model_ == 'swin_b':        
            device = backbone.head.weight.device

        case_1_novelty_type_loader = torch.utils.data.DataLoader(
            self.case_1_novelty_type_dataset,
            batch_size = 32,
            shuffle = False
        )
        case_2_novelty_type_loader = torch.utils.data.DataLoader(
            self.case_2_novelty_type_dataset,
            batch_size = 32,
            shuffle = False
        )
        case_3_novelty_type_loader = torch.utils.data.DataLoader(
            self.case_3_novelty_type_dataset,
            batch_size = 32,
            shuffle = False
        )
        
        with torch.no_grad():
            # Extract case 1 novelty scores
            case_1_scores = []
            case_1_labels = []
            for batch_images, batch_spatial_encodings, batch_labels in case_1_novelty_type_loader:
                batch_images = batch_images.to(device)
                batch_box_images = batch_images[:, :3]
                batch_whole_images = batch_images[:, 3]
                batch_spatial_encodings = batch_spatial_encodings.to(device)
                batch_labels = batch_labels.to(device)

                # The case 1 images are [N, 3, 3, 224, 224], where the first "3"
                # denotes subject, object, and verb box images separately. Flatten
                # the images so that they can all be processed by the backbone
                # at once.
                batch_flattened_images = batch_box_images.reshape(-1, *batch_box_images.shape[2:])
                
                # Extract features from flattened images
                batch_flattened_features = backbone(batch_flattened_images)

                # Reshape the flattened features to be [N, 3, D], where D is the
                # number of features (bottleneck dimension)
                batch_features = batch_flattened_features.reshape(-1, 3, batch_flattened_features.shape[-1])

                # Extract S/V/O logits from the three feature vectors for each
                # instance, concatenating the spatial encodings to the verb
                # box features
                batch_subject_logits = subject_classifier(batch_features[:, 0, :])
                batch_object_logits = object_classifier(batch_features[:, 1, :])
                batch_spatial_encodings = torch.flatten(batch_spatial_encodings, start_dim = 1)
                batch_verb_logits = verb_classifier(torch.cat((batch_spatial_encodings, batch_features[:, 2, :]), dim = 1))

                # Compute negative max logits (novelty scores) from each classifier
                batch_max_subject_logits, _ = torch.max(batch_subject_logits, dim = 1)
                batch_subject_novelty_scores = -batch_max_subject_logits
                batch_max_object_logits, _ = torch.max(batch_object_logits, dim = 1)
                batch_object_novelty_scores = -batch_max_object_logits
                batch_max_verb_logits, _ = torch.max(batch_verb_logits, dim = 1)
                batch_verb_novelty_scores = -batch_max_verb_logits

                # Compute whole-image activation statistic scores
                batch_whole_image_features = activation_statistical_model.compute_features(backbone, batch_whole_images)
                batch_activation_statistical_scores = torch.from_numpy(activation_statistical_model.score(batch_whole_image_features)).to(torch.float).to(device)

                # Concatenate scores for the case 1 logistic regression input.
                batch_scores = torch.stack((batch_subject_novelty_scores, batch_verb_novelty_scores, batch_object_novelty_scores, batch_activation_statistical_scores), dim = 1)

                case_1_scores.append(batch_scores)
                case_1_labels.append(batch_labels)
            
            case_1_scores = torch.cat(case_1_scores, dim = 0)
            case_1_labels = torch.cat(case_1_labels, dim = 0)

            # Extract case 2 novelty scores
            case_2_scores = []
            case_2_labels = []
            for batch_images, batch_spatial_encodings, batch_labels in case_2_novelty_type_loader:
                batch_images = batch_images.to(device)
                batch_box_images = batch_images[:, :2]
                batch_whole_images = batch_images[:, 2]
                batch_spatial_encodings = batch_spatial_encodings.to(device)
                batch_labels = batch_labels.to(device)

                # The case 2 images are [N, 2, 3, 224, 224], where the "2"
                # indexes subject and verb box images separately. Flatten
                # the images so that they can all be processed by the backbone
                # at once.
                batch_flattened_images = batch_box_images.reshape(-1, *batch_box_images.shape[2:])
                
                # Extract features from flattened images
                batch_flattened_features = backbone(batch_flattened_images)

                # Reshape the flattened features to be [N, 2, D], where D is the
                # number of features (bottleneck dimension)
                batch_features = batch_flattened_features.reshape(-1, 2, batch_flattened_features.shape[-1])

                # Extract S/V logits from the two feature vectors for each
                # instance, concatenating the spatial encodings to the verb
                # box features
                batch_subject_logits = subject_classifier(batch_features[:, 0, :])
                batch_spatial_encodings = torch.flatten(batch_spatial_encodings, start_dim = 1)
                batch_verb_logits = verb_classifier(torch.cat((batch_spatial_encodings, batch_features[:, 1, :]), dim = 1))

                # Compute negative max logits (novelty scores) from each classifier
                batch_max_subject_logits, _ = torch.max(batch_subject_logits, dim = 1)
                batch_subject_novelty_scores = -batch_max_subject_logits
                batch_max_verb_logits, _ = torch.max(batch_verb_logits, dim = 1)
                batch_verb_novelty_scores = -batch_max_verb_logits

                # Compute whole-image activation statistic scores
                batch_whole_image_features = activation_statistical_model.compute_features(backbone, batch_whole_images)
                batch_activation_statistical_scores = torch.from_numpy(activation_statistical_model.score(batch_whole_image_features)).to(torch.float).to(device)
                
                # Concatenate scores for the case 2 logistic regression input.
                batch_scores = torch.stack((batch_subject_novelty_scores, batch_verb_novelty_scores, batch_activation_statistical_scores), dim = 1)

                case_2_scores.append(batch_scores)
                case_2_labels.append(batch_labels)
            
            case_2_scores = torch.cat(case_2_scores, dim = 0)
            case_2_labels = torch.cat(case_2_labels, dim = 0)

            # Extract case 3 novelty scores
            case_3_scores = []
            case_3_labels = []
            for batch_images, batch_labels in case_3_novelty_type_loader:
                batch_images = batch_images.to(device)
                batch_box_images = batch_images[:, 0]
                batch_whole_images = batch_images[:, 1]
                batch_labels = batch_labels.to(device)

                # Extract features from images
                batch_features = backbone(batch_box_images)

                # Extract logits from the object classifier
                batch_object_logits = object_classifier(batch_features)
                
                # Compute negative max logits (novelty scores) from the object
                # classifier
                batch_max_object_logits, _ = torch.max(batch_object_logits, dim = 1)
                batch_object_novelty_scores = -batch_max_object_logits

                # Compute whole-image activation statistic scores
                batch_whole_image_features = activation_statistical_model.compute_features(backbone, batch_whole_images)
                batch_activation_statistical_scores = torch.from_numpy(activation_statistical_model.score(batch_whole_image_features)).to(torch.float).to(device)
                
                # Concatenate scores for the case 3 logistic regression input.
                batch_scores = torch.stack((batch_object_novelty_scores, batch_activation_statistical_scores), dim = 1)

                case_3_scores.append(batch_scores)
                case_3_labels.append(batch_labels)
            
            case_3_scores = torch.cat(case_3_scores, dim = 0)
            case_3_labels = torch.cat(case_3_labels, dim = 0)

        # Fit the three logistic regressions
        print('Fitting novelty type logistic regressions...')
        fit_logistic_regression(case_1_logistic_regression, case_1_scores, case_1_labels, epochs = 3000)
        fit_logistic_regression(case_2_logistic_regression, case_2_scores, case_2_labels, epochs = 3000)
        fit_logistic_regression(case_3_logistic_regression, case_3_scores, case_3_labels, epochs = 3000)

    def train_novelty_detection_module(self, backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model):
        subject_classifier = detector.classifier.subject_classifier
        object_classifier = detector.classifier.object_classifier
        verb_classifier = detector.classifier.verb_classifier
        subject_calibrator = detector.confidence_calibrator.subject_calibrator
        object_calibrator = detector.confidence_calibrator.object_calibrator
        verb_calibrator = detector.confidence_calibrator.verb_calibrator
        
        # Retrain the backbone and classifiers
        self.train_backbone_and_classifiers(backbone, subject_classifier, object_classifier, verb_classifier)

        self.fit_activation_statistics(backbone, activation_statistical_model)
        
        # Retrain the detector's temperature scaling calibrators
        self.calibrate_temperature_scalers(backbone, subject_classifier, object_classifier, verb_classifier, subject_calibrator, object_calibrator, verb_calibrator)

        # Retrain the logistic regressions
        self.train_novelty_type_logistic_regressions(backbone, subject_classifier, object_classifier, verb_classifier, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression, activation_statistical_model)
