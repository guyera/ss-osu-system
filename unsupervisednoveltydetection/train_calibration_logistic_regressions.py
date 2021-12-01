import torch
import unsupervisednoveltydetection
import noveltydetectionfeatures
import sklearn.metrics
import noveltydetection
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser('Train calibration logistic regressions')

parser.add_argument(
    '--device',
    type = str,
    default = 'cpu'
)
parser.add_argument(
    '--epochs',
    type = int,
    default = 5000
)
parser.add_argument(
    '--lr',
    type = float,
    default = 0.01
)
parser.add_argument(
    '--detector-load-file',
    type = str,
    required = True
)
parser.add_argument(
    '--calibration-logistic-regressions-save-file',
    type = str,
    required = True
)

args = parser.parse_args()

detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(12544, 12616, 1024, 5, 12, 8)
detector = detector.to(args.device)

state_dict = torch.load(args.detector_load_file)
detector.load_state_dict(state_dict)

testing_set = noveltydetectionfeatures.NoveltyFeatureDataset(
    name = 'Custom',
    data_root = 'Custom',
    csv_path = 'Custom/annotations/dataset_v3_val.csv',
    training = False,
    image_batch_size = 16,
    feature_extraction_device = args.device
)

spatial_features = []
subject_appearance_features = []
object_appearance_features = []
verb_appearance_features = []
subject_labels = []
object_labels = []
verb_labels = []
        
for example_spatial_features, example_subject_appearance_features, example_object_appearance_features, example_verb_appearance_features, subject_label, object_label, verb_label in testing_set:
    spatial_features.append(example_spatial_features)
    subject_appearance_features.append(example_subject_appearance_features)
    object_appearance_features.append(example_object_appearance_features)
    verb_appearance_features.append(example_verb_appearance_features)
    subject_labels.append(subject_label)
    object_labels.append(object_label)
    verb_labels.append(verb_label)

results = detector(spatial_features, subject_appearance_features, verb_appearance_features, object_appearance_features, torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], device = args.device))

subject_scores = results['subject_novelty_score']
object_scores = results['object_novelty_score']
verb_scores = results['verb_novelty_score']

case_1_type_0_scores = []
case_1_type_1_scores = []
case_1_type_2_scores = []
case_1_type_3_scores = []
case_2_type_0_scores = []
case_2_type_1_scores = []
case_2_type_2_scores = []
case_3_type_0_scores = []
case_3_type_3_scores = []

for idx in range(len(subject_scores)):
    subject_score = subject_scores[idx]
    object_score = object_scores[idx]
    verb_score = verb_scores[idx]
    subject_label = subject_labels[idx]
    object_label = object_labels[idx]
    verb_label = verb_labels[idx]
    
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

    # Add the scores and labels to the case inputs
    if subject_score is not None and object_score is not None:
        # All boxes present; append scores to case 1
        if type_label == 0:
            case_1_type_0_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
        elif type_label == 1:
            case_1_type_1_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
        elif type_label == 2:
            case_1_type_2_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
        elif type_label == 3:
            case_1_type_3_scores.append(torch.stack((subject_score, verb_score, object_score), dim = 0))
    if subject_score is not None:
        # At least the subject box is present; append subject and verb scores
        # to case 2
        if type_label == 0:
            case_2_type_0_scores.append(torch.stack((subject_score, verb_score), dim = 0))
        elif type_label == 1:
            case_2_type_1_scores.append(torch.stack((subject_score, verb_score), dim = 0))
        elif type_label == 2:
            case_2_type_2_scores.append(torch.stack((subject_score, verb_score), dim = 0))
    if object_score is not None:
        # At least the object box is present; append object score to case 3
        if type_label == 0:
            case_3_type_0_scores.append(object_score.unsqueeze(0))
        if type_label == 3:
            case_3_type_3_scores.append(object_score.unsqueeze(0))

case_1_type_0_scores = torch.stack(case_1_type_0_scores, dim = 0)
case_1_type_1_scores = torch.stack(case_1_type_1_scores, dim = 0)
case_1_type_2_scores = torch.stack(case_1_type_2_scores, dim = 0)
case_1_type_3_scores = torch.stack(case_1_type_3_scores, dim = 0)
case_2_type_0_scores = torch.stack(case_2_type_0_scores, dim = 0)
case_2_type_1_scores = torch.stack(case_2_type_1_scores, dim = 0)
case_2_type_2_scores = torch.stack(case_2_type_2_scores, dim = 0)
case_3_type_0_scores = torch.stack(case_3_type_0_scores, dim = 0)
case_3_type_3_scores = torch.stack(case_3_type_3_scores, dim = 0)

# Randomly shuffle score rows
case_1_type_0_scores = case_1_type_0_scores[torch.randperm(len(case_1_type_0_scores), device = args.device)]
case_1_type_1_scores = case_1_type_1_scores[torch.randperm(len(case_1_type_1_scores), device = args.device)]
case_1_type_2_scores = case_1_type_2_scores[torch.randperm(len(case_1_type_2_scores), device = args.device)]
case_1_type_3_scores = case_1_type_3_scores[torch.randperm(len(case_1_type_3_scores), device = args.device)]
case_2_type_0_scores = case_2_type_0_scores[torch.randperm(len(case_2_type_0_scores), device = args.device)]
case_2_type_1_scores = case_2_type_1_scores[torch.randperm(len(case_2_type_1_scores), device = args.device)]
case_2_type_2_scores = case_2_type_2_scores[torch.randperm(len(case_2_type_2_scores), device = args.device)]
case_3_type_0_scores = case_3_type_0_scores[torch.randperm(len(case_3_type_0_scores), device = args.device)]
case_3_type_3_scores = case_3_type_3_scores[torch.randperm(len(case_3_type_3_scores), device = args.device)]

# Now, we need to balance classes. Compute novelty type with fewest instances in
# each case
case_1_n_per_class = min(len(case_1_type_0_scores), len(case_1_type_1_scores), len(case_1_type_2_scores), len(case_1_type_3_scores))
case_2_n_per_class = min(len(case_2_type_0_scores), len(case_2_type_1_scores), len(case_2_type_2_scores))
case_3_n_per_class = min(len(case_3_type_0_scores), len(case_3_type_3_scores))

# Balance the classes
case_1_type_0_scores = case_1_type_0_scores[:case_1_n_per_class]
case_1_type_1_scores = case_1_type_1_scores[:case_1_n_per_class]
case_1_type_2_scores = case_1_type_2_scores[:case_1_n_per_class]
case_1_type_3_scores = case_1_type_3_scores[:case_1_n_per_class]
case_2_type_0_scores = case_2_type_0_scores[:case_2_n_per_class]
case_2_type_1_scores = case_2_type_1_scores[:case_2_n_per_class]
case_2_type_2_scores = case_2_type_2_scores[:case_2_n_per_class]
case_3_type_0_scores = case_3_type_0_scores[:case_3_n_per_class]
case_3_type_3_scores = case_3_type_3_scores[:case_3_n_per_class]

# Construct concatenated score tensors and label tensors
case_1_scores = torch.cat((case_1_type_0_scores, case_1_type_1_scores, case_1_type_2_scores, case_1_type_3_scores), dim = 0).detach()
case_2_scores = torch.cat((case_2_type_0_scores, case_2_type_1_scores, case_2_type_2_scores), dim = 0).detach()
case_3_scores = torch.cat((case_3_type_0_scores, case_3_type_3_scores), dim = 0).detach()
case_1_labels = torch.cat((torch.full(size = (len(case_1_type_0_scores),), fill_value = 0, dtype = torch.long, device = args.device), torch.full(size = (len(case_1_type_1_scores),), fill_value = 1, dtype = torch.long, device = args.device), torch.full(size = (len(case_1_type_2_scores),), fill_value = 2, dtype = torch.long, device = args.device), torch.full(size = (len(case_1_type_3_scores),), fill_value = 3, dtype = torch.long, device = args.device)), dim = 0)
case_2_labels = torch.cat((torch.full(size = (len(case_2_type_0_scores),), fill_value = 0, dtype = torch.long, device = args.device), torch.full(size = (len(case_2_type_1_scores),), fill_value = 1, dtype = torch.long, device = args.device), torch.full(size = (len(case_2_type_2_scores),), fill_value = 2, dtype = torch.long, device = args.device)), dim = 0)
case_3_labels = torch.cat((torch.full(size = (len(case_3_type_0_scores),), fill_value = 0, dtype = torch.long, device = args.device), torch.full(size = (len(case_3_type_3_scores),), fill_value = 1, dtype = torch.long, device = args.device)), dim = 0)

# Fit logistic regressions
case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(args.device)
case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(args.device)
case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(args.device)

criterion = torch.nn.CrossEntropyLoss()
case_1_optimizer = torch.optim.SGD(case_1_logistic_regression.parameters(), lr = args.lr, momentum = 0.9)
case_2_optimizer = torch.optim.SGD(case_2_logistic_regression.parameters(), lr = args.lr, momentum = 0.9)
case_3_optimizer = torch.optim.SGD(case_3_logistic_regression.parameters(), lr = args.lr, momentum = 0.9)

progress = tqdm(total = args.epochs)
for epoch in range(args.epochs):
    case_1_optimizer.zero_grad()
    case_1_logits = case_1_logistic_regression(case_1_scores)
    case_1_loss = criterion(case_1_logits, case_1_labels)
    case_1_loss.backward()
    case_1_optimizer.step()
    
    case_2_optimizer.zero_grad()
    case_2_logits = case_2_logistic_regression(case_2_scores)
    case_2_loss = criterion(case_2_logits, case_2_labels)
    case_2_loss.backward()
    case_2_optimizer.step()

    case_3_optimizer.zero_grad()
    case_3_logits = case_3_logistic_regression(case_3_scores)
    case_3_loss = criterion(case_3_logits, case_3_labels)
    case_3_loss.backward()
    case_3_optimizer.step()

    progress.set_description(f'Case 1 loss: {case_1_loss.detach().cpu().item()} | Case 2 loss: {case_2_loss.detach().cpu().item()} | Case 3 loss: {case_3_loss.detach().cpu().item()}')
    progress.update()

progress.close()

def _state_dict(module):
    return {k: v.cpu() for k, v in module.state_dict().items()}

case_1_state_dict = _state_dict(case_1_logistic_regression)
case_2_state_dict = _state_dict(case_2_logistic_regression)
case_3_state_dict = _state_dict(case_3_logistic_regression)

state_dict = {}
state_dict['case_1_logistic_regression'] = case_1_state_dict
state_dict['case_2_logistic_regression'] = case_2_state_dict
state_dict['case_3_logistic_regression'] = case_3_state_dict

torch.save(state_dict, args.calibration_logistic_regressions_save_file)
