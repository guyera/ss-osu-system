from torch import nn
from torchvision.models import resnet50

import unsupervisednoveltydetection
import noveltydetection

device = 'cuda:0'

backbone = resnet50(pretrained = True)
backbone.fc = nn.Linear(backbone.fc.weight.shape[1], 256)
backbone = backbone.to(device)

classifier = unsupervisednoveltydetection.common.ClassifierV2(256, 5, 12, 8, 72)
detector = unsupervisednoveltydetection.UnsupervisedNoveltyDetector(classifier, 5, 12, 8)
detector = detector.to(device)

case_1_logistic_regression = noveltydetection.utils.Case1LogisticRegression().to(device)
case_2_logistic_regression = noveltydetection.utils.Case2LogisticRegression().to(device)
case_3_logistic_regression = noveltydetection.utils.Case3LogisticRegression().to(device)

trainer = unsupervisednoveltydetection.training.NoveltyDetectorTrainer()

trainer.train_novelty_detection_module(backbone, detector, case_1_logistic_regression, case_2_logistic_regression, case_3_logistic_regression)
