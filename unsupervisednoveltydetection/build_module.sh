#!/bin/bash

python unsupervisednoveltydetection/train_classifier.py --device "cuda:0" --classifier-save-file unsupervisednoveltydetection/classifier.pth
python unsupervisednoveltydetection/train_confidence_classifier.py --device "cuda:0" --classifier-save-file unsupervisednoveltydetection/confidence_classifier.pth
python unsupervisednoveltydetection/train_confidence_calibrator.py --device "cuda:0" --classifier-load-file unsupervisednoveltydetection/confidence_classifier.pth --calibrator-save-file unsupervisednoveltydetection/confidence_calibrator.pth
python unsupervisednoveltydetection/compute_known_combinations.py --device "cuda:0" --known-combinations-save-file unsupervisednoveltydetection/known_combinations.pth
python unsupervisednoveltydetection/generate_anomaly_score_contexts.py --device "cuda:0" --classifier-load-file unsupervisednoveltydetection/classifier.pth --score-contexts-save-file unsupervisednoveltydetection/score_contexts.pth
python unsupervisednoveltydetection/compile_module.py --classifier-load-file unsupervisednoveltydetection/classifier.pth --confidence-calibrator-load-file unsupervisednoveltydetection/confidence_calibrator.pth --score-contexts-load-file unsupervisednoveltydetection/score_contexts.pth --known-combinations-load-file unsupervisednoveltydetection/known_combinations.pth --module-save-file unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth
