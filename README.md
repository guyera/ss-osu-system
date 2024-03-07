This package is the DCA system described in the paper, "NOVEL-SS: A Dataset for Integrated Novelty-Aware Computer Vision Systems". This package currently only contains the DCA system. The dataset and API server that serve up trials to this system will be supplied elsewhere.

This branch has been modified from the original stable branch for anonymization purposes. It has not been tested post-anonymization, so the system reflected in this branch may not run properly.

## Important entry points
- `train_novelty_detection_module.py`: Used to pretrain the DCA system on the pretraining (non-novel and known-novel) data. This includes training the backbone and classification heads for species and activity bounding box classification, classifier calibrators (temperature scalers), activation statistical model (i.e., PCA and KDE over early layer model activations) for computing novel environment scores, and multinomial logistic regression for novelty category prediction.
- `precompute_backbone_features.py`: Used to precompute features from the pretrained backbone onm the pretraining and validation data to speed up trials
- `main.py`: Used to run the DCA system on one or more trials. The API server must be started and hosted at the address specified via the command line arguments. The API server is not included in this package.

More details and example execution commands will be provided post-publication.
