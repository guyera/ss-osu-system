This package is the DCA system described in the paper, "NOVEL-SS: A Dataset for Integrated Novelty-Aware Computer Vision Systems". This package currently only contains the DCA system. The dataset and API server that serve up trials to this system will be supplied elsewhere.

## Important entry points
- `train_novelty_detection_module.py`: Used to pretrain the DCA system on the pretraining (non-novel and known-novel) data. This includes training the backbone and classification heads for species and activity bounding box classification, classifier calibrators (temperature scalers), activation statistical model (i.e., PCA and KDE over early layer model activations) for computing novel environment scores, and multinomial logistic regression for novelty category prediction.
- `precompute_backbone_features.py`: Used to precompute features from the pretrained backbone onm the pretraining and validation data to speed up trials
- `main.py`: Used to run the DCA system on one or more trials. The API server must be started and hosted at the address specified via the command line arguments. The API server is not included in this package.

### `main.py`
After training the DCA system (and optionally precomputing backbone features to speed up DCA trials involving retraining of just the logit layer), `main.py` can be used to run the DCA system on one or more trials. The API server must be started and accessible by the DCA system at the address to be specified by the `--url` command line argument in `main.py`.

This system-side codebase is organized into three main components: 1) the BBN Session, which communicates with the SS API server to request trial data and post predictions, 2) the DCA System, whose public-facing interface is primarily provided by the `toplevel/` package, and 3) the OSU Interface, which serves as a communication layer for trial data and predictions between the BBN Session and the OSU Interface.

The following command line arguments are important for configuring the behavior of the BBN Session and trials:
- `--data-root`: The root data directory containing the SS data
- `--detection-feedback`: Specifying this flag enables detection feedback in the trials
- `--sys_results_dir`: Specifies where system result files should be stored
- `--url`: Specifies the HTTP URL (including the hostname / IP and port) of the SS API server
- `--hintA`: Specifying this flag enables hint type A during the trial. This hint type provides the system with oracle knowledge of the trial-level novelty type, which allows it to avoid predicting tuples of a novelty type that contradicts the trial's true novelty type. That is, this hint type makes the system an oracle novelty characterizer when presented with images containing novelty. Combining this with `--hintB` makes the system an oracle novelty detector as well, which can be used to run ablation studies that focus on accommodation.
- `--hintB`: Specifying this flag enables hint type B during the trial. This hint type provides the system with oracle 1-bit knowledge stating whether each image is novel or not. That is, providing this hint type makes the system an oracle novelty detector. Combining this with `--hintA` makes the system an oracle novelty characterizer as well, which can be used to run ablation studies that focus on accommodation.

There are many other command line arguments in `main.py` as well. Some of them are obsolete and should be left to their default values, whereas others configure the behavior of the DCA system.

## Creating and benchmarking a new system
To create a new system to benchmark against the SS trials, simply create your own class that replicates the interface of the OSU Interface module (found in `session/osu_interface.py`), pass an instance of your class to the `BBNSession` constructor as in `main.py`, and call the `run()` method on the `BBNSession` object.
