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

## Example to Run the DCA System

Before running the DCA system, ensure the API server is installed, up and running using the [SS API repository](https://github.com/guyera/ss-api). Follow these steps:

## Running Server
```bash
cd ~/ss-api/
sail_on_server_ss --url 127.0.0.1:8005 \
--data-directory '/test_trials/api_tests/' \
--bboxes-json-file 'test.json' \
--results-directory 'Experiments/Exp_1_EWC'
```

## Running Client 

```bash
cd ~/ss-osu-system
torchrun \
   --nnodes=1 \
   --nproc_per_node=4 \
   --rdzv_id=103 \
   --rdzv_endpoint=localhost:28319 \
   main.py \
   --detection-feedback \
   --url 'http://127.0.0.1:8005' \
   --trial-size 3000 \
   --trial-batch-size 10 \
   --test_ids OND.102.000 OND.103.000 OND.104.000 OND.105.000 OND.1061.000 OND.1062.000 OND.1063.000 OND.1064.000 \
   --root-cache-dir .data-cache \
   --train-csv-path /nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.csv \
   --pretrained-models-dir 'pretrained-models-balanced-normalization-corrected/train-heads/hack' \
   --precomputed-feature-dir '.features/hack/resizedpad=224/none/normalized' \
   --classifier-trainer ewc-train \
   --retraining-lr 1e-5 \
   --retraining-batch-size 64 \
   --retraining-max-epochs 50 \
   --gan_augment False \
   --distributed \
   --feedback-sampling-configuration combined \
   --feedback-loss-weight 0.5 \
   --should-log \
   --log-dir 'Experiments/Exp_1_EWC/logs'
```

## Evaluation 
The following command line arguments are required for evaluation:
- `--test_root`: Directory containing the test ground truth files.
- `--sys_output_root`: Directory where the DCA system's output is stored, the parent directory of 'OND'.
- `--log_dir`: Directory where the scoring reults will be written.
- `--detection_threshold`: Threshold for detection.
- `--activity_presence_threshold`: Threshold for activity presence.
- `--species_presence_threshold`: Threshold for species presence.
- `--box_pred_dir`: Directory containing the DCS system's prediction for each box, including predicted species and activity.


```bash
cd ~/ss-osu-system
python ./umd_test_score/score_known_vs_novel.py \
--test_root '/test_trials/api_tests/OND/image_classification/' \
--sys_output_root 'Experiments/Exp_1_EWC/OND/image_classification/' \
--log_dir 'Experiments/Exp_1_EWC/Results' \
--detection_threshold 0.6 \
--box_pred_dir 'Experiments/Exp_1_EWC/logs' \
--species_presence_threshold 0.4 \
--activity_presence_threshold 0.4

```

## Creating and benchmarking a new system
To create a new system to benchmark against the SS trials, simply create your own class that replicates the interface of the OSU Interface module (found in `session/osu_interface.py`), pass an instance of your class to the `BBNSession` constructor as in `main.py`, and call the `run()` method on the `BBNSession` object.

## License
This repo is a product of collaboration between multiple entities, and different portions of the source code are licensed under different terms. For example, the code written by OSU is licensed under GPLv3, whereas the code written by BBN and UMD are subject to custom licenses explained in copyright notices written in comments in the respective source code. Please familiarize yourself with `DISCLAIMER`, `LICENSE`, and the copyright notices documented in the source code.
