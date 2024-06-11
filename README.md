This package is the DCA system described in the paper, "NOVEL-SS: A Dataset for Integrated Novelty-Aware Computer Vision Systems". This repo only contains the DCA system, the system interface, and the client code that connects to the SS API server to run a trial.

## Running the DCA System
This section explains how to run the DCA system on one or more trials.

The following entrypoint scripts are important when running the DCA system:

- `train_novelty_detection_module.py`: Used to pretrain the DCA system on the pretraining (non-novel and known-novel) data. This includes training the backbone and classification heads for species and activity bounding box classification, classifier calibrators (temperature scalers), the activation statistical model (i.e., PCA and KDE over early layer model activations) for computing novel environment scores, and the multinomial logistic regression for novelty category prediction.
- `precompute_backbone_features.py`: Used to precompute features from the pretrained backbone on the pretraining and validation data to speed up trials that retrain only the logit layer during accommodation.
- `main.py`: Used to run the DCA system on one or more trials. The API server must be started and hosted at the address specified via the command line arguments. The API server is not included in this package.

### `train_novelty_detection_module.py`
`train_novelty_detection_module.py` is used to pretrain the DCA system. It's configured to run via torchrun. The following command was used to pretrain the DCA system for all results reported in the paper:

```bash
torchrun \
	--nnodes=1 \
	--nproc_per_node=1 \
	--rdzv_id=100 \
	--rdzv_backend=c10d \
	--rdzv_endpoint=localhost:28317 \
	train_novelty_detection_module.py \
	--label-smoothing 0.00 \
	--lr 0.005 \
	--augmentation rand_augment \
	--batch-size 128 \
	--n-known-val 4068 \
	--scheduler-type cosine \
	--max-epochs 200 \
	--root-log-dir "$root_log_dir" \
	--root-checkpoint-dir "$root_checkpoint_dir" \
	--data-root /nfs/hpc/share/sail_on3 \
	--root-cache-dir data-cache \
	--train-csv-path sail_on3/final/osu_train_cal_val/train.csv \
	--cal-csv-path sail_on3/final/osu_train_cal_val/calib.csv \
	--classifier-trainer end-to-end \
	--loss-fn cross-entropy \
	--balance-class-frequencies \
	--save-dir pretrained-models \
	--no-memory-cache
```

The above command-line arguments are explained in the source code. Replace `--data-root`, `--train-csv-path`, and `--cal-csv-path` with the appropriate paths for your environment. Note that the first epoch will be slower than the rest as it will be loading the raw high-resolution SS data, passing it through various transformations (including rescaling and cropping), and caching it to the specified `--root-cache-dir`. Once a full pass has been made through the data, the transformed data will all be cached in `--root-cache-dir`, which significantly speeds up subsequent passes (including in other entrypoints, such as `main.py`).

torchrun's `--nproc_per_node` command line argument can be increased to speed up pretraining with multi-GPU parallelization. Each process will use a separate GPU on the current node. This script has not been tested on multi-node parallelization, so increasing `--nnodes` above 1 may not work as expected.

### `precompute_backbone_features.py`
`precompute_backbone_features.py` can be used to precompute features from the pretrained DCA system backbone on all SS training data. This is useful for running trials that retrain only the DCA system's logit layer during accommodation, leaving the backbone frozen. If you don't intend to run such trials, then you can skip this step. The following command can be used to precompute backbone features for logit-layer-accommodation experiments using a backbone pretrained as described in the previous section:

```bash
python -u precompute_backbone_features.py \
	--n-known-val 4068 \
	--batch-size 64 \
	--data-root sail_on3 \
	--root-cache-dir data-cache \
	--train-csv-path sail_on3/final/osu_train_cal_val/train.csv \
	--cal-csv-path sail_on3/final/osu_train_cal_val/calib.csv \
	--pretrained-backbone-path pretrained-models/swin_t/backbone.pth \
	--root-save-dir precomputed-features
```

The above command-line arguments are explained in the source code. Replace `--data-root`, `--train-csv-path`, and `--cal-csv-path` with the appropriate paths for your environment. Make sure to use the same `--root-cache-dir` as was used in the pretraining step in the previous section.

### `main.py`
After training the DCA system (and optionally precomputing backbone features to speed up DCA trials involving retraining of just the logit layer during accommodation), `main.py` can be used to run the DCA system on one or more trials. The API server must be started and accessible by the DCA system at the address to be specified by the `--url` command line argument in `main.py`.

This system-side codebase is organized into three main components: 1) the BBN Session, which communicates with the SS API server to request trial data and post predictions, 2) the DCA System, whose public-facing interface is primarily provided by the `toplevel/` package, and 3) the OSU Interface, which serves as a communication layer for trial data and predictions between the BBN Session and the DCA System. There are various command-line arguments in `main.py` that control various behaviors of these three components. They're explained in the source code.

Before running the DCA system via `main.py`, ensure that the [SS API server](https://github.com/guyera/ss-api) is installed and running.

The following is an example command that can be used to run the DCA system on trials OND.102.000 and OND.103.000 with EWC accommodation:

```bash
torchrun \
   --nnodes=1 \
   --nproc_per_node=1 \
   --rdzv_id=103 \
   --rdzv_endpoint=localhost:28319 \
   main.py \
   --detection-feedback \
   --url 'http://127.0.0.1:8005' \
   --trial-size 3000 \
   --trial-batch-size 10 \
   --test_ids OND.102.000 OND.103.000 \
   --root-cache-dir data-cache \
   --train-csv-path sail_on3/final/osu_train_cal_val/train.csv \
   --pretrained-models-dir pretrained-models \
   --precomputed-feature-dir 'precomputed-features/resizedpad=224/none/normalized' \
   --classifier-trainer ewc-train \
   --retraining-lr 1e-5 \
   --retraining-batch-size 64 \
   --retraining-max-epochs 50 \
   --gan_augment False \
   --distributed \
   --feedback-loss-weight 0.5 \
   --should-log \
   --log-dir 'logs'
```

## Evaluation 
After running the DCA system on one or more trials, its predictions from the trial will be stored in the results directory specified when starting the SS API server. They can then be scored for evaluation.

The following command line arguments are required for evaluation:
- `--test_root`: Directory containing the test ground truth files.
- `--sys_output_root`: Directory where the DCA system's output is stored; the child directory of 'OND' within the results directory supplied to the SS API server at startup.
- `--log_dir`: Directory where the scoring results will be written.
- `--detection_threshold`: Threshold for detection.
- `--activity_presence_threshold`: Threshold for activity presence.
- `--species_presence_threshold`: Threshold for species presence.
- `--box_pred_dir`: Directory containing the DCA system's prediction for each box, including predicted species and activity. Used to compute confusion matrices. The DCA system stores these predictions in its log directory (i.e., the directory specified to `--log-dir` when running `main.py`).

The following command was used to generate results for the trials presented in the paper:

```bash
python umd_test_score/score_known_vs_novel.py \
--test_root '/test_trials/api_tests/OND/image_classification/' \
--sys_output_root 'Experiments/Exp_1_EWC/OND/image_classification/' \
--log_dir 'Experiments/Exp_1_EWC/Results' \
--detection_threshold 0.6 \
--box_pred_dir 'logs' \
--species_presence_threshold 0.4 \
--activity_presence_threshold 0.4
```

## Creating and benchmarking a new system
To benchmark a new system against the SS trials, you'll need to create your own system interface layer that communicates between the `BBNSession` and your system, just as the `OSUInterface` class communicates between the `BBNSession` and the DCA system. To do this, create your own class that inherits from the system interface abstract base class, `session/ss_novel_system_interface:SSNovelSystemInterface`, and implement its abstract methods. Then, construct and pass an instance of your system interface to the `BBNSession` constructor, just as in `main.py`, and call the `run()` method on the `BBNSession` object.

## License
This repo is a product of collaboration between multiple entities, and different portions of the source code are licensed under different terms. For example, the code written by OSU is licensed under GPLv3, whereas the code written by BBN and UMD are subject to custom licenses explained in copyright notices written in comments in the respective source code. Please familiarize yourself with `DISCLAIMER`, `LICENSE`, and the copyright notices documented in the source code.
