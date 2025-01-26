# This software was created by Oregon State University under Army Research
# Office (ARO) Award Number W911NF-22-2-0149. ARO, as the Federal awarding
# agency, reserves a royalty-free, nonexclusive and irrevocable right to
# reproduce, publish, or otherwise use this software for Federal purposes, and
# to authorize others to do so in accordance with 2 CFR 200.315(b).

import sys
import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from session.bbn_session import BBNSession
from session.osu_interface import OSUInterface
from argparse import ArgumentParser
from pathlib import Path

from backbone import Backbone
from tupleprediction.training import Augmentation, SchedulerType, LossFnEnum, DistributedRandomBoxImageBatchSampler, FeedbackSamplingConfigurationOption
from toplevel import TopLevelApp, gen_retrain_fn
import distributedutils

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--data-root', default='/nfs/hpc/share/sail_on3/', help='Root data directory.')
    p.add_argument('--scg-ensemble', default='./ensemble/pretrained', help='(Obsolete) Specified directory containing an SCG ensemble; no longer used.')
    p.add_argument('--pretrained-models-dir', default='./pretrained-models', help='Path to directory containing pretrained DCA system models.')
    p.add_argument('--backbone-architecture', type=Backbone.Architecture, choices=list(Backbone.Architecture), default=Backbone.Architecture.swin_t, help='Architecture for DCA system backbone.')
    p.add_argument('--should-log', action='store_true', dest='log', help='Specify this flag to log system outputs.')
    p.add_argument('--log-dir', default='./session/temp/logsDete10', help='Path to log directory for system outputs.')
    p.add_argument('--ignore-verb-novelty', default=False, action='store_true', help='(Obsolete) Configures the system to ignore activity novelty; no longer used.')
    p.add_argument('--detection-feedback', action='store_true', help='Enables feedback querying in the DCA system.')
    p.add_argument('--given-detection', default=False, action='store_true', help='Enables given detection mode, wherein the system is told exactly when novelty is introduced into the trial; used to run oracle studies.')
    p.add_argument('--train-csv-path', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.csv', help='Path to SS-Novel train CSV file.')
    p.add_argument('--val-csv-path', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/calib.csv', help='Path to SS-Novel calibration CSV file.')
    p.add_argument('--trial-size', type=int, default=200, help='The number of examples in a trial.')
    p.add_argument('--trial-batch-size', type=int, default=10, help='The number of examples in a batch / round within a trial.')
    p.add_argument('--disable-retraining', default=False, action='store_true', help='If specified, disables accommodation in the DCA system.')
    p.add_argument('--url', default='http://127.0.0.1:8002', help='URL at which the SS API server is located and accessible.')
    p.add_argument('--class_count', type=int, default=29, help='(Obsolete) Number of subject classes; no longer used.')
    p.add_argument('--batch_size', type=int, default=4, help='(Obsolete) Number of images in a batch; no longer used; replaced with trial-batch-size.')
    p.add_argument('--domain', default='image_classification', help='Specifies the task domain. This no longer needs to be configurable; should always be set to "image_classification".')
    p.add_argument('--detector_seed', type=int, default=1234)
    p.add_argument('--version', default='101', help='Novelty detector version; passed along to SS API server when starting a session.')
    p.add_argument('--sys_results_dir', default='./GAN_Tests/resutls/SS_GAN_Augment', help='Directory for GAN Augmentation experiment results; only used when --gan_augment is set to True.')
    p.add_argument('--test_ids', nargs="+", default=None, help='IDs of test trials to run.')
    p.add_argument('--hintA', default=False, help='Enables hint type A. This is a hint supplied to the client that indicates what kind of novelty appears in the given test trial. See session/ss_novel_system_interface.py documentation for more info.')
    p.add_argument('--hintB', default=False, help='Enables hint type B. This is a hint supplied to the client that indicates whether each image contains novelty. See session/ss_novel_system_interface.py documentation for more info.')
    p.add_argument('--root-cache-dir', type=str, default='/nfs/hpc/share/sail_on3/.data-cache', help='Root cache directory for transformed data.')
    p.add_argument('--n-known-val', type=int, default=4068, help='Number of training images to hold out for model validation during accommodation.')
    p.add_argument('--classifier-trainer', type=TopLevelApp.ClassifierTrainer, choices=list(TopLevelApp.ClassifierTrainer), default=TopLevelApp.ClassifierTrainer.end_to_end, help='Specifies what classifier training method to use during accommodation.')
    p.add_argument('--precomputed-feature-dir', type=str, default='./.features/resizepad=224/none/normalized', help='Path to directory containing precomputed features. Used to speed up logit-layer-accommodation experiments. This argument is ignored when --classifier-trainer is not logit-layer.')
    p.add_argument('--retraining-augmentation', type=Augmentation, choices=list(Augmentation), default=Augmentation.none, help='Augmentation strategy during accommodation.')
    p.add_argument('--retraining-lr', type=float, default=0.005, help='Learning rate used during accommodation.')
    p.add_argument('--retraining-batch-size', type=int, default=32, help='Batch size used during accommodation.')
    p.add_argument('--retraining-val-interval', type=int, default=1, help='Specifies how often validation should be performed during accommodation. Setting this argument to 1 performs validation after every epoch; setting it to 2 performs validation after every other epoch; and so on.')
    p.add_argument('--retraining-patience', type=int, default=50, help='Patience for early-stopping during accommodation.')
    p.add_argument('--retraining-min-epochs', type=int, default=1, help='Minimum number of epochs to train for during accommodation (early stopping will not terminate accommodation retraining prior to this many epochs)')
    p.add_argument('--retraining-max-epochs', type=int, default=1000, help='Maximum number of epochs to train for during accommodation.')
    p.add_argument('--retraining-label-smoothing', type=float, default=0.0, help='Label smoothing used during accommodation.')
    p.add_argument('--retraining-scheduler-type', type=SchedulerType, choices=list(SchedulerType), default=SchedulerType.none, help='Schedule type used during accommodation.')
    p.add_argument('--feedback-sampling-configuration', type=(lambda s : FeedbackSamplingConfigurationOption[s]), choices=list(FeedbackSamplingConfigurationOption), default=FeedbackSamplingConfigurationOption.none, help='Specifies how feedback data should be resampled / mixed into non-feedback data during accommodation. Default is `none`, which keeps feedback and non-feedback data separate, and separate batches are sampled from each in an alternating fashion. `combined` combines the feedback data with the non-feedback data. EWC training requires the sampling configuration to be `none`.')
    p.add_argument('--feedback-loss-weight', type=float, default=0.5, help='Loss weight for feedback data used during accommodation.')
    p.add_argument('--detection-threshold', type=float, default=0.5, help='Detection threshold for novelty. No longer needs to be configured; should always be 0.5.')
    p.add_argument('--retraining-loss-fn', type=LossFnEnum, choices=list(LossFnEnum), default=LossFnEnum.cross_entropy, help='Loss function used during accommodation.')
    p.add_argument('--balance-class-frequencies', action='store_true', help='Enables class-balancing by reweighting per-image loss gradients inversely to class frequency.')
    p.add_argument('--gan_augment', type=bool, default=False, help='Specifies whether GAN Augmentation should be enabled.')
    p.add_argument('--distributed', action='store_true', help='Sets the system to run in distributed mode. Should be specified if and only if running via `torchrun`.')
    p.add_argument('--device', type=str, default='cuda:0', help='Torch device to use when not running in distributed mode (in distributed mode, each process uses its local rank as a CUDA device ordinal).')
    p.add_argument('--oracle-training', type=bool, default=False, help='Specifies whether oracle training should be enabled.')
    p.add_argument('--ewc-lambda', type=float, default=1000, help='EWC lambda hyperparameter for EWC accommodation.')
    p.add_argument('--p-ni-query-threshold', type=float, default=0.0, help='P(N_i) threshold at which to query for feedback on an image')
    p.add_argument('--feedback-budget-override', type=float, default=None, help='Value that client should use for feedback budget, ignoring budget set by the API. Must be smaller than the budget set by the API, but can be fractional (e.g., 2.5 would alternate between querying 2 and 3 images each round)')

    args = p.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    # assert ((not args.distributed) or (args.classifier_trainer == TopLevelApp.ClassifierTrainer.end_to_end)),\
    #     'Only end-to-end training is supported in distributed mode'
    assert ((not args.distributed) or 
        (args.classifier_trainer in [TopLevelApp.ClassifierTrainer.end_to_end, \
        TopLevelApp.ClassifierTrainer.logit_layer, \
        TopLevelApp.ClassifierTrainer.ewc_logit_layer_train, \
        TopLevelApp.ClassifierTrainer.ewc_train])),\
        'Only end-to-end, logit_layer, ewc_logit_layer_train, and ewc_train training are supported in distributed mode'

    from datetime import timedelta
    DEFAULT_TIMEOUT = timedelta(seconds=1000000)
    # torch.set_default_tensor_type(torch.DoubleTensor)

    if args.distributed:

        dist.init_process_group('nccl', timeout = DEFAULT_TIMEOUT)
        rank = dist.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

        world_size = dist.get_world_size()
        device_id = local_rank
        # device = f'cuda:{device_id}'
        device = torch.device(f'cuda:{device_id}')


        torch.cuda.set_device(local_rank)

        def _model_unwrap_fn(backbone, classifier):
            classifier.un_ddp
            return backbone.module, classifier
        model_unwrap_fn = _model_unwrap_fn

        def train_sampler_fn(train_dataset):
            return DistributedSampler(
                train_dataset,
                num_replicas=world_size,
                rank=rank
            )

        def val_reduce_fn(count_tensor):
            return dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)

        def feedback_batch_sampler_fn(box_counts):
            return DistributedRandomBoxImageBatchSampler(
                box_counts,
                args.retraining_batch_size,
                world_size,
                rank
            )

        allow_write = rank == 0
        allow_print = local_rank == 0

        if rank != 0:
            terminate = False
            # Prepare the retraining function that will be called with received
            # (broadcasted-by-rank-0) arguments
            retrain_fn = gen_retrain_fn(
                device_id,
                train_sampler_fn,
                feedback_batch_sampler_fn,
                allow_write,
                allow_print,
                distributed=True
            )
            # Run the retrain function in a loop until told to terminate
            # (signified by a skip of the function call, in-turn signified by
            # broadcasted Nonetype objects for the function's arguments)
            while not terminate:
                # Run the specified function
                _, run = distributedutils.receive_call(
                    retrain_fn,
                    src=0,
                    device=device
                )

                # If the retraining function was run successfully, then keep
                # looping. Otherwise, terminate
                terminate = not run

            # Terminate the process (do not proceed to init the system---that
            # is reserved for the rank 0 process)
            sys.exit(0)
        else:
            # Generate the argument-broadcasting master retraining function
            # to be used by the system in this (rank 0) master process. Each
            # time this function is called with the retraining arguments, those
            # arguments will be broadcasted to the slave processes so that they
            # can run their respective retraining functions. Each process's
            # retraining function is pre-conditioned on the process-specific
            # retraining arguments (device, samplers, allow_write, allow_print)
            # retrain_fn = distributedutils.gen_broadcast_call(
            #     gen_retrain_fn(
            #         device_id,
            #         train_sampler_fn,
            #         feedback_batch_sampler_fn,
            #         allow_write,
            #         allow_print,
            #         distributed=True
            #     ),
            #     src=0,
            #     device=device
            # )
            if world_size != 1:
                retrain_fn = distributedutils.gen_broadcast_call(
                    gen_retrain_fn(
                        device_id,
                        train_sampler_fn,
                        feedback_batch_sampler_fn,
                        allow_write,
                        allow_print,
                        distributed=True
                    ),
                    src=0,
                    device=device
                )
            else:
                retrain_fn = gen_retrain_fn(
                    device_id,
                    train_sampler_fn,
                    feedback_batch_sampler_fn,
                    allow_write,
                    allow_print,
                    distributed=True
                )
    else:
        device = args.device
        if device == 'cpu':
            device_id = None
        else:
            device_id = device.split(':')[1]
        train_sampler_fn = None
        val_reduce_fn = None
        feedback_batch_sampler_fn = None
        allow_write = True
        allow_print = True

        retrain_fn = gen_retrain_fn(
            device_id,
            train_sampler_fn,
            feedback_batch_sampler_fn,
            allow_write,
            allow_print,
            distributed=False
        )

        model_unwrap_fn = None

    torch.backends.cudnn.benchmark = False

    detection_threshold = args.detection_threshold

    if args.log:
        p = Path(args.log_dir)
        if not p.exists():
            p.mkdir()


    # import ipdb; ipdb.set_trace()

    osu_int = OSUInterface(scg_ensemble=args.scg_ensemble, 
        data_root=args.data_root, 
        pretrained_models_dir=args.pretrained_models_dir,
        backbone_architecture=args.backbone_architecture,
        feedback_enabled=args.detection_feedback,
        given_detection=args.given_detection,
        log=args.log,
        log_dir=args.log_dir,
        ignore_verb_novelty=args.ignore_verb_novelty, 
        train_csv_path=args.train_csv_path,
        val_csv_path=args.val_csv_path,
        trial_size=args.trial_size,
        trial_batch_size=args.trial_batch_size,
        disable_retraining=args.disable_retraining,
        root_cache_dir=args.root_cache_dir,
        n_known_val=args.n_known_val,
        classifier_trainer=args.classifier_trainer,
        precomputed_feature_dir=args.precomputed_feature_dir,
        retraining_augmentation=args.retraining_augmentation,
        retraining_lr=args.retraining_lr,
        retraining_batch_size=args.retraining_batch_size,
        retraining_val_interval=args.retraining_val_interval,
        retraining_patience=args.retraining_patience,
        retraining_min_epochs=args.retraining_min_epochs,
        retraining_max_epochs=args.retraining_max_epochs,
        retraining_label_smoothing=args.retraining_label_smoothing,
        retraining_scheduler_type=args.retraining_scheduler_type,
        feedback_loss_weight=args.feedback_loss_weight,
        retraining_loss_fn=args.retraining_loss_fn,
        balance_class_frequencies=args.balance_class_frequencies,
        gan_augment=args.gan_augment,
        device=device,
        retrain_fn=retrain_fn,
        val_reduce_fn=val_reduce_fn,
        model_unwrap_fn=model_unwrap_fn,
        feedback_sampling_configuration=args.feedback_sampling_configuration,
        oracle_training = args.oracle_training,
        ewc_lambda = args.ewc_lambda,
        p_ni_query_threshold=args.p_ni_query_threshold
    )

    test_session = BBNSession('OND', args.domain, args.class_count, 
        args.detection_feedback,
        args.given_detection, args.data_root,
        args.sys_results_dir, args.url, args.batch_size,
        args.version, detection_threshold,
        None, osu_int, args.hintA, args.hintB, args.feedback_budget_override)

    test_session.run(args.detector_seed, args.test_ids)

    # If distributed, tell the slave processes to skip their next call (which
    # signifies that they should terminate)
    if args.distributed:
        distributedutils.broadcast_skip(src=0, device=device)
