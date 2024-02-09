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
    p.add_argument('--data-root', default='/nfs/hpc/share/sail_on3/')
    p.add_argument('--scg-ensemble', default='./ensemble/pretrained')
    p.add_argument('--pretrained-models-dir', default='./pretrained-models')
    p.add_argument('--backbone-architecture', type=Backbone.Architecture, choices=list(Backbone.Architecture), default=Backbone.Architecture.swin_t)
    p.add_argument('--should-log', action='store_true', dest='log')
    p.add_argument('--log-dir', default='./session/temp/logsDete10')
    p.add_argument('--ignore-verb-novelty', default=False, action='store_true')
    p.add_argument('--detection-feedback', action='store_true')
    p.add_argument('--given-detection', default=False, action='store_true')
    p.add_argument('--train-csv-path', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/train.csv')
    p.add_argument('--val-csv-path', default='/nfs/hpc/share/sail_on3/final/osu_train_cal_val/calib.csv')
    p.add_argument('--trial-size', type=int, default=200)
    p.add_argument('--trial-batch-size', type=int, default=10)
    p.add_argument('--disable-retraining', default=False, action='store_true')
    p.add_argument('--url', default='http://127.0.0.1:8002')
    p.add_argument('--class_count', type=int, default=29)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--domain', default='image_classification')
    p.add_argument('--detector_seed', type=int, default=1234)
    p.add_argument('--version', default='101')
    p.add_argument('--sys_results_dir', default='./GAN_Tests/resutls/SS_GAN_Augment')
    p.add_argument('--test_ids', nargs="+", default=None)
    p.add_argument('--hintA', default=False)
    p.add_argument('--hintB', default=False)
    p.add_argument('--root-cache-dir', type=str, default='/nfs/hpc/share/sail_on3/.data-cache')
    p.add_argument('--n-known-val', type=int, default=4068)
    p.add_argument('--classifier-trainer', type=TopLevelApp.ClassifierTrainer, choices=list(TopLevelApp.ClassifierTrainer), default=TopLevelApp.ClassifierTrainer.end_to_end)
    p.add_argument('--precomputed-feature-dir', type=str, default='./.features/resizepad=224/none/normalized')
    p.add_argument('--retraining-augmentation', type=Augmentation, choices=list(Augmentation), default=Augmentation.none)
    p.add_argument('--retraining-lr', type=float, default=0.005)
    p.add_argument('--retraining-batch-size', type=int, default=32)
    p.add_argument('--retraining-val-interval', type=int, default=1)
    p.add_argument('--retraining-patience', type=int, default=50)
    p.add_argument('--retraining-min-epochs', type=int, default=1)
    p.add_argument('--retraining-max-epochs', type=int, default=1000)
    p.add_argument('--retraining-label-smoothing', type=float, default=0.0)
    p.add_argument('--retraining-scheduler-type', type=SchedulerType, choices=list(SchedulerType), default=SchedulerType.none)
    p.add_argument('--feedback-sampling-configuration', type=(lambda s : FeedbackSamplingConfigurationOption[s]), choices=list(FeedbackSamplingConfigurationOption), default=FeedbackSamplingConfigurationOption.none)
    p.add_argument('--feedback-loss-weight', type=float, default=0.5)
    p.add_argument('--detection-threshold', type=float, default=0.5)
    p.add_argument('--retraining-loss-fn', type=LossFnEnum, choices=list(LossFnEnum), default=LossFnEnum.cross_entropy)
    p.add_argument('--class-frequency-file', type=str, default=None)
    p.add_argument('--gan_augment', type=bool, default=False)
    p.add_argument('--distributed', action='store_true')
    p.add_argument('--device', type=str, default='cuda:0')

    args = p.parse_args()
    # torch.autograd.set_detect_anomaly(True)
    # assert ((not args.distributed) or (args.classifier_trainer == TopLevelApp.ClassifierTrainer.end_to_end)),\
    #     'Only end-to-end training is supported in distributed mode'
    assert ((not args.distributed) or 
        (args.classifier_trainer in [TopLevelApp.ClassifierTrainer.end_to_end, TopLevelApp.ClassifierTrainer.ewc_train])),\
        'Only end-to-end and ewc_train training are supported in distributed mode'

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
        class_frequency_file=args.class_frequency_file,
        gan_augment=args.gan_augment,
        device=device,
        retrain_fn=retrain_fn,
        val_reduce_fn=val_reduce_fn,
        model_unwrap_fn=model_unwrap_fn,
        feedback_sampling_configuration=args.feedback_sampling_configuration
    )

    test_session = BBNSession('OND', args.domain, args.class_count, 
        args.detection_feedback,
        args.given_detection, args.data_root,
        args.sys_results_dir, args.url, args.batch_size,
        args.version, detection_threshold,
        None, osu_int, args.hintA, args.hintB)

    test_session.run(args.detector_seed, args.test_ids)

    # If distributed, tell the slave processes to skip their next call (which
    # signifies that they should terminate)
    if args.distributed:
        distributedutils.broadcast_skip(src=0, device=device)
