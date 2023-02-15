import torch
from session.bbn_session import BBNSession
from session.osu_interface import OSUInterface
from argparse import ArgumentParser
from pathlib import Path

from backbone import Backbone
from tupleprediction.training import Augmentation, SchedulerType

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--data-root', default='/nfs/hpc/share/sail_on3/')
    p.add_argument('--scg-ensemble', default='./ensemble/pretrained')
    p.add_argument('--pretrained-models-dir', default='/nfs/hpc/share/guyera/projects/ss-osu-system/pretrained-models')
    p.add_argument('--backbone-architecture', type=Backbone.Architecture, choices=list(Backbone.Architecture), default=Backbone.Architecture.swin_t)
    p.add_argument('--log', action='store_true')
    p.add_argument('--log-dir', default='/nfs/hpc/share/guyera/projects/ss-osu-system/logsDete10')
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
    p.add_argument('--sys_results_dir', default='./session/temp/SVO_10_test_trials_csv60_with_detection_feedback')
    p.add_argument('--test_ids', nargs="+", default=None)
    p.add_argument('--hintA', default=False)
    p.add_argument('--hintB', default=False)
    p.add_argument('--root-cache-dir', type=str, default='/nfs/hpc/share/sail_on3/.data-cache')
    p.add_argument('--n-known-val', type=int, default=4068)
    p.add_argument('--precomputed-feature-dir', type=str, default='/nfs/hpc/share/guyera/projects/ss-osu-system/.features/resizepad=224/none/normalized')
    p.add_argument('--retraining-augmentation', type=Augmentation, choices=list(Augmentation), default=Augmentation.none)
    p.add_argument('--retraining-lr', type=float, default=0.005)
    p.add_argument('--retraining-batch-size', type=int, default=32)
    p.add_argument('--retraining-patience', type=int, default=15)
    p.add_argument('--retraining-min-epochs', type=int, default=1)
    p.add_argument('--retraining-max-epochs', type=int, default=1000)
    p.add_argument('--retraining-label-smoothing', type=float, default=0.0)
    p.add_argument('--retraining-scheduler-type', type=SchedulerType, choices=list(SchedulerType), default=SchedulerType.none)
    p.add_argument('--feedback-loss-weight', type=float, default=0.5)

    args = p.parse_args()

    torch.backends.cudnn.benchmark = False

    detection_threshold = 0.5

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
        precomputed_feature_dir=args.precomputed_feature_dir,
        retraining_augmentation=args.retraining_augmentation,
        retraining_lr=args.retraining_lr,
        retraining_batch_size=args.retraining_batch_size,
        retraining_patience=args.retraining_patience,
        retraining_min_epochs=args.retraining_min_epochs,
        retraining_max_epochs=args.retraining_max_epochs,
        retraining_label_smoothing=args.retraining_label_smoothing,
        retraining_scheduler_type=args.retraining_scheduler_type,
        feedback_loss_weight=args.feedback_loss_weight
    )
    
    test_session = BBNSession('OND', args.domain, args.class_count, 
        args.detection_feedback,
        args.given_detection, args.data_root,
        args.sys_results_dir, args.url, args.batch_size,
        args.version, detection_threshold,
        None, osu_int, args.hintA, args.hintB)
        
    test_session.run(args.detector_seed, args.test_ids)
