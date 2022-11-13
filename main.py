import torch
from session.bbn_session import BBNSession
from session.api_stubs import APIStubs
from session.osu_interface import OSUInterface
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--data-root', default='.')
    p.add_argument('--scg-ensemble', default='./ensemble/pretrained')
    p.add_argument('--pretrained-unsupervised-novelty-path', default='./unsupervisednoveltydetection/unsupervised_novelty_detection_module_2.pth')
    p.add_argument('--pretrained-backbone-path', default='./unsupervisednoveltydetection/backbone_2.pth')
    p.add_argument('--api-stubs', action='store_true')
    p.add_argument('--log', action='store_true')
    p.add_argument('--log-dir', default='./logs')
    p.add_argument('--ignore-verb-novelty', default=False, action='store_true')
    p.add_argument('--detection-feedback', action='store_true')
    p.add_argument('--given-detection', default=False, action='store_true')
    p.add_argument('--train-csv-path', default='./dataset_v4/dataset_v4_2_train.csv')
    p.add_argument('--val-csv-path', default='./dataset_v4/dataset_v4_2_val.csv')
    p.add_argument('--val-incident-csv-path', default='./dataset_v4/dataset_v4_2_cal_incident.csv')
    p.add_argument('--val-corruption-csv-path', default='./dataset_v4/dataset_v4_2_cal_corruption.csv')
    p.add_argument('--trial-size', type=int, default=600)
    p.add_argument('--trial-batch-size', type=int, default=10)
    p.add_argument('--retraining-batch-size', type=int, default=80)
    p.add_argument('--disable-retraining', default=False, action='store_true')
    p.add_argument('--api-dir', default='./session/api')
    p.add_argument('--tests-dir', default='./session/tests')
    p.add_argument('--url', default='http://localhost:8000')
    p.add_argument('--class_count', type=int, default=29)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--domain', default='svo_classification')
    p.add_argument('--classification_feedback', action="store_true")
    p.add_argument('--detector_seed', type=int, default=1234)
    p.add_argument('--version', default='101')
    p.add_argument('--sys_results_dir', default='./session/temp/sys_results_ClinetServerTestHintA')
    p.add_argument('--test_ids', nargs="+", default=None)
    p.add_argument('--hintsflag', default= True)
    
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
        pretrained_unsupervised_novelty_path=args.pretrained_unsupervised_novelty_path,
        pretrained_backbone_path=args.pretrained_backbone_path,
        feedback_enabled=args.detection_feedback,
        given_detection=args.given_detection,
        log=args.log,
        log_dir=args.log_dir,
        ignore_verb_novelty=args.ignore_verb_novelty, 
        train_csv_path=args.train_csv_path,
        val_csv_path=args.val_csv_path,
        val_incident_csv_path=args.val_incident_csv_path,
        val_corruption_csv_path=args.val_corruption_csv_path,
        trial_size=args.trial_size,
        trial_batch_size=args.trial_batch_size,
        retraining_batch_size=args.retraining_batch_size,
        disable_retraining=args.disable_retraining)
    
    api = APIStubs(args.api_dir, args.tests_dir) if args.api_stubs else None
    
    test_session = BBNSession('OND', args.domain, args.class_count, 
        args.classification_feedback, args.detection_feedback,
        args.given_detection, args.data_root,
        args.sys_results_dir, args.url, args.batch_size,
        args.version, detection_threshold,
        api, osu_int, args.hintsflag)
        
    test_session.run(args.detector_seed, args.test_ids)
