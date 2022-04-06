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
    p.add_argument('--pretrained-unsupervised-novelty-path', default='./unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    p.add_argument('--api-stubs', action='store_true')
    p.add_argument('--log', action='store_true')
    p.add_argument('--log-dir', default='./logs')
    p.add_argument('--ignore-verb-novelty', default=False, action='store_true')
    p.add_argument('--detection-feedback', action='store_true')
    p.add_argument('--url', default='http://localhost:6789')
    p.add_argument('--class_count', type=int, default=29)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--domain', default='svo_classification')
    p.add_argument('--classification_feedback', action="store_true")
    p.add_argument('--given_detection', action='store_true')
    p.add_argument('--detector_seed', type=int, default=1234)
    p.add_argument('--version', default='101')
    p.add_argument('--sys_results_dir', default='./session/temp/sys_results')
    p.add_argument('--api-dir', default='./session/api')
    p.add_argument('--tests-dir', default='./session/tests')
    p.add_argument('--test_ids', nargs="+", default=None)
    
    args = p.parse_args()

    if args.given_detection:
        print("Given Detection is not currently supported. Exiting...")
    else:
        torch.backends.cudnn.benchmark = False
    
        if args.log:
            p = Path(args.log_dir)
            if not p.exists():
                p.mkdir()
            
        osu_int = OSUInterface(scg_ensemble=args.scg_ensemble, 
            data_root=args.data_root, 
            pretrained_unsupervised_novelty_path=args.pretrained_unsupervised_novelty_path, 
            feedback_enabled=args.detection_feedback,
            given_detection=False,
            log=args.log,
            log_dir=args.log_dir,
            ignore_verb_novelty=args.ignore_verb_novelty)
        
        api = APIStubs(args.api_dir, args.tests_dir) if args.api_stubs else None
        
        test_session = BBNSession('OND', args.domain, args.class_count, 
            args.classification_feedback, args.detection_feedback,
            False, args.data_root,
            args.sys_results_dir, args.url, args.batch_size,
            args.version, 0.5,
            api, osu_int)
            
        test_session.run(args.detector_seed, args.test_ids)
