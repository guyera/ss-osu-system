import torch
from session.bbn_session import BBNSession
from session.osu_interface import OSUInterface
from argparse import ArgumentParser

if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('--data-root', default='.')
    p.add_argument('--scg-ensemble', default='./ensemble/pretrained')
    p.add_argument('--pretrained-unsupervised-novelty-path', default='./unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth')
    p.add_argument('--api_stubs', action='store_true')
    p.add_argument('--url', default='http://localhost:6789')
    p.add_argument('--class_count', type=int, default=29)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--domain', default='svo_classification')
    p.add_argument('--classification_feedback', action="store_true")
    p.add_argument('--detection_feedback', action='store_true')
    p.add_argument('--given_detection', action='store_true')
    p.add_argument('--detector_seed', type=int, default=1234)
    p.add_argument('--version', default='101')
    p.add_argument('--sys_results_dir', default='./session/temp/sys_results')
    p.add_argument('--detection_threshold', type=float, default=0.65)
    
    args = p.parse_args()

    torch.backends.cudnn.benchmark = False

    osu_int = OSUInterface(scg_ensemble=args.scg_ensemble, 
        data_root=args.data_root, 
        pretrained_unsupervised_novelty_path=args.pretrained_unsupervised_novelty_path, 
        cusum_thresh=args.detection_threshold,
        feedback_enabled=args.detection_feedback,
        given_detection=args.given_detection)
    
    test_session = BBNSession('OND', args.domain, args.class_count, 
        args.classification_feedback, args.detection_feedback,
        args.given_detection, args.data_root,
        args.sys_results_dir, args.url, args.batch_size,
        args.version, args.detection_threshold,
        args.api_stubs, osu_int)

    test_session.run(args.detector_seed)
