import os
import torch
from toplevel import TopLevelApp

class Args:
    def __init__(self):
        self.world_size = None
        self.dataset = None
        self.data_root = None
        self.csv_path = None
        self.num_subj_cls = None
        self.num_obj_cls = None
        self.num_action_cls = None
        self.num_workers = None
        self.batch_size = None
        self.max_object = None
        self.max_subject = None
        self.num_iter = None
        self.box_score_thresh = None
        self.num_iter = None
        self.top_k = None
        self.checkpoint_path = None
        self.ensemble_path = None
        self.pretrained_unsupervised_novelty_module_path = None

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    args = Args()

    args.world_size = 1
    args.dataset = 'Custom'
    args.data_root = 'Custom'
    args.csv_path = 'Custom/annotations/val_dataset_v1_val.csv'
    args.ensemble_path = 'pretrained'
    args.num_iter = 2
    args.box_score_thresh = 0.0
    args.num_subj_cls = 6
    args.num_obj_cls = 9
    args.num_action_cls = 8
    args.num_workers = 1
    args.batch_size = 1
    args.pretrained_unsupervised_novelty_module_path = 'unsupervisednoveltydetection/unsupervised_novelty_detection_module.pth'

    # rank = 1

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    app = TopLevelApp(args.ensemble_path, args.num_subj_cls, args.num_obj_cls, args.num_action_cls, 
        args.data_root, args.csv_path, args.pretrained_unsupervised_novelty_module_path, test_batch_size=100)

    app.run()