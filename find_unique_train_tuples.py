import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from data.data_factory import DataFactory
from utils import custom_collate
import pickle

class Args:
    def __init__(self):
        self.multiporcessing = None
        self.world_size = None
        self.dataset = None
        self.data_root = None
        self.partitions = None
        self.csv_path = None
        self.num_subj_cls = None
        self.num_obj_cls = None
        self.num_action_cls = None
        self.num_workers = None
        self.batch_size = None
        self.num_classes = None
        self.max_object = None
        self.num_iter = None
        self.max_subject = None
        self.box_score_thresh = None
        self.num_iter = None
        self.top_k = None
        self.checkpoint_path = None
        self.ensemble_path = None


if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    args = Args()

    args.world_size = 1
    args.dataset = 'Custom'
    args.data_root = 'Custom'
    args.csv_path = 'Custom/annotations/val_dataset_v1_train.csv'
    args.ensemble_path = 'pretrained'
    args.num_iter = 2
    args.box_score_thresh = 0.0
    args.max_subject = 15
    args.max_object = 15
    args.num_subj_cls = 6
    args.num_obj_cls = 9
    args.num_action_cls = 8
    args.num_workers = 1
    args.batch_size = 1
    args.box_score_thresh = 0.0
    args.num_iter = 2
    args.top_k = 3

    rank = 1

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False

    valset = DataFactory(
        name=args.dataset, partition=None,
        data_root=args.data_root,
        csv_path=args.csv_path,
        training=False,
        num_subj_cls=args.num_subj_cls,
        num_obj_cls=args.num_obj_cls,
        num_action_cls=args.num_action_cls
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset,
            num_replicas=args.world_size,
            rank=rank) if args.multiporcessing else None
    )

    gt_tuples = []
    gt_subjects = []
    gt_objects = []
    gt_verbs = []

    for i, batch in enumerate(val_loader):
        inputs = batch[:-1]
        img_id = inputs[1][0]['img_id']
        inputs[1][0].pop('img_id', None)

        gt_triplet = [batch[-1][0]["subject"].item(), batch[-1][0]["verb"].item(), batch[-1][0]["object"].item()]
        gt_subj = gt_triplet[0]
        gt_obj = gt_triplet[2]
        gt_verb = gt_triplet[1]

        gt_tuples.append(tuple(gt_triplet))
        gt_subjects.append(gt_subj)
        gt_objects.append(gt_obj)
        gt_verbs.append(gt_verb)

    result = {}
    result['tuples'] = list(set(gt_tuples))
    result['subjects'] = list(set(gt_subjects))
    result['objects'] = list(set(gt_objects))
    result['verbs'] = list(set(gt_verbs))

    print(result)

    with open('./ensemble/unique.pkl', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

