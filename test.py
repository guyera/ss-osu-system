import os
import torch
import argparse
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

import pickle
import pocket
import numpy as np

from models.scg import SpatiallyConditionedGraph as SCG
from data.data_factory import DataFactory, CustomInput
from utils import custom_collate


class Test(object):
    def __init__(self, net, model_name):
        self.net = net
        self.func_map = {
            'scg': self.scg,
            'drg': self.drg,
            'idn': self.idn,
            'cascaded-hoi': self.cascaded_hoi,
        }
        self.test = self.func_map[model_name]

    def scg(self, input_data, ov_interaction_map):
        input_data = pocket.ops.relocate_to_cuda(input_data)
        with torch.no_grad():
            output = self.net(*input_data)
            # Batch size is fixed as 1 for inference
            assert len(output) == 1, "Batch size is not 1"
            output = pocket.ops.relocate_to_cpu(output[0])
            # Format detections
            box_idx = output['index']
            # boxes_h = output['boxes_h'][box_idx]
            # boxes_o = output['boxes_o'][box_idx]
            objects = output['object'][box_idx]
            scores = output['scores']
            verbs = output['prediction']
            interactions = torch.tensor([
                ov_interaction_map[o][v]
                for o, v in zip(objects, verbs)
            ])
            # Associate detected pairs with ground truth pairs
            labels = torch.zeros_like(scores)
        return scores, interactions, labels

    def drg(self, input_data):
        raise NotImplementedError

    def idn(self, input_data):
        raise NotImplementedError

    def cascaded_hoi(self, input_data):
        raise NotImplementedError


def main(rank, args):
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = False
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=rank
    )

    valset = DataFactory(
        name=args.dataset, partition=args.partitions[1],
        data_root=args.data_root,
        detection_root=args.detection_dir
    )

    val_loader = DataLoader(
        dataset=valset,
        collate_fn=custom_collate, batch_size=args.batch_size,
        num_workers=args.num_workers, pin_memory=True,
        sampler=DistributedSampler(
            valset,
            num_replicas=args.world_size,
            rank=rank)
    )
    if args.dataset == 'hicodet':
        object_to_target = val_loader.dataset.dataset.object_to_verb
        human_idx = 49
        num_classes = 117
    elif args.dataset == 'vcoco':
        object_to_target = val_loader.dataset.dataset.object_to_action
        human_idx = 1
        num_classes = 24

    # num_anno = torch.tensor(HICODet(None, anno_file=os.path.join(
    #     args.data_root, 'instances_train2015.json')).anno_interaction)
    # rare = torch.nonzero(num_anno < 10).squeeze(1)
    # non_rare = torch.nonzero(num_anno >= 10).squeeze(1)

    net = SCG(
        object_to_target, human_idx, num_classes=num_classes,
        num_iterations=args.num_iter, postprocess=False,
        max_human=args.max_human, max_object=args.max_object,
        box_score_thresh=args.box_score_thresh,
        distributed=True
    )

    if os.path.exists(args.model_path):
        print("Loading model from ", args.model_path)
        checkpoint = torch.load(args.model_path, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
    elif len(args.model_path):
        print("\nWARNING: The given model path does not exist. "
              "Proceed to use a randomly initialised model.\n")

    net.cuda()
    net.eval()
    # Sample input test
    test_input = pickle.load(open('inputs.pkl', 'rb'))
    image = np.array(test_input[0][0].cpu())
    boxes = np.array(test_input[1][0]['boxes'].cpu())
    labels = np.array(test_input[1][0]['labels'].cpu())
    scores = np.array(test_input[1][0]['scores'].cpu())
    # TODO: Pass model_name through args here, also implement conditional calling based on models
    tester = Test(net, 'scg').test
    converter = CustomInput('scg').converter
    input_data = converter(image, boxes, labels, scores)
    tester(input_data, val_loader.dataset.dataset.object_n_verb_to_interaction)
    # timer = pocket.utils.HandyTimer(maxlen=1)

    # with timer:
    #     test_ap = test(net, dataloader)
    # print("Model at epoch: {} | time elapsed: {:.2f}s\n"
    #       "Full: {:.4f}, rare: {:.4f}, non-rare: {:.4f}".format(
    #     epoch, timer[0], test_ap.mean(),
    #     test_ap[rare].mean(), test_ap[non_rare].mean()
    # ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an interaction head")
    parser.add_argument('--world-size', required=True, type=int,
                        help="Number of subprocesses/GPUs to use")
    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument('--partitions', nargs='+', default=['train2015', 'test2015'], type=str)
    parser.add_argument('--data-root', default='hicodet', type=str)
    parser.add_argument('--detection-dir', default='hicodet/detections/test2015',
                        type=str, help="Directory where detection files are stored")
    parser.add_argument('--partition', default='test2015', type=str)
    parser.add_argument('--num-iter', default=2, type=int,
                        help="Number of iterations to run message passing")
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--max-human', default=15, type=int)
    parser.add_argument('--max-object', default=15, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--model-path', default='', type=str)
    parser.add_argument('--batch-size', default=1, type=int,
                        help="Batch size for each subprocess")

    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "8888"

    mp.spawn(main, nprocs=args.world_size, args=(args,))